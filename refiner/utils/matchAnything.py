import os
from copy import deepcopy
from typing import Optional
from pathlib import Path

import torch
import cv2
import numpy as np

import poselib
import pyceres
import pycolmap
import pycolmap.cost_functions

from third_party.MatchAnything.src.config.default import get_cfg_defaults
from third_party.MatchAnything.src.utils.misc import lower_config
from third_party.MatchAnything.third_party.ROMA.roma.matchanything_roma_model import MatchAnything_Model

from refiner.layers.gaussian_model import GaussianModel, Renderer
from refiner.layers.render_raster import RendererRaster
from refiner.camera.camera import Camera
from refiner.utils.named_tensor import NamedTensor
from refiner.utils.constants import CHANNELS, CHANNELS_GROUPS

def show_matches(img1, img2, kpts_0, kpts_1, nmatches = 5):
    left_pixels, right_pixels = deepcopy(kpts_0), deepcopy(kpts_1)
    right_pixels[:, 0] += img1.shape[1]

    img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype = img1.dtype)
    img[:img1.shape[0], :img1.shape[1]] = img1
    img[:img2.shape[0], img1.shape[1]:img2.shape[1] + img1.shape[1]] = img2

    rand_indices = np.random.randint(0, len(left_pixels), nmatches)
    for i in range(nmatches):
        lp = left_pixels[rand_indices[i]]
        rp = right_pixels[rand_indices[i]]
        cv2.line(img, (int(lp[0]), int(lp[1])), (int(rp[0]), int(rp[1])), [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)], 2)
    return img

def calibrate_pyceres(points3d, points2d, extr, intr, dist, flags):
    """
    points3d: N x 3
    You can't fit distortion without intrinsics...
    You can't fix principal points...
    flags: (use extr << 0) + (use intr << 1) + (use dist << 2)
    """
    intr = np.array([intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2], dist[0]], dtype = np.float64)
    extr = [pycolmap.Rigid3d(e[:3].astype(np.float64)) for e in extr]

    # Define problem
    prob = pyceres.Problem()
    loss = pyceres.TrivialLoss()
    manifold = pyceres.ProductManifold(pyceres.EigenQuaternionManifold(), pyceres.EuclideanManifold(3))

    if ((flags & 1 << 2) >> 2):
        model = "DIVISION"
    else:
        model = "SIMPLE_PINHOLE"
        intr = intr[[0, 2, 3]]

    for i in range(len(extr)):
        points3di, points2di, extri = points3d[i], points2d[i], extr[i]

        for p2, p3 in zip(points2di, points3di):
            cost = pycolmap.cost_functions.ReprojErrorCost(model, p2)
            params = [
                p3,
                extri.params,
                intr,
            ]
            prob.add_residual_block(cost, loss, params)
            prob.set_parameter_block_constant(p3)

        prob.set_manifold(extri.params, manifold)

        if not ((flags & 1 << 0) >> 0):
            prob.set_parameter_block_constant(extri.params)

    if not ((flags & (1 << 1)) >> 1):
        prob.set_parameter_block_constant(intr)

    # Solve
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout = True
    options.num_threads = -1
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)

    dist = np.array([intr[-1]]) if model == "DIVISION" else np.array([0])
    intr = np.array([
        [intr[0],       0, intr[2]],
        [      0, intr[1], intr[3]],
        [      0,       0,       1]
    ]) if model == "DIVISION" else np.array([
        [intr[0],       0, intr[1]],
        [      0, intr[0], intr[2]],
        [      0,       0,       1]
    ])

    extr = [np.concatenate((e.matrix(), np.array([[0, 0, 0, 1]])), axis = 0) for e in extr]

    return extr, intr, dist

def register_matchAnything(camera: Camera, gaussians: GaussianModel, base_img: NamedTensor, renderer: Optional[Renderer] = None):
    main_cfg_path = "third_party/MatchAnything/configs/models/roma_model.py"
    chkpt = Path(os.getenv("MATCHANYTHING_WEIGHTS_DIR")) / "matchanything_roma.ckpt"

    config =  get_cfg_defaults()
    config.merge_from_file(main_cfg_path)    
    config = lower_config(config)

    model = MatchAnything_Model(config=config['roma'], test_mode=True)
    state_dict = torch.load(chkpt, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.cuda()

    img0 = renderer.render(gaussians, camera)

    pointmap = RendererRaster(CHANNELS_GROUPS.xyz).render_points(gaussians.get_pc, camera)
    pointmap.channel_dim = 2
    pointmap = pointmap[CHANNELS_GROUPS.xyz].detach().cpu().numpy()

    img0 = img0[img0.unary_names(CHANNELS_GROUPS.xyz + [CHANNELS.depth])].mean(dim = img0.channel_dim)
    img0 = img0.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    img1 = base_img[base_img.names].mean(dim = -1).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    images = {
        "image0_rgb": img0,
        "image1_rgb": img1
    }

    matches = model.forward_inference(images)
    kpts0, kpts1 = matches["mkpts0_f"], matches["mkpts1_f"]

    img0 = (img0[0].moveaxis(0, 2).detach().cpu().numpy() * 255).astype(np.uint8)
    img1 = (img1[0].moveaxis(0, 2).detach().cpu().numpy() * 255).astype(np.uint8)

    kpts0 = kpts0.detach().cpu().numpy().astype(np.uint32)
    points2d = kpts1.detach().cpu().numpy().astype(np.float64)
    points3d = pointmap[kpts0[:, 1], kpts0[:, 0]].astype(np.float64)

    fil = np.sum(np.abs(points3d),axis = -1) > 0
    points2d, points3d = points2d[fil], points3d[fil] 
    
    W, H = camera.intrinsics.W, camera.intrinsics.H
    intr = camera.intrinsics.intrinsics_matrix.detach().cpu().numpy()
    camera_dict = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]]}
    pose, _ = poselib.estimate_absolute_pose(points2d, points3d, camera_dict=camera_dict, opt = {'max_reproj_error': 16.0})

    extr = np.eye(4, dtype = np.float32)
    extr[:3] = pose.pose.Rt

    flag  = camera.extrinsics.backend.params.requires_grad << 0
    flag += camera.intrinsics.backend.params.requires_grad << 1
    flag += camera.distortion.params.requires_grad  
    extr, intr, dist = calibrate_pyceres([points3d], [points2d], [extr], intr, np.zeros(2, dtype = np.float32), flag)
    dtype, device = camera.rescaled_transform_matrix.dtype, camera.transform_matrix.device

    extr = torch.tensor(extr[0], dtype = dtype, device = device)
    intr = torch.tensor(intr, dtype = dtype, device = device)
    dist = torch.tensor(dist[[0]], dtype = dtype, device = device)

    return extr, intr, dist