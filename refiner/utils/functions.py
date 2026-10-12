import os
from typing import Dict, List
import logging
from pathlib import Path
import math

from omegaconf import OmegaConf
import torch
import numpy as np
import matplotlib as mpl
import cv2

from refiner.utils.config_types import LogCfg
from refiner.utils.named_tensor import NamedTensor
from refiner.utils.metrics import rot_error, transl_error, reproj_error
from refiner.utils.constants import CHANNELS, CHANNELS_GROUPS
from refiner.camera.camera import Camera
from refiner.layers.gaussian_model import GaussianModel
from refiner.layers.render_raster import RendererRaster

OmegaConf.register_resolver(
    "extract_0", lambda x: [y["comp"][0] for z in x if x[z] for y in x[z]["channels"]]
)
OmegaConf.register_resolver(
    "extract_1", lambda x: [y["comp"][1] for z in x if x[z] for y in x[z]["channels"]]
)
OmegaConf.register_resolver("add_xyz", lambda x: CHANNELS_GROUPS.xyz + x)
OmegaConf.register_resolver("unique", lambda x: list(set(x)))
OmegaConf.register_resolver(
    "keep_synth", lambda x: [y for y in x if y in CHANNELS_GROUPS.depth]
)
OmegaConf.register_resolver(
    "filter_synth", lambda x: [y for y in x if y not in CHANNELS_GROUPS.depth]
)
OmegaConf.register_resolver(
    "viz_chan",
    lambda x: ([y for y in [CHANNELS.gray, CHANNELS.reflectivity, CHANNELS.normals_mean] + CHANNELS_GROUPS.normals if y in x] + [""])[
        0
    ],
)
OmegaConf.register_resolver("get_clip_plane", lambda x: [x[1], x[2]])


def apply_cmap(img: np.ndarray, cmap: str = "Greys_r") -> np.ndarray:
    """
    See https://matplotlib.org/stable/users/explain/colors/colormaps.html for colormap options
    """
    istorch = isinstance(img, torch.Tensor)
    if istorch:
        device, dtype = img.device, img.dtype
        img = img.detach().cpu().numpy()
    cmap = mpl.colormaps[cmap]
    shapes = list(img.shape)
    img = img.reshape(-1)
    img = (cmap(img)[:, :3] * 255).astype(np.uint8)
    img = img.reshape(*shapes, 3)

    if istorch:
        img = torch.tensor(img, dtype=dtype, device=device)
    return img


def fuse_images(front_img: torch.Tensor, back_img: torch.Tensor, cmap: str = "hsv"):
    front_indices = torch.where(front_img != 0)

    front_img = (front_img - front_img.min()) / (front_img.max() - front_img.min())
    back_img = (back_img - back_img.min()) / (back_img.max() - back_img.min())

    if len(back_img.shape) < 3:
        back_img = torch.stack((back_img, back_img, back_img), dim=2)  # H x W x 3
    elif back_img.shape[2] == 1:
        back_img = torch.concatenate((back_img, back_img, back_img), dim=2)
    back_img[front_indices[0], front_indices[1]] = (
        0.5 * back_img[front_indices[0], front_indices[1]]
        + 0.5 * apply_cmap(front_img[front_indices[0], front_indices[1]], cmap) / 255
    )

    return back_img


def get_fused_img(
    img: NamedTensor, gaussians: GaussianModel, camera: Camera
) -> torch.Tensor:
    synth = RendererRaster([CHANNELS.depth]).render(gaussians, camera)
    synth = fuse_images(synth[CHANNELS.depth], img[img.names[0]])
    synth = (synth * 255).to(torch.uint8)
    return synth

def get_match_img(
    img: NamedTensor, points: torch.Tensor, pixels: torch.Tensor, camera: Camera
) -> torch.Tensor:

    proj = camera.world_to_image_filtered(points)
    
    proj = proj[:, :2].to(torch.int32).detach().cpu().numpy()
    pixels = pixels.to(torch.int32).detach().cpu().numpy()
    img = (img[img.names[0]] * 255).to(torch.uint8)
    img = img.unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()

    s = 15
    for po, pi in zip(proj, pixels):
        img = cv2.line(img, po - np.array([s, 0]), po + np.array([s, 0]), [255, 0, 0], thickness=1)
        img = cv2.line(img, po - np.array([0, s]), po + np.array([0, s]), [255, 0, 0], thickness=1)

        img = cv2.line(img, pi - np.array([s, s]), pi + np.array([s, s]), [0, 255, 0], thickness=1)
        img = cv2.line(img, pi - np.array([-s, s]), pi + np.array([-s, s]), [0, 255, 0], thickness=1)
    return torch.tensor(img)

def get_steps(camera: Camera) -> Dict[str, torch.Tensor]:
    extr_step = torch.linalg.norm(camera.extrinsics.backend.tau) / math.sqrt(len(camera.extrinsics.backend.tau))
    intr_step = torch.linalg.norm(camera.intrinsics.backend.params_step) / math.sqrt(len(camera.intrinsics.backend.params))
    dist_step = torch.linalg.norm(camera.distortion.params_step) / math.sqrt(len(camera.distortion.params)) / 25
    return {"extr_step": extr_step, "intr_step": intr_step, "dist_step": dist_step}


def get_errors(
    camera: Camera, target_camera: Camera, eval_points: NamedTensor
) -> Dict[str, torch.Tensor]:
    R_cam = camera.extrinsics.view_matrix[:3, :3]
    R_tar = target_camera.extrinsics.view_matrix[:3, :3]
    rot = rot_error(R_cam, R_tar)
    R_diff = R_cam @ R_tar.T

    t_new = R_diff @ target_camera.extrinsics.view_matrix[:3, 3].unsqueeze(-1)
    t_new = t_new[:, 0]  # 3 x 3 @ 3 x 1 = 3 x 1
    transl = transl_error(camera.extrinsics.view_matrix[:3, 3], t_new)

    t_scaled_new = R_diff @ target_camera.extrinsics.rescaled_view_matrix[:3, 3]
    t_scaled_new = t_scaled_new.unsqueeze(-1)
    t_scaled_new = t_scaled_new[:, 0]  # 3 x 3 @ 3 x 1 = 3 x 1
    transl_rescaled = transl_error(
        camera.extrinsics.rescaled_view_matrix[:3, 3], t_scaled_new
    )

    pix = reproj_error(camera, eval_points[CHANNELS_GROUPS.xyz], eval_points[CHANNELS_GROUPS.img_xy])

    focal = transl_error(camera.intrinsics.f, target_camera.intrinsics.f)
    center = transl_error(camera.intrinsics.c, target_camera.intrinsics.c)

    return {
        "rot_err": rot,
        "transl_err": transl,
        "transl_rescaled_err": transl_rescaled,
        "pix_err": pix,
        "focal_err": focal, 
        "center_point_err": center,
    }


def check_convergence(i: int, steps_list: List[Dict], cfg: Dict[str, float]) -> bool:
    logger = logging.getLogger(f"{__name__}.check_convergence")
    iter_params = cfg
    converged = True
    if len(steps_list) < iter_params[1]:
        converged = False
    for step in steps_list:
        for s in list(step.values()):
            converged = converged and (s < iter_params[0])
    max_iter = i > iter_params[2]
    if converged:
        logger.info(f"Algorithm converged")
    if max_iter:
        logger.info(f"Reached maximum number of iterations")
    stop = converged or max_iter
    return stop


def update_logger_handlers(cfg: LogCfg, path: Path):
    logger = logging.getLogger("root")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.handlers.clear()
    if cfg.log_to_file:
        os.makedirs(path, exist_ok=True)
        h = logging.FileHandler(f'{path / "log.txt"}')
        h.setFormatter(formatter)
        logger.addHandler(h)
    if cfg.log_to_console:
        h = logging.StreamHandler()
        h.setFormatter(formatter)
        logger.addHandler(h)
