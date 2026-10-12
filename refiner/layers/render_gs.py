#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
from typing import List

import torch

from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from refiner.camera.camera import Camera
from refiner.utils.named_tensor import NamedTensor
from refiner.layers.gaussian_model import GaussianModel, Renderer

def compute_depth_channels(
    depth: torch.Tensor, depth_channels: List[str]
) -> NamedTensor:
    imgs, names = [], []
    if "d" in depth_channels:
        imgs.append(depth)
        names.append("d")
        depth = depth.clone()

    if "dn" in depth_channels:
        imgs.append((depth - depth.min()) / (depth.max() - depth.min()))
        names.append("dn")

    return NamedTensor(imgs, names, 0)


def rotate_abs_normals(normals: torch.Tensor, camera: Camera) -> torch.Tensor:
    H, W = normals.shape[1:]
    normals = normals.reshape(3, -1).T  # N x 3
    normals = camera.extrinsics.backend.apply_rotation(normals).T.reshape(3, H, W)
    normals = torch.abs(normals)
    return normals


class RendererGS(Renderer):
    def __init__(self, render_channels: List[str] = []):
            super(RendererGS, self).__init__(render_channels)
            self._max_channels = 3
            self._fixed_features_size = True

    def render(
        self,
        gaussians: GaussianModel,
        camera: Camera
    ) -> NamedTensor:
        """
        Render the scene.
        """
        mask = camera.frustrum_mask(gaussians.get_xyz)

        # Rasterizer
        bg_color = torch.zeros(
            self._max_channels, dtype=gaussians.dtype, device=gaussians.device
        )  # /!\ number of 0 is NUM_CHANNELS (what a pain to debug)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.intrinsics.H),
            image_width=int(camera.intrinsics.W),
            tanfovx=math.tan(camera.intrinsics.fovx * 0.5),
            tanfovy=math.tan(camera.intrinsics.fovy * 0.5),
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.extrinsics.rescaled_view_matrix.transpose(0, 1),
            projmatrix=camera.rescaled_transform_matrix.transpose(0, 1),
            projmatrix_raw=camera.intrinsics.projection_matrix.transpose(
                0, 1
            ),
            sh_degree=0,
            campos=camera.extrinsics.cam_center,
            prefiltered=True,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Rasterization
        means3D = gaussians.get_rescaled_xyz
        screenspace_points = torch.zeros_like(
                means3D, dtype=gaussians.dtype, requires_grad=False, device=gaussians.device
            )
        features = self.prepare_features(gaussians)

        rendered_image, radii, depth, opacity, n_touched, sum_T = rasterizer(
            means3D=means3D[mask],
            means2D=screenspace_points[mask],
            shs=None,
            colors_precomp=features[mask],
            opacities=gaussians.get_opacity[mask],
            scales=gaussians.get_scaling[mask],
            rotations=gaussians.get_rotation[mask],
            cov3D_precomp=None,
            theta=camera.extrinsics.backend.theta,
            rho=camera.extrinsics.backend.rho,
        )
        render_pkg = {
            "render": rendered_image[:len(self.feat_channels)],
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth[0],
            "opacity": opacity[0],
            "n_touched": n_touched,
            "sum_T": sum_T,
        }

        return self.post_process(render_pkg, camera)
