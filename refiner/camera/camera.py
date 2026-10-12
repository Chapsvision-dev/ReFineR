from typing import Tuple, Optional

import torch

from refiner.camera.utils import up_to_homogeneous, down_from_homogeneous
from refiner.camera.extrinsics import Extrinsics
from refiner.camera.intrinsics import Intrinsics
from refiner.camera.backends import IntrinsicsBackend, DistortionBackend


class Camera:
    def __init__(
        self,
        view_matrix: torch.Tensor,
        intr_matrix: torch.Tensor,
        size: Tuple[int],
        dist: Optional[torch.Tensor] = None,

        extr_grad: bool = False,
        intr_grad: bool = False,
        dist_grad: bool = False,

        rescale_factor: float = 1,
        clip_plane: Tuple[float, float] = (1, 100),
        ndc_ranges: Tuple[int, int, int, int, int, int] = (-1, 1, 1, -1, 0, 1),
        viewport: Tuple[int, int, int, int, int, int] = (0, -1, -1, 0, 0, 1),
        intr_model: IntrinsicsBackend.MODEL = IntrinsicsBackend.MODEL.F,
        dist_model: DistortionBackend.MODEL = DistortionBackend.MODEL.DIV
    ):

        self.extrinsics = Extrinsics(
            view_matrix, rescale_factor, extr_grad
        )
        self.intrinsics = Intrinsics(
            intr_matrix, size, clip_plane, ndc_ranges, viewport, rescale_factor, intr_model, intr_grad
        )
        if dist is None:
            dist = torch.zeros(1, device = view_matrix.device, dtype = view_matrix.dtype)
        self.distortion = DistortionBackend(dist, dist_model, dist_grad)

    def clone(self):
        return Camera(
            self.extrinsics.view_matrix,
            self.intrinsics.intrinsics_matrix,
            self.intrinsics.size,
            self.distortion.params,

            self.extrinsics.backend.params.requires_grad,
            self.intrinsics.backend.params.requires_grad,
            self.distortion.params.requires_grad,

            self.extrinsics.rescale_factor,
            self.intrinsics.clip_plane,
            self.intrinsics._ndc_ranges,
            self.intrinsics._viewport,
            self.intrinsics.backend._model,
            self.distortion._model
        )

    @property
    def transform_matrix(self) -> torch.Tensor:
        return self.intrinsics.projection_matrix @ self.extrinsics.view_matrix

    @property
    def rescaled_transform_matrix(self) -> torch.Tensor:
        return self.intrinsics.projection_matrix @ self.extrinsics.rescaled_view_matrix

    @property
    def rays(self) -> Tuple[torch.Tensor, torch.Tensor]:
        xys = self.intrinsics.pixel_coords # H x W x 2
        xys = xys.reshape(-1, 2) # H*W x 2
        xs, ys = xys[..., 0], xys[..., 1]

        dist = 1 / (1 + self.distortion.params * (xs * xs + ys * ys))
        xs = dist * xs
        ys = dist * ys

        lookat: torch.Tensor = torch.stack((xs, ys, torch.ones_like(xs)), axis=-1)
        lookat = lookat.reshape(-1, 3) # (H*W) x 3
        lookat / torch.linalg.norm(lookat, axis=-1, keepdims=True)
        origin = torch.zeros_like(lookat) # (H*W) x 3

        origin = self.extrinsics.camera_to_world(origin, False)
        lookat = self.extrinsics.camera_to_worl_rot(lookat)

        return origin.reshape(self.intrinsics.H, self.intrinsics.W, 3), lookat.reshape(self.intrinsics.H, self.intrinsics.W, 3)

    @property
    def dist_mask(self) -> torch.Tensor:
        xys = self.intrinsics.pixel_coords # H x W x 2
        xs, ys = xys[..., 0], xys[..., 1]
        dist = 1 / (1 + self.distortion.params * (xs * xs + ys * ys)) # H x W
        m, s = 1, 0.6 
        return torch.exp(-(dist - m) ** 2 / (2 * s * s))

        # return ((dist > 0.4)&(dist < 1.6)).detach()
    
    def world_to_image(self, points: torch.Tensor) -> torch.Tensor:
        """
        return x, y, depth
        """
        points = self.extrinsics.world_to_camera(points)
        depth = points[:, [2]]
        points = self.intrinsics.camera_to_image(points) # N x 3
        return torch.concatenate([points[:, :2], depth], dim = 1)

    def world_to_image_filtered(self, points: torch.Tensor) -> torch.Tensor:
        """
        return filtered x, y, depth
        """
        points = self.extrinsics.world_to_camera(points)
        depth = points[:, [2]]

        points = self.intrinsics.camera_to_ndc(points) # N x 3
        mask = self.intrinsics.ndc_filter(points)
        points, depth = points[mask], depth[mask]

        points = self.intrinsics.ndc_to_image(points) # N x 3

        return torch.concatenate([points[:, :2], depth], dim = 1)

    def frustrum_mask(self, points: torch.Tensor) -> torch.Tensor:
        points = up_to_homogeneous(points)
        points = (self.transform_matrix @ points.T).T
        points = down_from_homogeneous(points)
        return self.intrinsics.ndc_filter(points)