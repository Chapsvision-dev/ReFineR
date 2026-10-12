from typing import Tuple, List, Union
import math

import torch

from refiner.camera.utils import up_to_homogeneous, down_from_homogeneous
from refiner.camera.backends import IntrinsicsBackend

class Intrinsics:
    def __init__(
        self,
        intr_matrix: torch.Tensor,
        size: Tuple[int, int],
        clip_plane: Tuple[float, float] = (1, 100),
        ndc_ranges: Tuple[int, int, int, int, int] = (-1, 1, 1, -1, 0, 1),
        viewport: Tuple[int] = (0, -1, -1, 0, 0, 1),
        rescale_factor: float = 1,
        model: IntrinsicsBackend.MODEL = IntrinsicsBackend.MODEL.F,
        requires_grad: bool = False,
    ):
        self._W, self._H = size
        D = math.sqrt(self._W ** 2 + self._H **2)
        self.backend = IntrinsicsBackend(intr_matrix, model, D, requires_grad)
        self.resize_factor = 1.
        self._ndc_ranges = ndc_ranges
        
        self.rescale_factor = rescale_factor
        self.near, self.far = (
            clip_plane[0] / self.rescale_factor,
            clip_plane[1] / self.rescale_factor,
        )

        self._viewport = viewport
        if any([self._viewport[i] == -1 and self._viewport[i + 1] == -1 for i in [0, 2]]):
            raise ValueError(
                f"Two values of the same dimension are given as -1 in viewport {self._viewport}"
            )

    @property
    def ndc_ranges(self):
        return tuple(
            [(self._ndc_ranges[i], self._ndc_ranges[i + 1]) for i in range(0, len(self._ndc_ranges), 2)]
        )
    @staticmethod
    def center_crop_intrinsics(
        intr: torch.Tensor, size: Tuple[int, int], crop: Tuple[int, int]
    ) -> torch.Tensor:
        intr = torch.clone(intr)
        cxcy = (
            torch.tensor(
                [size[0] - crop[0], size[1] - crop[1]],
                dtype=intr.dtype,
                device=intr.device,
            )
            // 2
        )
        intr[:2, 2] = intr[:2, 2] - cxcy
        return intr

    @staticmethod
    def resize_intrinsics(
        intr: torch.Tensor, size: Tuple[int, int], new_size: Tuple[int, int]
    ) -> torch.Tensor:
        intr = torch.clone(intr)
        factors = torch.tensor(
            [new_size[0] / size[0], new_size[1] / size[1]],
            dtype=intr.dtype,
            device=intr.device,
        )  # 2
        intr[:2] = intr[:2] * factors.unsqueeze(1)
        return intr

    @staticmethod
    def intr_matrix_from_fov(
        fov: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], size: Tuple[int, int]
    ) -> torch.Tensor:
        if not isinstance(fov, tuple):
            fov = (fov, fov)

        fx = size[0] / (2 * torch.tan(fov[0] / 2))
        fy = size[1] / (2 * torch.tan(fov[1] / 2))

        # fmt: off
        intr_matrix = torch.tensor(
            [
                [fx,  0, size[0] / 2],
                [ 0, fy, size[1] / 2],
                [ 0,  0,           1]
            ], 
            dtype=fov[0].dtype,
            device=fov[0].device)
        # fmt: on
        return intr_matrix

    @staticmethod
    def fov_from_intr_matrix(intr: torch.Tensor, size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        fx, fy, W, H = intr[0, 0], intr[1, 1], size[0], size[1]
        return (2 * torch.arctan(W / (2 * fx)), 2 * torch.arctan(H / (2 * fy)))

    @property
    def viewport(self) -> List[int]:
        viewport = [v for v in self._viewport]
        for i, v in zip(
            list(range(len(self._viewport))), [self.W, self.W, self.H, self.H]
        ):
            if self._viewport[i] == -1:
                viewport[i] = v
        return viewport
    
    @property
    def H(self) -> int: 
        return int(self._H * self.resize_factor)

    @property
    def W(self) -> int: 
        return int(self._W * self.resize_factor)

    @property 
    def D(self) -> int:
        return math.sqrt(self.H ** 2 + self.W ** 2)
    
    @property
    def size(self) -> Tuple[int, int]:
        return self.W, self.H

    @property
    def fx(self) -> torch.Tensor:
        return self.backend.fx * self.resize_factor

    @property
    def fy(self) -> torch.Tensor:
        return self.backend.fy * self.resize_factor

    @property
    def f(self) -> torch.Tensor:
        return torch.stack([self.fx, self.fy])

    @property
    def cx(self) -> torch.Tensor:
        return self.backend.cx * self.resize_factor

    @property
    def cy(self) -> torch.Tensor:
        return self.backend.cy * self.resize_factor

    @property
    def c(self) -> torch.Tensor:
        return torch.stack([self.cx, self.cy])
    
    @property
    def px(self) -> torch.Tensor:
        c = (
            torch.tensor(
                [self.W, self.H], device=self.fx.device, dtype=self.fx.dtype
            )
            / 2
        )
        p = self.c * self.resize_factor - c  # 2 x 2 @ 2 x 1 = 2 x 1
        return p[0]

    @property
    def py(self) -> torch.Tensor:
        c = (
            torch.tensor(
                [self.W, self.H], device=self.fx.device, dtype=self.fx.dtype
            )
            / 2
        )
        p = self.c * self.resize_factor - c  # 2 x 2 @ 2 x 1 = 2 x 1
        return p[1]

    @property
    def fovx(self) -> torch.Tensor:
        return self.fov_from_intr_matrix(self.intrinsics_matrix, self.size)[0]

    @property
    def fovy(self) -> torch.Tensor:
        return self.fov_from_intr_matrix(self.intrinsics_matrix, self.size)[1]

    @property
    def clip_plane(self) -> Tuple[float, float]:
        return (self.near, self.far)

    @property
    def intrinsics_matrix(self) -> torch.Tensor:

        z = torch.zeros_like(self.fx)
        o = torch.ones_like(self.fx)
        # fmt: off
        return torch.stack(
            [
                torch.stack([self.fx,       z, self.cx], dim = 0),
                torch.stack([      z, self.fy, self.cy], dim = 0),
                torch.stack([      z,       z,       o], dim = 0)
            ],
            dim = 0
        )
        #fmt: on

    @property
    def perspective_matrix(self) -> torch.Tensor:
        
        z = torch.zeros_like(self.fx)
        o = torch.ones_like(self.fx)
        # fmt: off
        persp_mat = torch.stack(
            [
                torch.stack([self.fx,       z, self.px, z], dim = 0),
                torch.stack([      z, self.fy, self.py, z], dim = 0),
                torch.stack([      z,       z,       o, z], dim = 0),
                torch.stack([      z,       z,       z, o], dim = 0)
            ],
            dim = 0
        )
        # fmt: on
        return persp_mat

    def ndc_matrix(self, left, right, bottom, top, near, far) -> torch.Tensor:
        ax = (self.ndc_ranges[0][1] - self.ndc_ranges[0][0]) * near / (right - left)
        bx = (
            1
            / self.rescale_factor
            * (right * self.ndc_ranges[0][0] - left * self.ndc_ranges[0][1])
            / (right - left)
        )

        ay = (self.ndc_ranges[1][1] - self.ndc_ranges[1][0]) * near / (top - bottom)
        by = (
            1
            / self.rescale_factor
            * (top * self.ndc_ranges[1][0] - bottom * self.ndc_ranges[1][1])
            / (top - bottom)
        )

        az = (
            1
            / self.rescale_factor**2
            * (self.ndc_ranges[2][0] - self.ndc_ranges[2][1])
            * near
            * far
            / (far - near)
        )
        bz = (
            1
            / self.rescale_factor
            * -(self.ndc_ranges[2][0] * near - self.ndc_ranges[2][1] * far)
            / (far - near)
        )

        device, dtype = self.fx.device, self.fx.dtype

        # fmt: off
        ndc_mat = torch.tensor(
            [
                [ax,  0, bx,  0],
                [ 0, ay, by,  0],
                [ 0,  0, bz, az],
                [ 0,  0, 1,   0]
            ],
            device=device,
            dtype=dtype
        )
        # fmt: on
        return ndc_mat

    @property
    def projection_matrix(self) -> torch.Tensor:
        top = -self.H / 2 * self.near
        bottom = -top
        right = self.W / 2 * self.near
        left = -right
        ndc = self.ndc_matrix(left, right, bottom, top, self.near, self.far)
        proj = ndc @ self.perspective_matrix
        return proj

    @property
    def viewport_matrix(self) -> torch.Tensor:
        vl = float(self.viewport[0])
        vr = float(self.viewport[1])
        vb = float(self.viewport[2])
        vt = float(self.viewport[3])
        min_depth = float(self.viewport[4])
        max_depth = float(self.viewport[5])

        vw = self.ndc_ranges[0][1] - self.ndc_ranges[0][0]
        vxx = (vr - vl) / vw
        vxt = (self.ndc_ranges[0][1] * vl - self.ndc_ranges[0][0] * vr) / vw

        vh = self.ndc_ranges[1][1] - self.ndc_ranges[1][0]
        vyy = (vt - vb) / vh
        vyt = (self.ndc_ranges[1][1] * vb - self.ndc_ranges[1][0] * vt) / vh

        vd = self.ndc_ranges[2][1] - self.ndc_ranges[2][0]
        vzz = (max_depth - min_depth) / vd
        vzt = (
            self.ndc_ranges[2][1] * min_depth - self.ndc_ranges[2][0] * max_depth
        ) / vd

        device, dtype = self.fx.device, self.fx.dtype

        # fmt: off
        vp_mat = torch.tensor(
            [
                [vxx,   0,   0, vxt], 
                [  0, vyy,   0, vyt],
                [  0,   0, vzz, vzt],
                [  0,   0,   0,   1]
            ],
            device=device,
            dtype=dtype
        )
        # fmt: on
        return vp_mat

    @property
    def pixel_coords(self):
        device, dtype = self.backend.params.device, self.backend.params.dtype
        xs = torch.tile(torch.arange(self.W), (self.H,)) # H*W
        ys = torch.arange(self.H).repeat_interleave(self.W) # H*W
        xs = xs.to(dtype = dtype, device = device)
        ys = ys.to(dtype = dtype, device = device)

        xs = ((xs + 0.5) - self.cx) / self.fx
        ys = ((ys + 0.5) - self.cy) / self.fy
        xys = torch.stack([xs, ys], dim = -1) # H*W x 2
        xys = xys.reshape(self.H, self.W, 2) # H x W x 2
        return xys
        
    def camera_to_clip(self, points: torch.Tensor) -> torch.Tensor:
        points = up_to_homogeneous(points)  # N x 4
        points = (self.projection_matrix @ points.T).T  # N x 4
        return points  # N x 4

    def camera_to_ndc(self, points: torch.Tensor) -> torch.Tensor:
        points = self.camera_to_clip(points)  # N x 4
        points = down_from_homogeneous(points)  # N x 3
        return points  # N x 3

    def ndc_to_image(self, points: torch.Tensor) -> torch.Tensor:
        points = up_to_homogeneous(points)  # N x 4
        points = (self.viewport_matrix @ points.T).T  # N x 4
        points = down_from_homogeneous(points)  # N x 3
        return points  # N x 3

    def camera_to_image(self, points: torch.Tensor) -> torch.Tensor:
        points = up_to_homogeneous(points)
        mat = self.viewport_matrix @ self.projection_matrix
        points = (mat @ points.T).T
        points = down_from_homogeneous(points)
        return points # N x 2
    
    def image_to_ndc(self, points: torch.Tensor) -> torch.Tensor:
        points = up_to_homogeneous(points)  # N x 4
        points = (torch.linalg.inv(self.viewport_matrix) @ points.T).T  # N x 4
        points = down_from_homogeneous(points)  # N x 3
        return points  # N x 3

    def ndc_filter(self, points: torch.Tensor) -> torch.Tensor:
        filter = torch.ones(points.shape[0], device=points.device, dtype=torch.float32)
        for i in range(3):
            if self.ndc_ranges[i][1] > self.ndc_ranges[i][0]:
                filter = (
                    filter
                    * (points[:, i] > self.ndc_ranges[i][0])
                    * (points[:, i] < self.ndc_ranges[i][1])
                )
            else:
                filter = (
                    filter
                    * (points[:, i] < self.ndc_ranges[i][0])
                    * (points[:, i] > self.ndc_ranges[i][1])
                )
        return filter > 0
