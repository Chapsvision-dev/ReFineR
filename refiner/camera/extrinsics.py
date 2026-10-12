import torch
from torch import Tensor

from refiner.camera.backends import AxisAngleBackend
from refiner.utils.lie import inverse_transform


class Extrinsics:
    """
    Camera extrinsics class. Coordinate system is by defaul x right, y down, z forward
    """

    def __init__(
        self,
        view_matrix: torch.Tensor,
        rescale_factor: float = 1.0,
        requires_grad: bool = False,
    ):

        self.rescale_factor = rescale_factor
        view_matrix = view_matrix.clone()
        view_matrix[:3, 3] = view_matrix[:3, 3] / self.rescale_factor

        self.inverse_backend = False
        self.backend = AxisAngleBackend(view_matrix, requires_grad=requires_grad)

    @property
    def rescaled_view_matrix(self) -> Tensor:
        mat = self.backend.convert_to_mat()
        return mat

    @property
    def view_matrix(self) -> Tensor:
        mat = self.rescaled_view_matrix
        mat[:3, 3] = mat[:3, 3] * self.rescale_factor
        return mat

    @property
    def cam_center(self) -> Tensor:
        return self.inv_view_matrix[:3, 3]

    @property
    def inv_view_matrix(self):
        mat = self.backend.convert_to_mat()
        mat = inverse_transform(mat.unsqueeze(0)).squeeze()
        return mat

    def world_to_camera(self, points: torch.Tensor):
        points = points / self.rescale_factor
        points = self.backend.apply_transform(points)
        return points

    def camera_to_world(self, points: torch.Tensor, rescale: bool = True):
        points = self.backend.apply_inverse_transform(points)
        if rescale:
            points *= self.rescale_factor
        return points
    
    def camera_to_worl_rot(self, points: torch.Tensor):
        points = self.backend.apply_inverse_rotation(points)
        return points
