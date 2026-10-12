from typing import Tuple, Union, Optional
from torch import Tensor

from refiner.camera.camera import Camera
from refiner.camera.intrinsics import Intrinsics


def ready_camera(intr: Tensor, size: Tuple[int], extr: Tensor, dist: Optional[Tensor] = None, **kwargs) -> Camera:
    """
    size: W, H
    """
    cam_cfg = {
        "intr_matrix": intr,
        "clip_plane": (1, 100),
        "size": size,
        "dist": dist,
        "ndc_ranges": (-1, 1, 1, -1, 0, 1),
        "viewport": (0, -1, -1, 0, 0, 1),
        "view_matrix": extr,
        "extr_grad": False,
        "intr_grad": False,
        "dist_grad": False
    }
    for k in kwargs:
        cam_cfg[k] = kwargs[k]
    return Camera(**cam_cfg)


def ready_camera_fov(
    fov: Union[Tensor, Tuple[Tensor]], size: Tuple[int], extr: Tensor, **kwargs
) -> Camera:
    intr = Intrinsics.intr_matrix_from_fov(fov, size).to(
        dtype=extr.dtype, device=extr.device
    )
    cam_cfg = {
        "intr_matrix": intr,
        "clip_plane": (1, 25),
        "size": size,
        "ndc_ranges": (-1, 1, 1, -1, 0, 1),
        "viewport": (0, -1, -1, 0, 0, 1),
        "view_matrix": extr,
        "extr_grad": False,
        "intr_grad": False,
    }
    for k in kwargs:
        cam_cfg[k] = kwargs[k]
    return Camera(**cam_cfg)
