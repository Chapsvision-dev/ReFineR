from typing import List

import torch
from torch import Tensor
import numpy as np

from refiner.camera.camera import Camera

def threshs_errors(pixs: List[float], values: List[float]) -> List[float]:
    threshs = []
    pixs = np.array(pixs)
    for v in values:
        threshs.append((pixs < v).astype(np.float32).mean() * 100)
    return threshs


def rot_error(final_rot: Tensor, target_rot: Tensor) -> torch.Tensor:
    theta = torch.trace((final_rot @ target_rot.T)) % (2 * torch.pi)
    ad = torch.abs(3 - theta)
    if ad > 1e-1:
        return torch.arccos((theta - 1) / 2)
    else:
        return torch.sqrt(ad) + 1 / 24 * torch.pow(ad, 3 / 2)  # AD for theta going to 3


def transl_error(final_transl: Tensor, target_transl: Tensor) -> torch.Tensor:
    return torch.linalg.norm(final_transl - target_transl)


def reproj_error(
    cam: Camera, points: torch.Tensor, pixels: torch.Tensor  # N x (x, y, z, +)
) -> torch.Tensor:
    points = cam.world_to_image(points)  # N x x, y, d 
    points = points[:, :2]

    err = pixels - points
    err = err * err
    err = torch.sqrt(err.sum(dim = -1).mean())

    return err
