from typing import Tuple, List, Optional

import torch
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import Resize
import numpy as np

from refiner.utils.named_tensor import NamedTensor


def get_cityscape_pallet(inverse: bool = False):
    pallet = {
        0: (0, 0, 0),
        1: (128, 64, 128),
        2: (244, 35, 232),
        3: (70, 70, 70),
        4: (102, 102, 156),
        5: (190, 153, 153),
        6: (153, 153, 153),
        7: (250, 170, 30),
        8: (220, 220, 0),
        9: (107, 142, 35),
        10: (152, 251, 152),
        11: (70, 130, 180),
        12: (220, 20, 60),
        13: (255, 0, 0),
        14: (0, 0, 142),
        15: (0, 0, 70),
        16: (0, 60, 100),
        17: (0, 80, 100),
        18: (0, 0, 230),
        19: (119, 11, 32),
        20: (110, 190, 160),
        21: (170, 120, 50),
        22: (55, 90, 80),
        23: (45, 60, 150),
        24: (157, 234, 50),
        25: (81, 0, 81),
        26: (150, 100, 100),
        27: (230, 150, 140),
        28: (180, 165, 180),
    }
    if inverse:
        return {pallet[k]: k for k in pallet}
    return pallet

# Point cloud online channels
def pc_normals_from_cov(pc: NamedTensor, inc: List[str], outc: List[str]):
    rots = torch.tensor(
        R.from_rotvec(pc[inc].detach().cpu().numpy()).as_matrix(),
        dtype=pc.tensor.dtype,
        device=pc.tensor.device,
    )
    pc[outc] = rots[:, :, 0]
    return pc


def pc_lin_from_cov(pc: NamedTensor, inc: List[str], outc: List[str]):
    pc[outc[0]] = 1 - pc[inc[0]] / pc[inc[1]]
    return pc


def pc_plan_from_cov(pc: NamedTensor, inc: List[str], outc: List[str]):
    pc[outc[0]] = (pc[inc[1]] - pc[inc[0]]) / pc[inc[2]]
    return pc


def pc_sph_from_cov(pc: NamedTensor, inc: List[str], outc: List[str]):
    pc[outc[0]] = pc[inc[0]] / pc[inc[1]]
    return pc


def mean(t: NamedTensor, inc: List[str], outc: List[str]):
    t[outc[0]] = t[inc].mean(dim=t.channel_dim)
    return t


def pc_momo_to_zo(pc: NamedTensor, inc: List[str], outc: List[str]):
    pc[outc] = (pc[inc] + 1) / 2
    return pc


# Point cloud preprocessings
def random_sample(pc: NamedTensor, subsample: float = 1.0) -> torch.Tensor:
    """
    pc is a NamedTensor
    """
    n = pc.tensor.shape[0]
    ind = np.arange(n)
    ind = np.random.choice(ind, int(subsample * n))
    pc.tensor = pc.tensor[ind]
    return pc


def equalize(t: np.ndarray) -> np.ndarray:
    B = 256
    t = (t - t.min()) / (t.max() - t.min())
    t = (t * (B - 1)).astype(np.uint16)
    hist, _ = np.histogram(t, B, [0, B])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * (B - 1) / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")
    t = cdf[t] / (B - 1)
    return t


# Img Online channels
def img_gray_from_rgb(img: NamedTensor, inc: List[str], outc: List[str]) -> NamedTensor:
    img[outc[0]] = torch.mean(img[inc], dim=img.channel_dim)
    return img


def seg_city_to_idx(img: NamedTensor, inc: List[str], outc: List[str]) -> NamedTensor:
    pallet = get_cityscape_pallet(True)
    k1, k2, k3 = 2, 3, 4
    pallet = {k1 * k[0] + k2 * k[1] + k3 * k[2]: pallet[k] for k in pallet}
    values = k1 * img[inc[0]] + k2 * img[inc[1]] + k3 * img[inc[2]]
    values = values.to(torch.int32)

    s = values.shape
    values = values.reshape(-1)
    fil = torch.ones_like(values).to(torch.bool)
    for k in pallet:
        fil[values == k] = False
        values[values == k] = pallet[k]
    values[fil] = 0
    values = values.reshape(s)

    img[outc[0]] = values
    return img


# Image preprocessings
def img_center_crop(img: NamedTensor, crop: Tuple[int], pixels: Optional[torch.Tensor] = None) -> Tuple[NamedTensor, Optional[torch.Tensor]]:
    w, h = img.tensor.shape[1], img.tensor.shape[0]
    [cw, ch] = crop
    cx = (w - cw) // 2
    cy = (h - ch) // 2
    img.tensor = img.tensor[cy : cy + ch, cx : cx + cw]
    if pixels is not None:
        pixels[:, 0] = pixels[:, 0] - cx
        pixels[:, 1] = pixels[:, 1] - cy
    return img, pixels


def img_resize(img: NamedTensor, size: Tuple[int]) -> NamedTensor:
    prev_dim = img.channel_dim
    img.channel_dim = 0
    img.tensor = Resize((size[1], size[0]), antialias=None)(img.tensor)
    img.channel_dim = prev_dim
    return img
