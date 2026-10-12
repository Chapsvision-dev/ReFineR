import matplotlib
matplotlib.use("agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import cv2
from kornia.geometry.epipolar import numeric
import torch

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

from src.utils.plotting import error_colormap, dynamic_alpha
from src.utils.metrics import symmetric_epipolar_distance
from notebooks.notebooks_utils import make_matching_figure

def plot_matches(img0_origin, img1_origin, mkpts0, mkpts1, mconf, vertical, draw_match_type, alpha, save_path, inverse=False, match_error=None, error_thr=5e-3, color_type='error'):
    if inverse:
        img0_origin, img1_origin, mkpts0, mkpts1 = img1_origin, img0_origin, mkpts1, mkpts0
    img0_origin = np.copy(img0_origin) / 255.0
    img1_origin = np.copy(img1_origin) / 255.0
    # Draw
    alpha =dynamic_alpha(len(mkpts0), milestones=[0, 200, 500, 1000, 2000, 4000], alphas=[1.0, 0.5, 0.3, 0.2, 0.15, 0.09])
    if color_type == 'conf':
        color = error_colormap(mconf, thr=None, alpha=alpha)
    elif color_type == 'green':
        mconf = np.ones_like(mconf) * 0.15
        color = error_colormap(mconf, thr=None, alpha=alpha)
    else:
        color = error_colormap(np.zeros((len(mconf),)) if match_error is None else match_error, error_thr, alpha=alpha)

    text = [
        ''
    ]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig = make_matching_figure(img0_origin, img1_origin, mkpts0, mkpts1, color, text=text, path=save_path, vertical=vertical, plot_size_factor=1, draw_match_type=draw_match_type, r_normalize_factor=0.4)

def blend_img(img0, img1, alpha=0.4, save_path=None, blend_method='weighted_sum'):
    img0, img1 = Image.fromarray(np.array(img0)), Image.fromarray(np.array(img1))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Blend:
    if blend_method == 'weighted_sum':
        blended_img = Image.blend(img0, img1, alpha=alpha)
    else:
        raise NotImplementedError

    blended_img.save(save_path)

def warp_img(img0, img1, H, fill_white=False):
    img0 = np.copy(img0).astype(np.uint8)
    img1 = np.copy(img1).astype(np.uint8)
    if fill_white:
        img0_warped = cv2.warpAffine(np.array(img0), H[:2, :], [img1.shape[1], img1.shape[0]], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])
    else:
        img0_warped = cv2.warpAffine(np.array(img0), H[:2, :], [img1.shape[1], img1.shape[0]], flags=cv2.INTER_LINEAR)
    return img0_warped

def warp_img_and_blend(img0_origin, img1_origin, H, save_path, alpha=0.4, inverse=False):
    if inverse:
        img0_origin, img1_origin = img1_origin, img0_origin
        H = np.linalg.inv(H)
    img0_origin = np.copy(img0_origin).astype(np.uint8)
    img1_origin = np.copy(img1_origin).astype(np.uint8)
    
    # Warp
    img0_warpped = Image.fromarray(warp_img(img0_origin, img1_origin, H, fill_white=False))

    # Blend and save:
    blend_img(img0_warpped, Image.fromarray(img1_origin), alpha=alpha, save_path=save_path)

def epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1):
    Tx = numeric.cross_product_matrix(torch.from_numpy(T_0to1)[:3, 3])
    E_mat = Tx @ T_0to1[:3, :3]
    return symmetric_epipolar_distance(torch.from_numpy(mkpts0), torch.from_numpy(mkpts1), E_mat, torch.from_numpy(K0), torch.from_numpy(K1)).numpy()