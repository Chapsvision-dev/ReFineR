import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import yaml
import json

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from abc import abstractmethod

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from potree_plugin_C import potree_load
except:
    potree_load = None

from refiner.utils.named_tensor import NamedTensor
from refiner.utils.mdag import MDAG
from refiner.camera.pose_utils import (
    random_pose_rot_transl,
    random_pose_pixel_points,
    compute_scale,
)
from refiner.utils.constants import CHANNELS, CHANNELS_GROUPS
from refiner.camera.ready_camera import ready_camera_fov
from refiner.camera.intrinsics import Intrinsics
from refiner.data.ply import write_ply
from refiner.data.utils import (
    img_center_crop,
    img_resize,
    mean,
    pc_lin_from_cov,
    pc_momo_to_zo,
    pc_normals_from_cov,
    pc_sph_from_cov,
    pc_plan_from_cov,
    img_gray_from_rgb,
    seg_city_to_idx,
    equalize,
    random_sample
)

def write_camera(intr: np.ndarray, size: Tuple[int], extr: np.ndarray, out_path: Path):
    camera = {
        "extr": extr.tolist(),
        "intr": intr.tolist(),
        "size": list(size),
    }
    with open(out_path, "w") as f:
        yaml.dump(camera, f)

def read_camera(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_eval(points: np.ndarray, pixels: np.ndarray, size: Tuple[int], out_path: Path):
    out = {
        "points": [{
            "point3d": {
                "x": str(po[0]),
                "y": str(po[1]),
                "z": str(po[2]) 
            },
            "pixelN": {
                "x": str(pi[0] / size[0]),
                "y": str(pi[1] / size[1])
            }
        } for (po, pi) in zip(points, pixels) ]
    }

    with open(out_path, "w") as f:
        json.dump(out, f)

def map_channels(channels, mapping):
    for i, c in enumerate(channels):
        if c in mapping:
            v = mapping[c]
            channels[i] = v
    return channels


@dataclass
class PointImageMeta:
    size: Tuple[int, int]
    intr_init: torch.Tensor
    intr_target: torch.Tensor
    pose_init: torch.Tensor
    rescale_factor: torch.Tensor
    dist: torch.Tensor
    pose_target: Optional[torch.Tensor]
    fn: str


@dataclass
class Frame:
    scene: str
    img_fn: str
    pose: torch.Tensor
    intr: torch.Tensor
    dist: torch.Tensor
    size: Tuple[int, int]

class PointImageDataset(Dataset):
    def __init__(
        self,
        path: str,
        precomp_path: str = "",
        pc_channels: List[str] = [],
        pc_subsample_factor: float = 1.0,
        pc_dynamic_scale: bool = False,
        pc_view_filter: Tuple[int] = (-1, 1, 100),
        pc_lod_filter: float = -1,
        pc_normalize_intensity: bool = True,
        equalize_intensity: bool = False,
        img_channels: List[str] = [],
        img_min_size: int = -1,
        img_center_crop: Tuple[int] = (0, 0),
        absolute_normals: bool = False,
        pert_init: Dict[str, float] = {},
        seed: int = -1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):

        self._process_device = "cpu"

        self.path = Path(path)
        self.precomp_path = Path(precomp_path)
        self.pc_subsample_factor = pc_subsample_factor
        self.pc_dynamic_scale = pc_dynamic_scale
        self.equalize_intensity = equalize_intensity
        self.pc_normalize_intensity = pc_normalize_intensity

        self.img_min_size = img_min_size
        self.img_center_crop = img_center_crop
        self.absolute_normals = absolute_normals

        self.pc_view_filter = pc_view_filter[0] > 0
        if self.pc_view_filter:
            self.filter_delta_fov = pc_view_filter[0] * (torch.pi / 180)
            self.filter_clip = (pc_view_filter[1], pc_view_filter[2])

        self.pc_lod_filter = pc_lod_filter > -1
        self.lod_filter = int(pc_lod_filter)
        if self.pc_lod_filter and not potree_load:
            logger.warning("Missing Potree loader, ignoring")
            self.pc_lod_filter = False  
            self.lod_filter = -1

        self.pert_init = pert_init
        self.rng = np.random.default_rng(seed) if seed >= 0 else None

        self.device = device
        self.dtype = dtype

        # Pc Online preprocessings
        if not hasattr(self, "_pc_online_gray"):
            self._pc_online_gray = True

        self.pc_channels = pc_channels
        if not hasattr(self, "_pc_channels_mapping"):
            self._pc_channels_mapping = {}
        self._pc_full_online_computations = (
            [
                (([CHANNELS.linearity], [CHANNELS.scale_1, CHANNELS.scale_2]), pc_lin_from_cov),
                (([CHANNELS.planarity], [CHANNELS.scale_1, CHANNELS.scale_2, CHANNELS.scale_3]), pc_plan_from_cov),
                (
                    ([CHANNELS.sphericity], [CHANNELS.scale_1, CHANNELS.scale_3]),
                    pc_sph_from_cov,
                ),
                (([CHANNELS.normals_base_mean], CHANNELS_GROUPS.normals_base), mean),
            ]
            + [
                (([c], [ct]), pc_momo_to_zo)
                for c, ct in zip(
                    CHANNELS_GROUPS.normals_pos + [CHANNELS.normals_pos_mean],
                    CHANNELS_GROUPS.normals_base + [CHANNELS.normals_base_mean],
                )
            ]
            + [
                (([c], [ct]), self._pc_preprocess_normals)
                for c, ct in zip(
                    CHANNELS_GROUPS.normals + [CHANNELS.normals_mean],
                    CHANNELS_GROUPS.normals_base + [CHANNELS.normals_base_mean],
                )
            ]
            + [
                (([c], [ct]), self._pc_preprocess_intensity)
                for c, ct in zip([CHANNELS.reflectivity, CHANNELS.gray, CHANNELS.seg] + CHANNELS_GROUPS.colors, [CHANNELS.reflectivity_base, CHANNELS.gray_base, CHANNELS.seg_base] + CHANNELS_GROUPS.colors_base)
            ]
        )

        if self._pc_online_gray:
            self._pc_full_online_computations.append(
                (([CHANNELS.gray_base], CHANNELS_GROUPS.colors_base), mean)
            )
        if self._online_normals:
            self._pc_full_online_computations.append(
                (
                    (CHANNELS_GROUPS.normals_base, [CHANNELS_GROUPS.rot]),
                    pc_normals_from_cov,
                )
            )
        self._pc_full_mdag = MDAG([c[0] for c in self._pc_full_online_computations])
        self._pc_mdag, pc_indices = self._pc_full_mdag.extract_from_output(
            self.pc_channels
        )
        self._pc_online_computations = [
            self._pc_full_online_computations[i] for i in pc_indices
        ]
        self._pc_online_computations = [
            self._pc_online_computations[i] for i in self._pc_mdag.index_edges_orders
        ]

        # Img online preprocessings
        if not hasattr(self, "_img_online_gray"):
            self._img_online_gray = True

        self.img_channels = img_channels
        self._img_full_online_computations = ([
            (([CHANNELS.albedo_mean], CHANNELS_GROUPS.albedo), img_gray_from_rgb),
            (([CHANNELS.material_mean], CHANNELS_GROUPS.material), img_gray_from_rgb),
            (([CHANNELS.seg], [CHANNELS.seg_base]), self._img_preprocess_intensity),
            (([CHANNELS.normals_mean], CHANNELS_GROUPS.normals), mean),
            (([CHANNELS.seg_base], CHANNELS_GROUPS.seg_city), seg_city_to_idx)
        ] 
        + [(([c], [ct]), self._img_preprocess_intensity) for c, ct in zip([CHANNELS.gray] + CHANNELS_GROUPS.colors, [CHANNELS.gray_base] + CHANNELS_GROUPS.colors_base)])
        if self._img_online_gray:
            self._img_full_online_computations.append(
                (([CHANNELS.gray_base], CHANNELS_GROUPS.colors_base), img_gray_from_rgb)
            )

        self._img_full_mdag = MDAG([c[0] for c in self._img_full_online_computations])
        self._img_mdag, img_indices = self._img_full_mdag.extract_from_output(
            self.img_channels
        )
        self._img_online_computations = [
            self._img_full_online_computations[i] for i in img_indices
        ]
        self._img_online_computations = [
            self._img_online_computations[i] for i in self._img_mdag.index_edges_orders
        ]

        self.pcs: List[str] = []
        self.intrs: List[torch.Tensor] = []
        self.imgs: List[str] = []
        self.scenes: List[str] = []
        self.sizes: List[np.ndarray] = []
        self.poses: List[torch.Tensor] = []
        self.dists: List[torch.Tensor] = []
        self.groups: List[int] = []
        self._load_data()
        self._create_frames()

    # Frames creation i.e pairs of LiDAR images
    @property
    def uscenes(self) -> List[str]:
        return list(set(self.scenes))

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _load_data(self):
        pass

    def _get_random_pose(
        self,
        pose_init: torch.Tensor,
        intr: torch.Tensor,
        size: Tuple[int, int],
        eval_pc: Optional[NamedTensor],
    ) -> Tensor:
        se3_rng = (
            np.random.default_rng(self.rng.integers(0, 9999999))
            if self.rng is not None
            else None
        )
        if self.pert_init["r"] and self.pert_init["t"]:
            rand_pose = random_pose_rot_transl(
                self.pert_init["t"], self.pert_init["r"], rng=se3_rng
            )[0]
            rand_pose = rand_pose.to(device = pose_init.device, dtype = pose_init.dtype)
            return rand_pose @ pose_init

        if self.pert_init["p"]:
            if eval_pc is None:
                raise ValueError("Initializing in pixels with no given test points")
            rand_pose = random_pose_pixel_points(
                self.pert_init["p"],
                xyz=eval_pc[CHANNELS_GROUPS.xyz],
                pose=pose_init,
                intr=intr,
                size=size,
                rng=se3_rng,
            )
            rand_pose = rand_pose.to(
                device=pose_init.device, dtype=pose_init.dtype
            )  # N x 4 x 4
            return rand_pose @ pose_init

        return pose_init

    def _get_random_intr(self, intr: torch.Tensor) -> torch.Tensor:
        f = self.pert_init['f'] if 'f' in self.pert_init else 0
        c = self.pert_init['c'] if 'c' in self.pert_init else 0
        fp, cp = np.random.rand(2) - 0.5, np.random.rand(2) - 0.5
        fp, cp = fp / np.linalg.norm(fp) * f, cp / np.linalg.norm(cp) * c
        pert = torch.tensor([
            [fp[0], 0,     cp[0]],
            [0,     fp[1], cp[1]], 
            [0,     0,     0    ]
        ], dtype = intr.dtype, device = intr.device)
        return intr + pert
    
    def _create_frames(self):
        self.frames = [
            Frame(
                intr=intr,
                img_fn=img,
                scene=scene,
                size=(int(size[0]), int(size[1])),
                dist=dist,
                pose=pose,
            )
            for intr, img, size, dist, pose, scene in zip(
                self.intrs,
                self.imgs,
                self.sizes,
                self.dists,
                self.poses,
                self.scenes,
            )
        ]
        logger.info(
            f"Found {len(self.frames)} images over {len(set(self.scenes))} scenes"
        )

    def get_idx_from_name(self, name: str) -> int:
        for i, f in enumerate(self.frames):
            if Path(f.img_fn).stem == name:
                return i
        return -1

    def get_name_from_idx(self, idx: int) -> str:
        return str(Path(self.frames[idx].img_fn).stem)
    
    def get_group_scene(self, idx: int) -> str:
        return self.frames[self.groups[idx][0]].scene
    
    @staticmethod
    def _online_channels(
        pc: NamedTensor,
        online_channels: List[str],
        mdag: MDAG,
        online_computations: List[Tuple],
    ) -> NamedTensor:
        prune_schedule = mdag.prune_schedule_from_output(online_channels)
        for comp, k in zip(online_computations, prune_schedule):
            pc = comp[1](
                pc, comp[0][1], comp[0][0]
            )  # pc and computation inputs, outputs
            if len(k) > 0:
                pc.remove_channels(k)
        return pc

    # Point cloud creation
    def get_online_pc(
        self, scene: str, channels: List[str] = None
    ) -> NamedTensor:
        if channels:
            mdag, indices = self._pc_full_mdag.extract_from_output(channels)
            online_computations = [
                self._pc_full_online_computations[i] for i in indices
            ]
            online_computations = [
                online_computations[i] for i in mdag.index_edges_orders
            ]
            loading_channels = mdag.sources
            loading_channels = map_channels(loading_channels, self._pc_channels_mapping)
        else:
            loading_channels, channels, mdag, online_computations = (
                self.pc_raw_loading_channels,
                self.pc_channels,
                self._pc_mdag,
                self._pc_online_computations,
            )
        pc = self._read_pc(scene, loading_channels)

        rename_mapping = {}
        for k in self._pc_channels_mapping:
            if self._pc_channels_mapping[k] in pc:
                rename_mapping[self._pc_channels_mapping[k]] = k
        pc.rename_channels(rename_mapping)

        return self._online_channels(pc, channels, mdag, online_computations)

    @abstractmethod
    def _read_pc(
        self, scene: str, names: List[str]
    ) -> NamedTensor:
        pass

    @property
    def pc_loading_channels(self):
        return self._pc_mdag.sources

    @property
    def pc_raw_loading_channels(self):
        pc_loading_channels = [c for c in self.pc_loading_channels]
        pc_loading_channels = map_channels(
            pc_loading_channels, self._pc_channels_mapping
        )
        return pc_loading_channels

    def _pc_preprocess_normals(
        self, pc: NamedTensor, inc: List[str], outc: List[str]
    ) -> NamedTensor:
        norms = pc[inc]
        if self.absolute_normals:
            norms = torch.abs(norms)
        pc[outc] = norms
        return pc

    def _pc_intensity_custom(self, intensity: torch.Tensor) -> torch.Tensor:
        ind = intensity != 0
        intensity[ind] = torch.log(intensity[ind] + 1)
        return intensity

    def _pc_preprocess_intensity(
        self, pc: NamedTensor, inc: List[str], outc: List[str]
    ) -> NamedTensor:
        intensity = pc[inc]
        if inc[0] not in [CHANNELS.seg_base, CHANNELS.gray_base] + CHANNELS_GROUPS.colors_base:
            intensity = self._pc_intensity_custom(intensity)

        if self.pc_normalize_intensity:
            if inc[0] in [CHANNELS.gray_base] + CHANNELS_GROUPS.colors_base:
                intensity = intensity / 255
            elif inc[0] in [CHANNELS.seg_base]:
                intensity = intensity / 29
            else:
                intensity = (intensity - intensity.min()) / (
                    intensity.max() - intensity.min()
                )

        if self.equalize_intensity and inc[0] in [CHANNELS.gray_base, CHANNELS.reflectivity_base] + CHANNELS_GROUPS.colors_base:
            intensity = torch.tensor(
                equalize(intensity.detach().cpu().numpy()),
                dtype=self.dtype,
                device=self._process_device,
            )

        pc[outc] = intensity
        return pc

    def _pc_view_filter(
        self,
        pc: NamedTensor,
        pose_init: torch.Tensor,
        intr: torch.Tensor,
        size: Tuple[int, int],
    ) -> Tensor:
        fov = Intrinsics.fov_from_intr_matrix(intr, size)
        filter_fov = (fov[0] + self.filter_delta_fov, fov[1] + self.filter_delta_fov)

        filter_camera = ready_camera_fov(
            filter_fov, size, pose_init, clip_plane=self.filter_clip
        )
        mask = filter_camera.frustrum_mask(pc[CHANNELS_GROUPS.xyz])

        return mask 

    def _pc_lod_filter(
        self,
        pose_init: torch.Tensor,
        intr_init: torch.Tensor, 
        size: Tuple[int, int]
    ) -> Tuple[Tensor, Tensor]:
        
        potree_path  = self.precomp_path / "potree"
        idx, lod = potree_load(
            pose_init.to(torch.float64),
            intr_init.to(torch.float64),
            size[0], 
            size[1],
            str(potree_path), 
            self.lod_filter)
        idx = torch.tensor(idx, dtype = torch.int32)
        lod = torch.tensor(lod, dtype = torch.float32)
        return idx, lod
    
    # Image creation
    def get_online_img(self, ind: int, channels: List[str] = None) -> NamedTensor:
        if channels:
            mdag, indices = self._img_full_mdag.extract_from_output(channels)
            online_computations = [
                self._img_full_online_computations[i] for i in indices
            ]
            online_computations = [
                online_computations[i] for i in mdag.index_edges_orders
            ]
            loading_channels = mdag.sources
        else:
            loading_channels, channels, mdag, online_computations = (
                self.img_raw_loading_channels,
                self.img_channels,
                self._img_mdag,
                self._img_online_computations,
            )

        img = self._read_img(self.frames[ind], loading_channels)
        return self._online_channels(img, channels, mdag, online_computations)

    @abstractmethod
    def _read_img(self, frame: Frame, names: List[str] = None) -> NamedTensor:
        pass

    def _read_img_precomp(
        self, frame: Frame, names: List[str] = None
    ) -> Tuple[List, List]:
        if self.precomp_path == "":
            return [], []

        fn = str(Path(frame.img_fn).stem)
        names = self.img_raw_loading_channels if not names else names
        img, channels = [], []
        if CHANNELS_GROUPS.depth[3] in names:  # dn
            depth_path = self.precomp_path / "depth_npy" / f"{fn}_depth.npy"
            depth = np.expand_dims(np.load(depth_path), -1)
            img.append(depth)
            channels += [CHANNELS_GROUPS.depth[3]]
            logger.info(f"Loading precomputed depth at {depth_path}")

        if any([c in names for c in CHANNELS_GROUPS.normals]):
            normals_path = self.precomp_path / "normals_npy" / f"{fn}_normals.npy"
            normals = np.abs(np.load(normals_path))
            normals = np.moveaxis(normals, 0, -1)
            img.append(normals)
            channels += CHANNELS_GROUPS.normals
            logger.info(f"Loading precomputed normals at {normals_path}")

        if any([c in names for c in CHANNELS_GROUPS.albedo]):
            alb_path = self.precomp_path / "iid_appearance_npy" / f"{fn}_albedo.npy"
            alb = np.load(alb_path)
            img.append(alb)
            channels += CHANNELS_GROUPS.albedo
            logger.info(f"Loading precomputed albedo at {alb_path}")

        if any([c in names for c in CHANNELS_GROUPS.material]):
            mat_path = self.precomp_path / "iid_appearance_npy" / f"{fn}_material.npy"
            mat = np.load(mat_path)
            img.append(mat)
            channels += CHANNELS_GROUPS.material
            logger.info(f"Loading precomputed material at {mat_path}")

        return img, channels

    @property
    def img_loading_channels(self) -> List[str]:
        return self._img_mdag.sources

    @property
    def img_raw_loading_channels(self) -> List[str]:
        img_loading_channels = [c for c in self.img_loading_channels]
        return img_loading_channels

    # Img online channels
    def _img_preprocess_intensity(
        self, img: NamedTensor, inc: List[str], outc: List[str]
    ) -> NamedTensor:
        intensity = img[inc]

        if inc[0] in [CHANNELS.gray_base] + CHANNELS_GROUPS.colors_base:
            intensity = intensity / 255
        elif inc[0] in [CHANNELS.seg_base]:
            intensity = intensity / 29

        if self.equalize_intensity and inc[0] in [CHANNELS.gray_base] + CHANNELS_GROUPS.colors_base:
            W, H = img.tensor.shape[1], img.tensor.shape[0]
            intensity = torch.tensor(
                equalize(intensity.reshape(-1).detach().cpu().numpy()),
                device=self._process_device,
                dtype=self.dtype,
            ).reshape(H, W, 1)

        img[outc] = intensity
        return img

    # Dataset iterator
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ind) -> Tuple[NamedTensor, Optional[NamedTensor], NamedTensor, List[PointImageMeta]]:
        scene_names = [self.frames[g].scene for g in self.groups[ind]]
        if not len(list(set(scene_names))) == 1:
            raise ValueError("All images in the group should be associated to the same scene")
        
        pc = self.get_online_pc(scene_names[0])
        if (self.pc_subsample_factor < 1) and not self.pc_lod_filter:
            pc = random_sample(pc, self.pc_subsample_factor)

        view_mask = torch.zeros(pc.tensor.shape[0], dtype = torch.bool, device = pc.tensor.device) if self.pc_view_filter else torch.ones(pc.tensor.shape[0], dtype = torch.bool, device = pc.tensor.device)
        lod_mask = torch.zeros_like(view_mask) if self.pc_lod_filter else torch.ones_like(view_mask)
        lods = torch.zeros(pc.tensor.shape[0], dtype = torch.float32, device = pc.tensor.device)

        rescale_factor = torch.tensor(1.0, dtype=self.dtype)
        computed_rescale = False
        imgs: List[NamedTensor] = []
        metas: List[PointImageMeta] = []

        has_eval = hasattr(self, "_read_eval")
        eval_pc = None
        
        for img_ind in self.groups[ind]:
            frame: Frame = self.frames[img_ind]
            
            if has_eval:
                # /!\ Per frame test points only keeping last frame for now
                eval_pc = self._read_eval(frame)
                eval_pc.tensor = eval_pc.tensor.to(self.device)
            # Initialization
            pose_init = frame.pose
            pose_target = None
            intr_init, size, dist = frame.intr, frame.size, frame.dist

            if self.pc_view_filter:
                view_mask = view_mask|self._pc_view_filter(pc, pose_init, intr_init, size)

            if self.pc_lod_filter:
                logger.info("Loading Potree data")
                m, lod = self._pc_lod_filter(pose_init, intr_init, size)
                lods[m] = lod
                lod_mask[m] = 1

            # Image and Intrinsics
            img = self.get_online_img(img_ind)
            if self.img_center_crop[0] > 0 and self.img_center_crop[1] > 0:  # Center crop
                img, test_pixels = img_center_crop(img, self.img_center_crop, eval_pc[CHANNELS_GROUPS.img_xy])
                eval_pc[CHANNELS_GROUPS.img_xy] = test_pixels
                intr_init = Intrinsics.center_crop_intrinsics(intr_init, size, self.img_center_crop)
                size = (img.tensor.shape[1], img.tensor.shape[0])
            if self.img_min_size > 0:  # Resize
                factor = max(self.img_min_size / size[0], self.img_min_size / size[1])
                if eval_pc is not None:
                    eval_pc[CHANNELS_GROUPS.img_xy] = eval_pc[CHANNELS_GROUPS.img_xy] * factor
                new_size = (int(size[0] * factor / 2) * 2, int(size[1] * factor / 2) * 2)
                img = img_resize(img, new_size)
                intr_init = Intrinsics.resize_intrinsics(intr_init, size, new_size)
                size = (img.tensor.shape[1], img.tensor.shape[0])

            # To GPU
            img.tensor = img.tensor.to(self.device)
            pose_init = pose_init.to(self.device)
            intr_init = intr_init.to(self.device)
            
            meta = PointImageMeta(
                size=size,
                intr_init=intr_init,
                intr_target = None,
                pose_init=pose_init,
                pose_target=pose_target,
                rescale_factor=None,
                fn = str(Path(frame.img_fn).stem),
                dist = dist,
            )
            imgs.append(img)
            metas.append(meta)

        logger.info(
            f"View filtered {(view_mask == 0).sum().detach().cpu().numpy()} points"
        )
        logger.info(
            f"LoD filtered {((view_mask>0)&(lod_mask == 0)).sum().detach().cpu().numpy()} points"
        )
        if self.pc_lod_filter:
            pc["lod"] = lods

        pc.tensor = pc.tensor[view_mask > 0]
        lod_mask = lod_mask[view_mask > 0]

        pc.tensor = pc.tensor[lod_mask]
        pc.tensor = pc.tensor.to(self.device)

        logger.info(
            f"Remaining {pc.tensor.shape[0]} points"
        )
        
        # Compute poses and dynamic scale
        for meta in metas:
            # Poses
            if self.pert_init:
                meta.pose_target = meta.pose_init
                meta.intr_target = meta.intr_init
                with torch.no_grad():
                    meta.pose_init = self._get_random_pose(meta.pose_init, meta.intr_init, size, eval_pc)
                    meta.intr_init = self._get_random_intr(intr_init)

            # Rescale factor
            if self.pc_dynamic_scale and not computed_rescale:
                rescale_factor = compute_scale(pc[CHANNELS_GROUPS.xyz], pose_init, intr_init, size)
                rescale_factor = rescale_factor.to(self.device)

                computed_rescale = True
                logger.info("Computed rescale factor: {:.2f}".format(rescale_factor))

            meta.rescale_factor = rescale_factor

        return (
            pc,
            eval_pc,
            imgs,
            metas
        )

    def sample_to_common(self, ind: int, common_path: str, multiple_pc: bool = False):
        """
        Only works for datasets that have a single point cloud per scene
        Common
        |- TanksAndTemples
        ||- Meetingroom
        |||- pc.ply
        |||- images
        |||- cameras
        |- Waymo
        | ...

        """
        common_path = Path(common_path)
        scene = self.frames[ind].scene
        out_path = common_path / self.name / scene
        os.makedirs(out_path, exist_ok=True)

        pc, eval, img, meta = self[ind]
        img = img[0]
        meta = meta[0]
        size = (img.tensor.shape[1], img.tensor.shape[0])

        pose_init, intr = meta.pose_init, meta.intr_init
        if multiple_pc:
            pcs_path = out_path / "pcs"
            os.makedirs(pcs_path, exist_ok=True)
            pc_path = pcs_path / f"{ind}.ply"
        else:
            pc_path = out_path / "pc.ply"
        if not os.path.exists(pc_path):
            pc.channel_dim = 1
            dtype_map = {c: np.uint8 for c in CHANNELS_GROUPS.colors}
            names = pc.names
            pc = {c: pc[c].detach().cpu().numpy() for c in names}
            for c in pc:
                if c in dtype_map:
                    pc[c] = pc[c].astype(dtype_map[c])
            write_ply(str(pc_path), [pc[c] for c in names], [c for c in names])

        eval_path = out_path / "eval"
        os.makedirs(eval_path, exist_ok = True)
        write_eval(eval[CHANNELS_GROUPS.xyz].detach().cpu().numpy(), 
                   eval[CHANNELS_GROUPS.img_xy].detach().cpu().numpy(),
                   size, eval_path / f'{meta.fn}.json')
        
        img.channel_dim = 2
        img_path = out_path / "images"
        os.makedirs(img_path, exist_ok=True)
        if self._img_online_gray:
            chan = Image.fromarray(
                (img[CHANNELS_GROUPS.colors] * 255).detach().cpu().numpy().astype(np.uint8)
            )
        else:
            chan = Image.fromarray(
                (img["g"] * 255).detach().cpu().numpy().astype(np.uint8)
            )
        chan.save(str(img_path / f'{meta.fn}.png'))

        if all([c in img.names for c in CHANNELS_GROUPS.seg_city]):
            seg_path = out_path / "seg"
            os.makedirs(seg_path, exist_ok=True)
            chan = Image.fromarray(
                (img[CHANNELS_GROUPS.seg_city]).detach().cpu().numpy().astype(np.uint8)
            )
            chan.save(str(seg_path / f'{meta.fn}.png'))

        cameras_path = out_path / "cameras"
        os.makedirs(cameras_path, exist_ok=True)
        write_camera(intr.detach().cpu().numpy(), size, pose_init.detach().cpu().numpy(), cameras_path / f'{meta.fn}.yaml')