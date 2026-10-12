import os
import yaml
from typing import List
import logging
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import torch
from PIL import Image

from refiner.data.point_image_dataset import PointImageDataset, Frame
from refiner.data.ply import read_ply
from refiner.utils.named_tensor import NamedTensor
from refiner.utils.constants import CHANNELS, CHANNELS_GROUPS

class Common(PointImageDataset):
    def __init__(self, set: str, scene: str = "", **kwargs):
        self._scene = scene
        self.set = set

        self._online_normals = False
        self._pc_online_gray = True
        self._pc_channels_mapping = {
            CHANNELS.reflectivity_base: "r", 
            CHANNELS.normals_base_x: "nx", 
            CHANNELS.normals_base_y: "ny", 
            CHANNELS.normals_base_z: "nz", 
            CHANNELS.red_base: "red", 
            CHANNELS.green_base: "green", 
            CHANNELS.blue_base: "blue",
            CHANNELS.seg_base: "seg"}

        super(Common, self).__init__(**kwargs)

    @property
    def scene(self) -> str:
        return self._scene

    @property
    def name(self) -> str:
        return "Common"

    def _read_pc(
        self, scene: str, names: List[str]
    ) -> NamedTensor:
        scene_path = self.path / self.set / scene

        # Loading LiDAR point cloud
        pc_path = scene_path / "pc.ply"
        pc = read_ply(pc_path)
        pc = np.stack([pc[f] for f in names], axis=1)

        pc = torch.tensor(pc, device=self._process_device, dtype=self.dtype)
        pc = NamedTensor(pc, names, channel_dim=1)

        logger.info(f"Loaded point cloud {pc_path}")
        return pc

    def _read_eval(self, frame: Frame) -> torch.Tensor:
        eval_path = self.path / self.set / frame.scene / "eval" / f"{frame.img_fn.split('.')[0]}.json"

        with open(eval_path, "r") as f:
            file = json.load(f)

        matches = file["points"]
        matches = [p for p in matches if ((not "set" in p) or ("set" in p and p["set"] == "fit"))]
        points = [np.array([float(p["point3d"][i]) for i in CHANNELS_GROUPS.xyz]) for p in matches]
        pixels = [np.array([float(p["pixelN"][i]) for i in [CHANNELS.x, CHANNELS.y]]) for p in matches]

        points = np.stack(points, axis = 0)
        points = torch.tensor(points, device = self._process_device, dtype = self.dtype)

        pixels = np.stack(pixels, axis = 0)
        for i in [0, 1]:
            pixels[:, i] = pixels[:, i] * frame.size[i]
        pixels = torch.tensor(pixels, device = self._process_device, dtype = self.dtype)

        eval_pc = [points, pixels]
        names = CHANNELS_GROUPS.xyz + CHANNELS_GROUPS.img_xy

        if "channels" in matches[0]:
            chan_names = list(matches[0]["channels"].keys())
            channels = [np.array([float(p["channels"][n]) for n in chan_names]) for p in matches]
            channels = np.stack(channels, axis = 0)
            channels = torch.tensor(channels, device = self._process_device, dtype = self.dtype)
            eval_pc.append(channels)
            names += chan_names

        eval_pc = torch.concatenate(eval_pc, dim = 1)
        eval_pc = NamedTensor(eval_pc, names, 1)
        return eval_pc
    
    def _load_data(self):
        set_path = self.path / self.set

        # Loading Camera data
        (
            self.poses,
            self.intrs,
            self.sizes,
            self.imgs,
            self.dists,
            self.scenes,
            self.groups,
        ) = ([], [], [], [], [], [], [])
        scenes = os.listdir(set_path)
        scenes.sort()

        group_idx = {}
        pidx = idx = 0
        for scene in scenes:
            if self.scene != "" and self.scene != scene:
                continue
            scene = str(scene)
            scene_path = set_path / scene
            img_path = scene_path / "images"
            imgs = os.listdir(img_path)
            imgs.sort()
            pidx = idx
            for img in imgs:
                img_idx = img.split(".")[0]
                group_idx[img_idx] = idx
                idx += 1
                cam_path = scene_path / "cameras" / f"{img_idx}.yaml"
                with open(cam_path, "r") as f:
                    cam = yaml.safe_load(f)
                self.poses.append(cam["extr"])
                self.intrs.append(cam["intr"])
                self.sizes.append(cam["size"])
                self.dists.append([0, 0, 0, 0])
                self.imgs.append(img)
                self.scenes.append(scene)
            
            group_path = scene_path / "groups"
            if os.path.exists(group_path):
                groups = os.listdir(group_path)
                for g in groups:
                    with open(group_path / g, 'r') as f:
                        group = [group_idx[i.strip()] for i in f.readlines()]
                        self.groups.append(group)
            else:
                self.groups += [[i] for i in range(pidx, idx)]

        self.poses = torch.tensor(
            np.stack(self.poses, axis=0), device=self._process_device, dtype=self.dtype
        )
        self.intrs = torch.tensor(
            np.stack(self.intrs, axis=0), device=self._process_device, dtype=self.dtype
        )
        self.dists = torch.tensor(
            np.stack(self.dists, axis=0), device=self._process_device, dtype=self.dtype
        )
        self.sizes = np.stack(self.sizes, axis=0)

    def _read_img(self, frame: Frame, names: List[str]) -> NamedTensor:
        names = self.img_raw_loading_channels if not names else names
        scene_path = self.path / self.set / frame.scene
        fn = scene_path / "images" / frame.img_fn

        img, channels = self._read_img_precomp(frame, names)

        if any([c in names for c in CHANNELS_GROUPS.colors_base]):
            col_img = Image.open(fn)
            col_img = np.array(col_img)
            img.append(col_img)
            channels += [c for c in CHANNELS_GROUPS.colors_base]

        if any([c in names for c in CHANNELS_GROUPS.seg_city]):
            img.append(np.array(Image.open(scene_path / "seg" / frame.img_fn)))
            channels += [c for c in CHANNELS_GROUPS.seg_city]

        img = np.concatenate(img, axis=2)
        img = torch.tensor(img, device=self._process_device, dtype=self.dtype)
        img = NamedTensor(img, channels, 2)
        img = img.subtensor(names)
        logger.info(f"Loaded image {fn}")
        return img
