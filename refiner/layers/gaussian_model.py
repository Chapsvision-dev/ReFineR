from abc import ABC, abstractmethod
from typing import List, Dict

import torch

from simple_knn._C import distCUDA2

from refiner.camera.camera import Camera
from refiner.utils.named_tensor import NamedTensor
from refiner.data.point_image_dataset import CHANNELS, CHANNELS_GROUPS

def rotate_abs_normals(normals: torch.Tensor, camera: Camera) -> torch.Tensor:
    H, W = normals.shape[1:]
    normals = normals.reshape(3, -1).T  # N x 3
    normals = camera.extrinsics.backend.apply_rotation(normals).T.reshape(3, H, W)
    normals = torch.abs(normals)
    return normals

class GaussianModel:
    def __init__(
        self,
        pc: NamedTensor,
        point_size: float,
        opacity: float,
        mip_size: float = 2.,
        rescale_factor: float = 1,
    ):
        self.rescale_factor = rescale_factor

        self.feat_channels = pc.unary_names(CHANNELS_GROUPS.xyz)
        self.viz_channel = None

        points, self._features = pc[CHANNELS_GROUPS.xyz], pc[self.feat_channels]
        self.device, self.dtype = points.device, points.dtype

        # Points
        points = points / self.rescale_factor
        self._xyz = points
        
        # Features
        npoints, self._nfeatures = points.shape[0], self._features.shape[1]

        # Neighbors
        self.point_size = point_size
        self.mip_size = mip_size
        self.reset_sizes()

        # Opacities
        self._opacity = opacity * torch.ones((npoints, 1), dtype = self.dtype, device = self.device)

        # Rotations
        self._rotation = torch.zeros((npoints, 4), device=self.device)
        self._rotation[:, 0] = 1
        self._rotation = torch.nn.functional.normalize(self._rotation)

    def reset_sizes(self):
        neighs = distCUDA2(self.get_rescaled_xyz).unsqueeze(1)
        neighs = torch.sqrt(neighs) 
        if self.point_size > 0:
            neighs = neighs * self.point_size
        self._scaling = neighs.repeat(1, 3) 

    def update_mip_sizes(self, camera: Camera):
        with torch.no_grad():
            mip = camera.extrinsics.world_to_camera(self.get_xyz)[:, [-1]] / camera.intrinsics.fx    
        if self.mip_size > 0:
            self._scaling = torch.minimum(self._scaling, mip / self.rescale_factor * self.mip_size)

    @property
    def get_pc(self) -> torch.Tensor:
        pc = torch.concatenate([self.get_xyz, self.get_features], dim = 1)
        return NamedTensor(pc, CHANNELS_GROUPS.xyz + self.feat_channels, 1)
    
    @property
    def get_scaling(self) -> torch.Tensor:
        return self._scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._rotation

    @property
    def get_rescaled_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz * self.rescale_factor

    @property
    def get_features(self) -> torch.Tensor:
        return self._features

    @property
    def get_opacity(self) -> torch.Tensor:
        return self._opacity

    @property
    def nfeatures(self) -> int:
        return self._nfeatures
class Renderer(ABC):
    def __init__(self, render_channels: List[str] = []):
        self.feat_channels = [c for c in render_channels if c not in CHANNELS_GROUPS.synth]
        self.synth_channels = [c for c in render_channels if c in CHANNELS_GROUPS.synth]
        self._max_channels = 64
        self._fixed_features_size = False

    def prepare_features(self, gaussians: GaussianModel) -> torch.Tensor:
        if len(self.feat_channels) > self._max_channels:
            raise ValueError(f"Maximum number of channels ({self._max_channels}) reached ({len(self.feat_channels)})")
        features = gaussians.get_pc[self.feat_channels]
        trailing_zeros = self._max_channels - features.shape[1]
        if self._fixed_features_size and trailing_zeros > 0:
            z = torch.zeros((features.shape[0], trailing_zeros), dtype = features.dtype, device = features.device)
            features = torch.concatenate([features, z], 1)
        return features

    @abstractmethod
    def render(self, gaussians: GaussianModel, camera: Camera) -> NamedTensor:
        pass
    
    def post_process(self, render_pkg: Dict[str, torch.Tensor], camera: Camera) -> NamedTensor:
        img = render_pkg["render"]
        img = NamedTensor(img, self.feat_channels, 0)

        if CHANNELS.opacity in self.synth_channels:
            img[CHANNELS.opacity] = render_pkg["opacity"]

        if CHANNELS.depth in self.synth_channels:
            img[CHANNELS.depth] = render_pkg["depth"]

        if any([n in img.names for n in CHANNELS_GROUPS.normals]):
            img[CHANNELS_GROUPS.normals] = rotate_abs_normals(
                img[CHANNELS_GROUPS.normals], camera
            )

        if CHANNELS.xi in self.synth_channels:
            img[CHANNELS.xi] = render_pkg["pixel_coords"][0]

        if CHANNELS.yi in self.synth_channels:
            img[CHANNELS.yi] = render_pkg["pixel_coords"][1]

        if CHANNELS.idx in self.synth_channels:
            img[CHANNELS.idx] = render_pkg["index"]

        return img