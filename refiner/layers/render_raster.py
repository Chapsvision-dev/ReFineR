from typing import Tuple

import torch

import refiner_C

from refiner.layers.gaussian_model import GaussianModel, Renderer
from refiner.utils.named_tensor import NamedTensor
from refiner.camera.camera import Camera
from refiner.utils.constants import CHANNELS_GROUPS

class RendererRaster(Renderer):
    def render(
        self,
        gaussians: GaussianModel,
        camera: Camera
    ) -> NamedTensor:
        features = self.prepare_features(gaussians)
        render_pkg = self._rasterize(
            gaussians.get_xyz,
            features,
            camera)
    
        return self.post_process(render_pkg, camera)

    @staticmethod
    def _rasterize(xyz: torch.Tensor, features: torch.Tensor, camera: Camera) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype, device = features.dtype, features.device
        idx = torch.arange(xyz.shape[0], dtype = dtype, device = device).unsqueeze(-1)
        features = torch.concatenate([features, idx], dim = 1)

        synth, depth, pixel_coords = torch.ops.refiner_C.rasterize( \
                    xyz.to("cpu"), 
                    features.to("cpu"), 
                    camera.extrinsics.view_matrix.to("cpu"), 
                    camera.intrinsics.intrinsics_matrix.to("cpu"), 
                    camera.intrinsics.W, 
                    camera.intrinsics.H,
                    camera.distortion.params.detach().cpu().item())

        synth = torch.moveaxis(synth, -1, 0).to(device)
        depth = depth.unsqueeze(0).to(device)
        pixel_coords = torch.moveaxis(pixel_coords, -1, 0).to(device)

        render_pkg = {
            "render": synth[:-1],
            "depth": depth[0],
            "pixel_coords": pixel_coords,
            "index": synth[-1]
        }  

        return render_pkg

    def render_points(
        self, 
        pc: NamedTensor,
        camera: Camera
    ) -> NamedTensor:  
        render_pkg = self._rasterize(
            pc[CHANNELS_GROUPS.xyz],
            pc[self.feat_channels],
            camera
        )

        return self.post_process(render_pkg, camera)