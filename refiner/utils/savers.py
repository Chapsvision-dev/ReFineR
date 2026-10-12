import os
import glob
from pathlib import Path
import yaml
import csv
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from functools import reduce

import torch
import rerun as rr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from refiner.utils.named_tensor import NamedTensor
from refiner.utils.functions import apply_cmap, get_fused_img, get_match_img
from refiner.utils.rerun_bp import get_blueprint
from refiner.utils.constants import CHANNELS_GROUPS
from refiner.camera.camera import Camera
from refiner.layers.render_gs import GaussianModel
from refiner.data.ply import write_ply


def get_save_camera(camera: Camera):
    pose = camera.extrinsics.rescaled_view_matrix.detach().cpu().numpy()
    intr = camera.intrinsics.intrinsics_matrix.detach().cpu().numpy()
    size = camera.intrinsics.size
    camera = {"extr": pose.tolist(), "intr": intr.tolist(), "size": list(size)}
    return camera


def save_yaml(yml: Dict, path: Path):
    with open(path, "w") as f:
        yaml.dump(yml, f)


class Saver(ABC):
    def __init__(
        self,
        blueprint_names: List[str],
        path: str,
        pc_channel: str,
        infer: bool,
        modules: Tuple[str, ...] = (),
    ):

        self.path = Path(path)
        self.modules = modules
        self.pc_channel = pc_channel
        self.infer = infer
        self.blueprint_names = blueprint_names

    def save_cfg(self, cfg: Dict):
        pass

    @abstractmethod
    def save_gaussians(self, gaussians: GaussianModel):
        pass

    @abstractmethod
    def save_camera(self, camera: Camera, name: str):
        pass

    @abstractmethod
    def save_img_camera(
        self, img: NamedTensor, camera: Optional[Camera], camera_name: str
    ):
        pass

    @abstractmethod
    def save_raw_img(self, img: torch.Tensor, path):
        pass

    @abstractmethod
    def save_iter_camera(self, idx: int, camera: Camera):
        pass

    @abstractmethod
    def save_iter_img(self, idx: int, img: NamedTensor):
        pass

    @abstractmethod
    def save_metric(self, idx: int, metric: float, name: str):
        pass

    @staticmethod
    @abstractmethod
    def get_iter_path(name: str, idx: int) -> str:
        pass

    def save_init(
        self,
        cfg: Dict,
        img: NamedTensor,
        img_target: NamedTensor,
        gaussians: GaussianModel,
        camera: Camera,
        target_camera: Optional[Camera],
    ):
        # Save pc
        self.save_cfg(cfg)

        if "pc" in self.modules:
            self.save_gaussians(gaussians)

        # Save cameras
        if target_camera:
            self.save_camera(target_camera, "target")

            # Save img
            if "imgs_init" in self.modules:
                self.save_img_camera(img, target_camera, "target")
                self.save_img_camera(img_target, None, "target_synth")
        elif "imgs_init" in self.modules:
            self.save_img_camera(img, camera, "target")

        self.save_camera(camera, "init")


        # Save fuse
        if "fused" in self.modules:
            fused = get_fused_img(img, gaussians, camera).detach().cpu().numpy()
            self.save_raw_img(fused, "init")
            if "fused_iter" in self.modules:
                self.save_raw_img(fused, self.get_iter_path("fused", 0))


    def save_iter(
        self,
        idx: int,
        img: NamedTensor,
        base_img: NamedTensor,
        gaussians: GaussianModel,
        camera: Camera,
        eval_pc: NamedTensor
    ):

        # Save iter camera
        if "cams" in self.modules:
            self.save_iter_camera(idx, camera)

        # Save iter img
        if "imgs" in self.modules:
            self.save_iter_img(idx, img)

        # Save fuse
        if "fused_iter" in self.modules:
            fused = get_fused_img(base_img, gaussians, camera).detach().cpu().numpy()
            self.save_raw_img(fused, self.get_iter_path("fused", idx))
        
        if "match_pairs" in self.modules and not self.infer:
            matchs = get_match_img(base_img, eval_pc[CHANNELS_GROUPS.xyz], eval_pc[CHANNELS_GROUPS.img_xy], camera)
            self.save_raw_img(matchs, self.get_iter_path("matchs", idx))

        if "dist_mask" in self.modules:
            self.save_raw_img(camera.dist_mask, self.get_iter_path("dist_mask", idx))

    def save_final(self, img: NamedTensor, gaussians: GaussianModel, camera: Camera):
        self.save_camera(camera, "final")

        if "fused" in self.modules:
            fused = get_fused_img(img, gaussians, camera).detach().cpu().numpy()
            self.save_raw_img(fused, "final")

    def record_metrics(self, idx: int, metrics: Dict[str, torch.Tensor]):
        metrics = reduce(lambda m1, m2: m1 | m2, metrics)  # Merge dicts
        for n in metrics:
            m = metrics[n].detach().cpu().numpy()
            self.save_metric(idx, float(m), n)


class FileSaver(Saver):

    def __init__(
        self,
        blueprint_names: List[List[str]],
        path: str,
        pc_channel: str,
        infer: bool,
        modules: Tuple[str, ...] = (),
    ):

        super(FileSaver, self).__init__(
            blueprint_names, path, pc_channel, infer, modules
        )
        self.channels, _, loss_names = self.blueprint_names

        if "imgs" in self.modules:
            for n in self.channels:
                os.makedirs(self.path / "imgs" / n, exist_ok=True)

        if "cams" in self.modules:
            os.makedirs(self.path / "cams", exist_ok=True)

        if "graphs" in self.modules:
            os.makedirs(self.path / "graphs", exist_ok=True)

        if "csvs" in self.modules:
            os.makedirs(self.path / "csvs", exist_ok=True)

        if "fused_iter" in self.modules:
            os.makedirs(self.path / "imgs" / "fused", exist_ok=True)

        self.metrics = defaultdict(dict)
        self.graph_groups = {"losses": loss_names}

    @staticmethod
    def get_iter_path(name: str, idx: int) -> str:
        return f"imgs/{name}/{idx}"

    def save_cfg(self, cfg: Dict):
        save_yaml(OmegaConf.to_container(cfg, resolve=True), self.path / "cfg.yaml")

    def save_gaussians(self, gaussians: GaussianModel):
        fields = [gaussians.get_rescaled_xyz.detach().cpu().numpy()]  # N x 3
        names = [c for c in CHANNELS_GROUPS.xyz]

        if self.pc_channel != "":
            i = gaussians.feat_channels.index(self.pc_channel)
            fields.append(
                gaussians.get_features[:, i].unsqueeze(-1).detach().cpu().numpy()
            )  # N x 1
            names.append(self.pc_channel)

        write_ply(str(self.path / "pc.ply"), fields, names)

    def save_camera(self, camera: Camera, name: str):
        camera = get_save_camera(camera)
        save_yaml(camera, self.path / f"camera_{name}.yaml")

    def save_img_camera(
        self, img: NamedTensor, camera: Optional[Camera], camera_name: str
    ):
        for c in img.names:
            im = img[c].detach().cpu().numpy()
            im = apply_cmap(im, "Greys_r")
            self.save_raw_img(im, f"{c}")

            if "imgs" in self.modules:
                for ch in self.channels:
                    self.save_raw_img(im, f"imgs/{ch}/{camera_name}")

    def save_raw_img(self, img: np.ndarray, path):
        img = Image.fromarray(img)
        img.save(self.path / f"{path}.png")

    def save_iter_camera(self, idx: int, camera: Camera):
        camera = get_save_camera(camera)
        save_yaml(camera, self.path / "cams" / f"{idx}.yaml")

    def save_iter_img(self, idx: int, img: NamedTensor):
        for c in img.names:
            im = img[c].detach().cpu().numpy()
            im = apply_cmap(im, "Greys_r")
            self.save_raw_img(im, self.get_iter_path(c, idx))

    def save_metric(self, idx: int, metric: float, name: str):
        self.metrics[name][idx] = float(metric)

    def save_final(self, img: NamedTensor, gaussians: GaussianModel, camera: Camera):
        super(FileSaver, self).save_final(img, gaussians, camera)

        # Create videos
        if "video" in self.modules:
            print(os.getcwd())
            for c in self.channels + (["fused"] if os.path.exists(self.path / "imgs" / "fused") else []):
                channel_path = self.path / "imgs" / c
                cmd = ["ffmpeg", "-framerate", str(25), "-i", str(channel_path / "%000d.png"), "-c:v libx264", "-profile:v", "high", "-crf", str(20), "-pix_fmt", "yuv420p", str(channel_path / "output.mp4"), "-y"]
                os.system(" ".join(cmd))
                files = glob.glob("[0-9]*.png", root_dir = channel_path)
                for f in files:
                    os.remove(channel_path / f)

        # Save metrics graphs and txt files
        with open(self.path / f"metrics.txt", "w", newline="") as csvfile:
            writer = csv.writer(
                csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            writer.writerows(
                [[n, list(self.metrics[n].values())[-1]] for n in self.metrics]
            )

        graph_groups = {g: self.graph_groups[g] for g in self.graph_groups}
        grouped_names = reduce(
            lambda t, u: t + u, [graph_groups[g] for g in graph_groups], []
        )
        for i in self.metrics:
            if i not in grouped_names and len(self.metrics[i]) > 1:
                graph_groups[i] = [i]

        for group in graph_groups:
            names = graph_groups[group]
            metrics = [self.metrics[n] for n in names]
            x = list(metrics[0].keys())
            y = [list(m.values()) for m in metrics]

            if "graphs" in self.modules:
                plt.figure()
                for name, val in zip(names, y):
                    plt.plot(x, val, label=name)
                    plt.legend()
                plt.savefig(self.path / "graphs" / f"{group}.png")
                plt.close()

            if "csvs" in self.modules:
                y = [x] + y
                y = list(map(list, zip(*y)))
                with open(
                    self.path / "csvs" / f"{group}.txt", "w", newline=""
                ) as csvfile:
                    writer = csv.writer(
                        csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
                    )
                    writer.writerow(["it"] + names)
                    writer.writerows(y)


class RerunSaver(Saver):
    color_dict = {
        "target": [64, 196, 166],
        "init": [181, 34, 202],
        "iter/camera_iter": [177, 145, 0],
        "final": [255, 255, 255]
    }
    def __init__(
        self,
        blueprint_names: List[str],
        rescale_factor: torch.Tensor,
        path: str,
        pc_channel: str,
        infer: bool,
        modules: Tuple[str, ...] = (),
        spawn: bool = True,
        view_coordinates: str = "RIGHT_HAND_Z_UP",
    ):

        super(RerunSaver, self).__init__(
            blueprint_names, path, pc_channel, infer, modules
        )
        self.blueprint = get_blueprint(*self.blueprint_names, infer=self.infer)
        rr.init(str(self.path).replace("/", "-"), spawn=spawn)
        if not spawn:
            rr.save(str(self.path / "rerun.rrd"))
        rr.send_blueprint(self.blueprint)
        rr.set_time(timeline="main", sequence=0)
        rr.log("/", getattr(rr.ViewCoordinates, view_coordinates), static=True)

        rescale_factor = rescale_factor.detach().cpu().numpy()
        self._rescale_factor = 30
        self.img_plane_distance = 0.25 / rescale_factor * self._rescale_factor
        self.point_radius = 0.005 / rescale_factor * self._rescale_factor
        self.axis_length = 0.
        self.save_traj_cam = False

    @staticmethod
    def get_iter_path(name: str, idx: int) -> str:
        return f"iter/{name}"

    def save_gaussians(self, gaussians: GaussianModel):
        colors = None
        if self.pc_channel != "":
            i = gaussians.feat_channels.index(self.pc_channel)
            colors = (
                gaussians.get_features[:, i]
                .unsqueeze(-1)
                .repeat(1, 3)
                .detach()
                .cpu()
                .numpy()
            )  # N x 3
        rr.log(
            "points",
            rr.Points3D(
                positions=gaussians.get_rescaled_xyz.detach().cpu().numpy()
                * self._rescale_factor,
                radii=self.point_radius,
                colors=colors,
            ),
            static=True,
        )

    def save_camera(self, camera: Camera, name: str):
        pose = camera.extrinsics.rescaled_view_matrix.detach().cpu().numpy()
        intr = camera.intrinsics.intrinsics_matrix.detach().cpu().numpy()
        size = camera.intrinsics.size
        rr.log(
            f"camera/camera_{name}",
            rr.Transform3D(
                translation=pose[:3, 3] * self._rescale_factor,
                mat3x3=pose[:3, :3],
                relation=rr.TransformRelation.ChildFromParent,
            ),
            rr.TransformAxes3D(self.axis_length),
            static=False,
        )
        rr.log(
            f"camera/camera_{name}/image",
            rr.Pinhole(
                focal_length=intr[[0, 1], [0, 1]],
                principal_point=intr[:2, 2],
                width=size[0],
                height=size[1],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=self.img_plane_distance,
                color = self.color_dict[name],
            ),
            static=False,
        )

        coord = np.linalg.inv(intr) @ np.array([0, 0, self.img_plane_distance])

        rr.log(
            f"camera/camera_{name}/label",
            rr.Points3D(
                coord, radii=self.point_radius * 2, colors=self.color_dict[name], labels=name.split("/")[0], show_labels=True
            )
        )

    def save_img_camera(
        self, img: NamedTensor, camera: Optional[Camera], camera_name: str
    ):
        for c in img.names:
            im = img[c].detach().cpu().numpy()
            im = apply_cmap(im, "Greys_r")
            path = f"camera/camera_{camera_name}/views/{c}"
            self.save_raw_img(im, path)
            if camera:
                intr = camera.intrinsics.intrinsics_matrix.detach().cpu().numpy()
                size = camera.intrinsics.size
                rr.log(
                    path,
                    rr.Pinhole(
                        focal_length=intr[[0, 1], [0, 1]],
                        principal_point=intr[:2, 2],
                        width=size[0],
                        height=size[1],
                        camera_xyz=rr.ViewCoordinates.RDF,
                        image_plane_distance=self.img_plane_distance,
                        color = self.color_dict[camera_name]
                    ),
                    static=False,
                )

    def save_raw_img(self, img: np.ndarray, path):
        """
        img: tensor H x W x 3 between 0 and 255
        """
        rr.log(path, rr.Image(img), static=False)

    def save_iter_camera(self, idx: int, camera: Camera):
        rr.set_time("main", sequence=idx)

        pose = camera.extrinsics.rescaled_view_matrix.detach().cpu().numpy()
        intr = camera.intrinsics.intrinsics_matrix.detach().cpu().numpy()
        size = camera.intrinsics.size
        rr.log(
            f"traj/cam/{idx}",
            rr.Transform3D(
                translation=pose[:3, 3] * self._rescale_factor,
                mat3x3=pose[:3, :3],
                relation=rr.TransformRelation.ChildFromParent,
            ),
            rr.TransformAxes3D(0.),
        )
        if self.save_traj_cam:
            rr.log(
                f"traj/cam/{idx}",
                rr.Pinhole(
                    focal_length=intr[[0, 1], [0, 1]],
                    principal_point=intr[:2, 2],
                    width=size[0],
                    height=size[1],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=self.img_plane_distance / 5,
                    color = [159, 159, 159]
                ),
            )
        pose = (pose[:3, :3].T @ pose[:3, 3]) * self._rescale_factor
        rr.log(
            f"traj/points/{idx}",
            rr.Points3D(-pose, radii=self.point_radius, colors=[255, 0, 0]),
        )
        self.save_camera(camera, "iter/camera_iter")

    def save_iter_img(self, idx: int, img: NamedTensor):
        for c in img.names:
            im = img[c].detach().cpu().numpy()
            im = apply_cmap(im, "Greys_r")
            self.save_raw_img(im, self.get_iter_path(c, idx))

    def save_metric(self, idx: int, metric: float, name: str):
        rr.log(f"iter/metrics/{name}", rr.Scalars(metric))
