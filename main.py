import os
from pathlib import Path
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import gc
import argparse
from typing import Dict, Tuple, Optional, List
import torch
from omegaconf import OmegaConf

from refiner.utils.named_tensor import NamedTensor
from refiner.utils.functions import (
    get_errors,
    get_steps,
    check_convergence,
    update_logger_handlers,
)
from refiner.utils.config_types import FullCfg, MainCfg, RenderType
from refiner.utils.savers import Saver, RerunSaver, FileSaver
from refiner.data.common import Common
from refiner.data.point_image_dataset import PointImageMeta
from refiner.layers.render_gs import RendererGS
from refiner.layers.render_raster import RendererRaster
from refiner.layers.gaussian_model import GaussianModel, Renderer
from refiner.layers.loss import MultiLoss
from refiner.camera.camera import Camera
from refiner.utils.matchAnything import register_matchAnything


def log_errs(msg: str, errs: Dict[str, torch.Tensor]):
    logger.info(
        "{}: pix: {:.2f}px, rot: {:.2f}deg / {:.5f}rad, transl: {:.2f}cm, transl_rescaled: {:.5f}".format(
            msg,
            errs["pix_err"].detach().cpu().numpy(),
            errs["rot_err"].detach().cpu().numpy() * 180 / 3.14,
            errs["rot_err"].detach().cpu().numpy(),
            errs["transl_err"].detach().cpu().numpy() * 100,
            errs["transl_rescaled_err"].detach().cpu().numpy(),
        )
    )

def refine_init(
    cfg: FullCfg, pc: NamedTensor, base_img: NamedTensor, meta: PointImageMeta, infer: bool, init_camera: Optional[Camera] = None
) -> Tuple[
    GaussianModel,
    Camera,
    Renderer,
    MultiLoss,
    Optional[Camera],
    torch.optim.Adam,
    List[Saver],
]:
    # ===== Renderer =====
    match cfg.renderer.render_type:
        case RenderType.GS:
            renderer = RendererGS(cfg.renderer.render_channels)

    # ===== Inference mode =====
    if infer:
        logger.info("No target camera given, running in inference mode")
    else:
        logger.info(f"Found target camera, running in experiment mode")
    
    # ===== GS Model =====
    gaussians = GaussianModel(pc, **OmegaConf.to_container(cfg.gs, resolve=True), rescale_factor=meta.rescale_factor)

    # ===== Camera =====
    if init_camera is not None:
        camera = init_camera
    else:
        camera = Camera(
            **OmegaConf.to_container(cfg.camera, resolve=True),
            view_matrix=meta.pose_init,
            intr_matrix=meta.intr_init,
            size=meta.size,
            rescale_factor=meta.rescale_factor,
        )
        if cfg.matchAnything_init:
            ma_renderer = renderer if cfg.matchAnything_init_gs else RendererRaster(renderer.feat_channels)
            pose_init, intr_init, dist_init = register_matchAnything(camera, gaussians, base_img, ma_renderer) 
            camera = Camera(
                **OmegaConf.to_container(cfg.camera, resolve=True),
                view_matrix=pose_init,
                intr_matrix=intr_init,
                dist=dist_init,
                size=meta.size,
                rescale_factor=meta.rescale_factor,
            )
    gaussians.update_mip_sizes(camera)

    target_camera = (
        Camera(
            **OmegaConf.to_container(cfg.camera, resolve=True),
            view_matrix=meta.pose_target,
            intr_matrix=meta.intr_target,
            size=meta.size,
            rescale_factor=meta.rescale_factor,
        )
        if meta.pose_target is not None
        else None
    )

    # ===== Loss =====
    loss = MultiLoss(base_img, OmegaConf.to_container(cfg.loss, resolve=True))

    # ===== Optimizer =====
    optimizer = torch.optim.Adam([
            {
                "params": camera.extrinsics.backend.params,
                "lr": cfg.optim.lr,
                "name": "tau",
            },
            {
                "params": camera.intrinsics.backend.params,
                "lr": cfg.optim.lr,
                "name": "intr"
            },
            {
                "params": camera.distortion.params,
                "lr": cfg.optim.lr * 30,
                "name": "dist" 
            }
        ],
        amsgrad=True
    )


    # ===== Savers =====
    savers: List[Saver] = []
    if cfg.saver_rerun and (
        cfg.saver_rerun.path != "" or cfg.saver_rerun.spawn
    ):
        savers.append(
            RerunSaver(
                [loss.channels, loss.target_channels, loss.names],
                meta.rescale_factor,
                infer=infer,
                **OmegaConf.to_container(cfg.saver_rerun, resolve=True),
            )
        )
    if cfg.saver_file:
        savers.append(
            FileSaver(
                [loss.channels, loss.target_channels, loss.names],
                infer=infer,
                **OmegaConf.to_container(cfg.saver_file, resolve=True),
            )
        )

    return gaussians, camera, renderer, loss, target_camera, optimizer, savers

def refine(
    cfg: FullCfg, pc: NamedTensor, eval_pc: NamedTensor, base_img: NamedTensor, meta: PointImageMeta
):
    init_camera = None
    fi = 0
    for epoch in range(1):
        i = 1   
        logger.info(f"Epoch {epoch}")
        
        # Sample init
        infer = eval_pc is None
        gaussians, camera, renderer, loss, target_camera, optimizer, savers = (
            refine_init(cfg, pc, base_img, meta, infer, init_camera)
        )

        # Init errors
        img_target = None
        if not infer:
            errs = get_errors(camera, target_camera, eval_pc)
            log_errs("Initial errors", errs)
            for saver in savers:
                saver.record_metrics(0, [errs])

        if target_camera:
            img_target = renderer.render(gaussians, target_camera)
        for saver in savers:
            saver.save_init(cfg, base_img, img_target, gaussians, camera, target_camera)
            with torch.no_grad():
                img = renderer.render(gaussians, camera)
                saver.save_iter(0, img, base_img, gaussians, camera, eval_pc)
            
        t = time.time()
        stop = False
        steps_list = []
        times = {"time_render": 0., "time_loss": 0., "time_backward": 0.}
        best_loss, best_cam = torch.tensor(float("inf"), dtype = base_img.tensor.dtype, device = base_img.tensor.device), camera.clone()
        while not stop:

            # Optimizer
            optimizer.zero_grad()

            # Render
            t0 = time.time()
            img = renderer.render(gaussians, camera)
            t1 = time.time()
            times["time_render"] += t1 - t0

            # Computing loss
            losses = loss.forward(img, base_img, mask = camera.dist_mask)
            t2 = time.time()
            times["time_loss"] += t2 - t1
            l = losses["loss"]

            if l < best_loss:
                best_loss = l.detach()
                best_cam = camera.clone()

            # Backward / Step
            l.backward()
            optimizer.step()
            t3 = time.time()
            times["time_backward"] += t3 - t2

            # Metrics and logging
            with torch.no_grad():
                steps = get_steps(camera)
                err = camera.extrinsics.backend.update_params()
                camera.intrinsics.backend.update_params()
                camera.distortion.update_params()
                steps_list.append(steps)
                if len(steps_list) > cfg.optim.niter[1]:
                    steps_list.pop(0)

            if i % cfg.logger.log_freq == 0:
                with torch.no_grad():
                    m = [steps, losses, {"distortion": camera.distortion.params}]
                    if not infer:
                        m.append(get_errors(camera, target_camera, eval_pc))
                    for saver in savers:
                        saver.save_iter(i + fi, img, base_img, gaussians, camera, eval_pc)
                        saver.record_metrics(i, m)
                logger.info("it: {}, loss: {:.6f}, render time: {:.6f}, loss time: {:.6f}, backward time: {:.6f}".format(i, l.detach().cpu().numpy(), t1 - t0, t2 - t1, t3 - t2))
            if err > 0:
                logger.error("Failed registration")
                break

            # Convergence
            stop = check_convergence(i, steps_list, cfg.optim.niter)
            i += 1
        
        init_camera = camera
        fi += i
    # Final errors
    t = time.time() - t
    m = [steps, losses, {"time": torch.tensor(t), "iter": torch.tensor(i)}, {k: torch.tensor(times[k]) for k in times}]
    if not infer:
        errs = get_errors(best_cam, target_camera, eval_pc)
        m.append(errs)
        log_errs("Final errors", errs)
    logger.info(f"Computation Time: {t}s")
    for saver in savers:
        saver.record_metrics(i, m)
        saver.save_final(base_img, gaussians, best_cam)


if __name__ == "__main__":

    # CLI
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ReFineR main script",
    )
    parser.add_argument(
        "--cfg_file", type=str, help="Config file", nargs="*", default = ["config/main.yaml"]
    )
    parser.add_argument(
        "--cfg",  type=str, help="Manual config", nargs="*", default=[]
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data folder. Set REFINER_DATA_DIR for automatic default value",
        default="",
    )
    parser.add_argument(
        "--img_name", type=str, help="Image name to register", default=None
    )
    parser.add_argument(
        "--img_idx",
        type=int,
        help="Image index to register overrides <img_name>. If both <img_name> and <img_idx> are missing, process all images in the scene",
        default=None,
    )
    parser.add_argument("--set", type=str, help="Set from the dataset", required=True)
    parser.add_argument(
        "--scene",
        type=str,
        help="Scene from the dataset given set",
        required=False,
        default="",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        help="Number of repeats for each image for leading experiments",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--skips",
        type=int,
        help="Number skips",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory. Will be postfixed with /<set>/<scene>/<img_name> for each image registered",
        default="output/main",
    )
    parser.add_argument(
        "--precomp_dir",
        type=str,
        help="Precomputations directory. Will be postfixed with /<set>/<scene>",
        default="output/precomp",
    )
    parser.add_argument("--device", type=str, help="CUDA device", default="cuda:0")

    args = parser.parse_args()
    device = args.device
    cfgs = args.cfg_file
    skips = int(args.skips)
    args_cfg = {
        "out_path": Path(args.out_dir),
        "data": {"path": args.data_path, "set": args.set, "scene": args.scene},
    }

    # Config
    cfgs = [OmegaConf.load(cfg) for cfg in cfgs]
    inc_cfg: MainCfg = OmegaConf.merge(MainCfg, *[c for c in cfgs if "include" in c])
    inc_cfg = [
        OmegaConf.load(Path("config") / "components" / f"{c}.yaml") for c in inc_cfg.include
    ]
    rest_cfgs = [c for c in cfgs if "include" not in c]
    cfg: FullCfg = OmegaConf.merge(OmegaConf.structured(FullCfg),
                                   *inc_cfg,
                                   *rest_cfgs,
                                   args_cfg,
                                   OmegaConf.from_dotlist(args.cfg))

    # Logging init
    update_logger_handlers(cfg.logger, Path(cfg.out_path) / cfg.data.set)

    # Dataset init
    if cfg.data.path == "":
        cfg.data.path = os.getenv("REFINER_DATA_DIR")
    dataset = Common(**cfg.data, device=device)

    # Images selection
    images = []
    if args.img_idx is not None:
        images = [args.img_idx]
    elif args.img_name is not None:
        img_idx = dataset.get_idx_from_name(args.img_name)
        if img_idx == -1:
            raise FileNotFoundError(f"Image named {args.img_name} not found")
        images = [img_idx]
    else:
        images = list(range(len(dataset)))

    # Item loop
    skipped = 0
    for idx in images:
        item_name = dataset.get_name_from_idx(idx)
        scene = dataset.frames[idx].scene
        dataset.precomp_path = Path(args.precomp_dir) / args.set / scene

        for r in range(args.repeat):
            if skipped < skips:
                msg = f"Skipping image '{item_name}' from scene {cfg.data.set}/{scene}"
                if args.repeat > 1:
                    cfg.out_path = cfg.out_path / f"rep_{str(r)}"
                    msg += f" repeat {r}"

                logger.info(msg)
                skipped += 1
                continue
            cfg.out_path = Path(args.out_dir) / cfg.data.set / scene / item_name
            msg = f"Registering image '{item_name}' from scene {cfg.data.set}/{scene}"

            if args.repeat > 1:
                cfg.out_path = cfg.out_path / f"rep_{str(r)}"
                msg += f" repeat {r}"

            # Logging
            update_logger_handlers(cfg.logger, Path(cfg.out_path))

            logger.info(msg)
            pc, eval_pc, base_img, meta = dataset[idx]

            # ReFineR
            refine(cfg, pc, eval_pc, base_img[0], meta[0])
            # Manual Clean
            gc.collect()
            torch.cuda.empty_cache()
