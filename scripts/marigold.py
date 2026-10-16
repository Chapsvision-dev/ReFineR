import os
from pathlib import Path
import argparse
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

from omegaconf import OmegaConf
import torch
from PIL import Image

from refiner.utils.config_types import DataCfg
from refiner.data.common import Common

models_params = {"depth": "depth", "normals": "normals", "iid": "iid-appearance"}
if __name__ == "__main__":

    # CLI
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Marigold script for preprocessing",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data folder. Set REFINER_DATA_DIR for automatic default value",
        default=os.getenv("REFINER_DATA_DIR"),
    )
    parser.add_argument("--set", type=str, help="Set from the dataset", required=True)
    parser.add_argument(
        "--scene", type=str, help="Scene from the dataset given set", required=False, default = ""
    )
    parser.add_argument("--device", type=str, help="CUDA device", default="cuda:0")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Marigold models to run [depth, normals, iid]",
        choices=["depth", "normals", "iid"],
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory. Will be postfixed with /[set]/[scene]",
        default="output/precomp",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        help="Weight directory",
        default="weights",
    )

    args = parser.parse_args()
    set = args.set
    scene = args.scene
    device = args.device

    # Cfg
    cfg_path = Path("config") / "precomp" / "marigold.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg: DataCfg = OmegaConf.merge(
        OmegaConf.structured(DataCfg),
        cfg,
        {"path": args.data_path, "set": set, "scene": scene},
    )

    # Dataset init
    dataset = Common(**cfg, device="cpu")

    for i in range(len(dataset)):
        out_path = Path(args.out_dir) / set / dataset.frames[i].scene
        img_path = out_path / "images"
        os.makedirs(img_path, exist_ok=True)

        img = dataset.get_online_img(i)
        chans = list(cfg.img_channels)
        img = img[chans]
        img = (img * 255).to(torch.uint8).detach().cpu().numpy()
        img = Image.fromarray(img)
        fn = dataset.frames[i].img_fn
        img.save(img_path / fn)

    # Run
    weights_path = Path(args.weights_dir) / "Marigold"
    logger.info(f"Running Marigold models: {args.models}")
    for model in args.models:
        out_path_0 = Path(args.out_dir) / set
        scenes = os.listdir(out_path_0)
        for scene in scenes:
            out_path = out_path_0 / scene
            img_path = out_path / "images"
            cmd = [
                "bash",
                f"./scripts/marigold.sh",
                img_path.absolute().as_posix(),
                out_path.absolute().as_posix(),
                str(weights_path.absolute().as_posix()),
                model,
                models_params[model],
            ]

            os.system(" ".join(cmd))
