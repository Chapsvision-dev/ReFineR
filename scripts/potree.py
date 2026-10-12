import os
import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import laspy

from refiner.utils.config_types import DataCfg
from refiner.data.common import Common
from refiner.utils.constants import CHANNELS_GROUPS

if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Potree script for preprocessing",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data folder. Set REFINER_DATA_DIR for automatic default value",
        default="",
    )

    parser.add_argument("--set", type=str, help="Set from the dataset", required=True)

    parser.add_argument(
        "--scene", type=str, help="Scene from the dataset given set", required=False, default = ""
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory. Will be postfixed with /[set]/[scene]",
        default="output/precomp",
    )

    parser.add_argument(
        "--potree_converter_path",
        type=str,
        help="ath to the build potreeConverter",
        default="third_party/PotreeConverter/build/PotreeConverter",
    )

    args = parser.parse_args()
    set = args.set
    scene = args.scene
    potree_converter_path = args.potree_converter_path

    # Cfg
    cfg_path = Path("config") / "precomp" / "potree.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg: DataCfg = OmegaConf.merge(
        OmegaConf.structured(DataCfg),
        cfg,
        {"path": args.data_path, "set": set, "scene": scene},
    )

    # Dataset init
    if cfg.path == "":
        cfg.path = os.getenv("REFINER_DATA_DIR")
    dataset = Common(**cfg, device="cpu")
        
    for scene in dataset.uscenes:
        out_path = Path(args.out_dir) / set / scene
        potree_path = out_path / "potree"
        print(potree_path)
        os.makedirs(potree_path, exist_ok=True)

        # Add index for later
        pc = dataset.get_online_pc(scene, CHANNELS_GROUPS.xyz)
        idx = np.arange(pc.tensor.shape[0], dtype = np.uint64)
        las_header = laspy.LasHeader(version = "1.4", point_format = 1)
        las_ply = laspy.LasData(las_header)
        las_ply.xyz = pc[CHANNELS_GROUPS.xyz]
        idx = idx.view(np.float64) # This is a hack
        las_ply.gps_time = idx 
        tmp_las_path = str(potree_path / "potree_tmp.las")
        las_ply.write(tmp_las_path)

        # Convert to potree
        cmd = [potree_converter_path, tmp_las_path, "-o", str(potree_path)]
        os.system(" ".join(cmd))

        # # Clean folder
        os.remove(tmp_las_path)