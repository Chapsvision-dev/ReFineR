import os
from pathlib import Path
import shutil
from typing import Dict

move_dict = {
    "normals_ma/Carla": "refiner_ma/Carla_n_10",
    "normals_ma/ETH3D": "refiner_ma/ETH3D_n_10",
    "normals_ma/TanksAndTemples": "refiner_ma/TanksAndTemples_n_10",
    "normals_ma/Livox": "refiner_ma/Livox_n_10",
    "normals_ma/nuScenes": "refiner_ma/nuScenes_n_10",
    "normals_ma/RGBDScenes": "refiner_ma/RGBDScenes_n_10",

    "intensity_ma/ETH3D": "refiner_ma/ETH3D_g_10",
    "intensity_ma/TanksAndTemples": "refiner_ma/TanksAndTemples_g_10",
    "intensity_ma/Livox": "refiner_ma/Livox_r_10",
    "intensity_ma/nuScenes": "refiner_ma/nuScenes_r_10",
    "intensity_ma/RGBDScenes": "refiner_ma/RGBDScenes_g_10",

    "normals/Carla": "refiner/Carla_n_10",
    "normals/ETH3D": "refiner/ETH3D_n_10",
    "normals/TanksAndTemples": "refiner/TanksAndTemples_n_10",
    "normals/Livox": "refiner/Livox_n_10",
    "normals/nuScenes": "refiner/nuScenes_n_10",
    "normals/RGBDScenes": "refiner/RGBDScenes_n_10",

    "intensity/ETH3D": "refiner/ETH3D_g_10",
    "intensity/TanksAndTemples": "refiner/TanksAndTemples_g_10",
    "intensity/Livox": "refiner/Livox_r_10",
    "intensity/nuScenes": "refiner/nuScenes_r_10",
    "intensity/RGBDScenes": "refiner/RGBDScenes_g_10",

    "intensity_gom_ma/ETH3D": "refiner_gom_ma/ETH3D_i_10",
    "intensity_gom_ma/TanksAndTemples": "refiner_gom_ma/TanksAndTemples_i_10",
    "intensity_gom_ma/Livox": "refiner_gom_ma/Livox_i_10",
    "intensity_gom_ma/nuScenes": "refiner_gom_ma/nuScenes_i_10",
    "intensity_gom_ma/RGBDScenes": "refiner_gom_ma/RGBDScenes_i_10",
}

def add_ablations(exp_dir: Path, mapping: Dict):
    ablation_dir = exp_dir / "ablations"
    ablations = os.listdir(ablation_dir)
    for ab in ablations:
        ab_dir = ablation_dir / ab
        if ab in ["base_r", "base_g"]:
            save_ab = "base_i"
        elif ab in ["photo_r", "photo_g"]:
            save_ab = "photo_i"
        elif ab in ["no_ma_r", "no_ma_g"]:
            save_ab = "no_ma_i"
        else:
            save_ab = ab
        for set in os.listdir(ab_dir):
            compile_dir = ab_dir / set / "compiled_results"
            if os.path.exists(compile_dir):
                mapping[f"ablations/{ab}/{set}"] = f"refiner_ma/{set}_{save_ab}"
    return mapping


if __name__ == "__main__":
    bench_path = Path(os.getenv("BENCH_DIR")) / "data"
    out_path = Path("output") / "experiments"

    add_ablations(out_path, move_dict)

    for k in move_dict:
        src_dir = out_path / k / "compiled_results"
        dst_dir = bench_path / move_dict[k] / "compiled_results"
        if os.path.exists(src_dir):
            if os.path.exists(dst_dir):
                shutil.rmtree(bench_path / move_dict[k])
            shutil.copytree(src_dir, dst_dir)
