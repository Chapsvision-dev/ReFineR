import os
from pathlib import Path
import shutil
from typing import Dict

def add_ablations(exp_dir: Path, mapping: Dict):
    ablations = os.listdir(exp_dir)
    for ab in ablations:
        ab_dir = exp_dir / ab
        if ab in ["base_r", "base_g"]:
            save_ab = "base_i"
        elif ab in ["photo_r", "photo_g"]:
            save_ab = "photo_i"
        elif ab in ["photo_r_no_ma", "photo_g_no_ma"]:
            save_ab = "photo_i_no_ma"
        elif ab in ["no_ma_r", "no_ma_g"]:
            save_ab = "no_ma_i"
        else:
            save_ab = ab
        print(save_ab)
        for set in os.listdir(ab_dir):
            compile_dir = ab_dir / set / "compiled_results"
            if os.path.exists(compile_dir):
                mapping[f"{ab}/{set}"] = f"refiner_ma/{set}_{save_ab}"


if __name__ == "__main__":
    move_dict = {}
    bench_path = Path(os.getenv("BENCH_DIR")) / "data"
    out_path = Path("output") / "experiments"

    add_ablations(out_path, move_dict)

    for k in move_dict:
        src_dir = out_path / k / "compiled_results" / "fused"
        dst_dir = bench_path / move_dict[k] / "compiled_results" / "fused"
        print(src_dir)
        if os.path.exists(src_dir):
            if os.path.exists(dst_dir):
                continue

            shutil.copytree(src_dir, dst_dir)
