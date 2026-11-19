import os
from pathlib import Path
import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ReFineR experiments script"
    )

    parser.add_argument(
        "name",
        type = str, 
        help = "Experiment config file (omit the .yaml)"
    )

    parser.add_argument(
        "--sets",
        type = str,
        help = "Sets to run the experiment on",
        nargs="*",
        default = ["Carla", "ETH3D", "RGBDScenes", "TanksAndTemples", "Livox", "nuScenes"], 
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data folder. Set REFINER_DATA_DIR for automatic default value",
        default=os.getenv("REFINER_DATA_DIR"),
    )

    args = parser.parse_args()
    data_path = Path(args.data_path)
    test_sets = args.sets
    name = args.name

    exp_dir = Path("config") / "experiments"
    exp_dir2 = exp_dir
    if not os.path.exists(exp_dir / f"{name}.yaml"):
        exp_dir2 = exp_dir2 / "ablations"
    if not os.path.exists(exp_dir2 / f"{name}.yaml"):
        raise ValueError(f"Config file not found for experiment: {name}")

    out_dir =  Path("output") / "experiments"
    
    threshs = [1, 2, 3, 5, 6, 10, 15, 20, 50, 100]
    repeats = {
        "Carla": 2,
        "ETH3D": 2,
        "RGBDScenes": 5,
        "TanksAndTemples": 5,
        "Livox": 5,
        "nuScenes": 5
    }

    for set in test_sets:
        print(f"Starting ablation {name} for set {set}")
        subprocess.run(["python",
            "main.py",
            "--set", set,
            "--cfg", f"data.pc_subsample_factor={0.4 if set == 'TanksAndTemples' else 1.}",
            "--cfg_file", exp_dir / f"base.yaml", exp_dir2 / f'{name}.yaml',
            "--out_dir", out_dir / name,
            "--data_path", data_path,
            "--repeat", str(repeats[set])])
        subprocess.run(["python",
            "scripts/compile_results.py", 
            "--out_dir", 
            out_dir / name, 
            "--threshs",  
            *[str(t) for t in threshs]])