import os
from pathlib import Path

if __name__ == "__main__":
    channel = "n"
    sets = ["TanksAndTemples", "nuScenes", "ETH3D", "RGBDScenes", "Livox", "Carla"]
    ablations_dir = Path("config/experiments/ablations")
    out_dir =  Path("output/experiments/ablations")
    ablations = [ab.split(".")[0] for ab in os.listdir(ablations_dir)]

    ablations = [
        ["photo_g", ["ETH3D"]],
    ]
    threshs = [1, 2, 3, 5, 6, 10, 15, 20, 50, 100]
    repeats = {
        "Carla": 2,
        "ETH3D": 2,
        "RGBDScenes": 5,
        "TanksAndTemples": 5,
        "Livox": 5,
        "nuScenes": 5
    }
    for name in ablations:
        if isinstance(name, list):
            test_sets = name[1]
            name = name[0]
        else:
            test_sets = sets
        for set in test_sets:
            if set == "TanksAndTemples":
                add = "--cfg data.pc_subsample_factor=0.4"
            else:
                add = ""
            print(f"Starting ablation {name} for set {set}")
            cmd = ["python",
                "main.py",
                "--set", set,
                "--cfg_file", str(ablations_dir / f"base_{channel}.yaml"), str(ablations_dir / f'{name}.yaml'),
                add,
                "--out_dir", str(out_dir / name),
                "--repeat", str(repeats[set])]
            os.system(" ".join(cmd))
            cmd = ["python", "compile_results.py", "--out_dir", str(out_dir / name), "--threshs",  " ".join([str(t) for t in threshs])]
            os.system(" ".join(cmd))