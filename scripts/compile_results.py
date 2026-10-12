import os
import argparse
from pathlib import Path
import csv
import re
from collections import defaultdict
from typing import Dict
import shutil

import numpy as np

from refiner.utils.metrics import threshs_errors


def write_dict(dict: Dict[str, float], file: Path):
    with open(file, "w", newline="") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerows([[n, dict[n]] for n in dict])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compiling experimental results from the main script",
    )
    parser.add_argument(
        "--out_dir", type=str, help="Output directory of ReFineR", default="output/main"
    )
    parser.add_argument(
        "--set", type=str, help="Set from the dataset", required=False, default=""
    )
    parser.add_argument(
        "--threshs",
        type=int,
        help="Pixel error thresholds",
        required=False,
        nargs="+",
        default=[2, 3, 5, 10],
    )

    # Parse arguments
    args = parser.parse_args()
    set: str = args.set
    out_dir = Path(args.out_dir)
    threshs = args.threshs

    sets = os.listdir(out_dir) if set == "" else [set]

    for set in sets:
        set_dir = Path(args.out_dir) / set if set != "" else Path(args.out_dir)

        # Create directories
        result_dir = set_dir / "compiled_results"
        os.makedirs(result_dir, exist_ok=True)
        details_dir = result_dir / "details"
        os.makedirs(details_dir, exist_ok=True)
        ignore_dirs = ["log.txt", "compiled_results"]
        scenes = [s for s in os.listdir(set_dir) if s not in ignore_dirs]
        scenes.sort()
        metrics = defaultdict(list)
        fused_dir = result_dir / "fused"
        # Read final metrics
        names = []
        fails = 0
        for scene in scenes:
            scene_dir = set_dir / scene
            imgs = os.listdir(scene_dir)
            for img in imgs:
                img_dir = scene_dir / img
                reps_dir = os.listdir(img_dir)
                reps_dir = [img_dir / r for r in reps_dir if re.match("rep_[0-9]+", r)]
                if len(reps_dir) == 0:
                    reps_dir.append(img_dir)
                for rep_dir in reps_dir:
                    metrics_file = rep_dir / "metrics.txt"
                    if not os.path.exists(metrics_file):
                        fails += 1
                        continue
                    names.append(rep_dir)
                    with open(metrics_file, "r") as f:
                        reader = csv.reader(f, delimiter=" ", quotechar="|")
                        for line in reader:
                            metrics[line[0]].append(float(line[1]))
                    
                    fused_path = img_dir / "final.png"
                    if os.path.exists(fused_path):
                        os.makedirs(fused_dir, exist_ok=True)
                        shutil.copyfile(fused_path, fused_dir / f"{scene}_{img}.png")
        # Save metrics and thresholds
        np.savetxt(result_dir / "fails.txt", [fails], fmt = "%d")

        metrics["rot_err_deg"] = [m / np.pi * 180 for m in metrics["rot_err"]]
        metrics["transl_err_cm"] = [m * 100 for m in metrics["transl_err"]]
        means = {k: np.mean(metrics[k]) for k in metrics}
        meds = {k: np.median(metrics[k]) for k in metrics}
        stds = {k: np.std(metrics[k]) for k in metrics}

        write_dict(means, result_dir / "means.txt")
        write_dict(meds, result_dir / "meds.txt")
        write_dict(stds, result_dir / "stds.txt")

        t = threshs_errors(metrics["pix_err"], threshs)
        with open(result_dir / "thresholds.txt", "w", newline="") as csvfile:
            writer = csv.writer(
                csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            writer.writerows([[k, v] for k, v in zip([f"t_{th}" for th in threshs], t)])

        # Save details
        for m in metrics:
            with open(details_dir / m, "w", newline="") as csvfile:
                writer = csv.writer(
                    csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
                )
                writer.writerows([[k, v] for k, v in zip(names, metrics[m])])
