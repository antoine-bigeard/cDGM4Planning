import sys
import shutil

sys.path.insert(0, "/home/abigeard/RA_CCS/DeepGenerativeModelsCCS")
import matplotlib.pyplot as plt
import argparse
import os
import json
import csv
import numpy as np
from src.utils import read_yaml_config_file


def move_plots(path_log, out_dir):
    config = read_yaml_config_file(os.path.join(path_log, "config.yaml"))
    out_plots_dir = os.path.join(path_log, config["name_experiment"])
    for f in os.listdir(path_log):
        if ".png" in f:
            shutil.copyfile(os.path.join(path_log, f), out_plots_dir)
    with open(os.path.join(path_log, "metrics_img_path.json")) as f:
        paths_dict = json.load(f)
    for k, v in paths_dict.items():
        if k not in ["L2_measures", "dist_cond_measures", "samples_per_sec"]:
            local_out_dir = os.path.join(out_dir, config["name_experiment"], f"{v:.3f}")
            os.makedirs(local_out_dir, exist_ok=True)
            shutil.copy(k, local_out_dir)
            path_gt = os.path.join("/".join(k.split("/")[:-1]), "ground_truth.png")
            shutil.copy(path_gt, local_out_dir)
    return paths_dict, config["name_experiment"]


def plot(path_logs, out_dir):
    fig = plt.figure()
    values = []
    for path_log in path_logs:
        paths_dict, label = move_plots(path_log, out_dir)
        plt.hist(
            paths_dict["L2_measures"],
            bins=100,
            density=True,
            histtype="step",
            cumulative=True,
            label=label,
        )
        values.append(
            [
                label,
                np.mean(paths_dict["L2_measures"]),
                np.mean(paths_dict["dist_cond_measures"]),
                # np.mean(paths_dict["measures_dist_cond"]),
                paths_dict["samples_per_sec"],
            ]
        )
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(out_dir, "cum_distrib_L2_dist"))
    fields = ["model", "avg L2 metric", "avg L2 cond", "nb samples /sec"]

    with open(os.path.join(out_dir, "table.csv"), "w") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_logs",
        help="log path that contains logs from previous tests.",
        default=[
            "logs/ore_maps_ddpm_100/version_1",
            "logs/ore_maps_ddpm_500/version_2",
            "logs/ore_maps_ddpm_1000/version_2",
        ],
        required=False,
    )

    parser.add_argument(
        "--out_dir",
        help="log path that contains logs from previous tests.",
        default="logs/test",
        required=False,
    )

    args = parser.parse_args()

    plot(args.path_logs, args.out_dir)
