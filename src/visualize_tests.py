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
import matplotlib.pylab as pl

LABELS = {
    "ore_maps_ddpm_100": "DDPM - 100",
    "ore_maps_ddpm_250": "DDPM - 250",
    "ore_maps_ddpm_500": "DDPM - 500",
    "ore_maps_ddpm_1000": "DDPM - 1000",
    "ore_maps_gan_2inject_wgp": "WGP-GAN - condition injected twice",
    "ore_maps_gan_fullSN_wgp": "WGP-GAN - condition injected at every layer, Spectral Norm",
    "ore_maps_gan_fullinject_noT": "WGP-GAN - condition injected at every layer, no Transposed Conv",
    "ore_maps_gan_1inject_noT": "WGP-GAN - condition injected once, no Transposed Convolution",
}


def move_plots(path_log, out_dir):
    config = read_yaml_config_file(os.path.join(path_log, "config.yaml"))
    out_plots_dir = os.path.join(path_log, config["name_experiment"])
    for f in os.listdir(path_log):
        if ".png" in f:
            shutil.copyfile(os.path.join(path_log, f), out_plots_dir)
    with open(os.path.join(path_log, "metrics_img_path.json")) as f:
        paths_dict = json.load(f)
    for k1, v1 in paths_dict.items():
        if k1 not in [
            "L2_measures",
            "dist_cond_measures",
            "samples_per_sec",
            "L1_measures",
            "dist_cond_measures",
            "mean_L1",
            "mean_L2",
            "mean_dist_cond",
        ]:
            if isinstance(v1, dict):
                for k, v in v1.items():
                    local_out_dir = os.path.join(
                        out_dir, config["name_experiment"], f"{v:.3f}"
                    )
                    os.makedirs(local_out_dir, exist_ok=True)
                    shutil.copy(k, local_out_dir)
                    path_gt = os.path.join(
                        "/".join(k.split("/")[:-1]), "ground_truth.png"
                    )
                    shutil.copy(path_gt, local_out_dir)
            else:
                local_out_dir = os.path.join(
                    out_dir, config["name_experiment"], f"{v1:.3f}"
                )
                os.makedirs(local_out_dir, exist_ok=True)
                shutil.copy(k1, local_out_dir)
                path_gt = os.path.join("/".join(k1.split("/")[:-1]), "ground_truth.png")
                shutil.copy(path_gt, local_out_dir)
    return paths_dict, config["name_experiment"]


def plot(path_logs, out_dir):
    cm_ddpm = pl.cm.Reds(
        np.linspace(0.5, 1, len([path for path in path_logs if "ddpm" in path]))
    )
    cm_gan = pl.cm.Greens(
        np.linspace(0.5, 1, len([path for path in path_logs if "gan" in path]))
    )
    i_ddpm = 0
    i_gan = 0
    colors = []
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    values = []
    for path_log in path_logs:
        paths_dict, label = move_plots(path_log, out_dir)
        if "ddpm" in path_log:
            color = cm_ddpm[i_ddpm]
            i_ddpm += 1
        else:
            color = cm_gan[i_gan]
            i_gan += 1
        axis = ax.hist(
            paths_dict["L2_measures"],
            bins=100,
            density=True,
            histtype="step",
            cumulative=True,
        )
        colors.append(
            plt.plot(
                [],
                color=color,
                label=LABELS[label],
            )[0]
        )
        poly = axis[2][0]
        print(poly)
        vertices = poly.get_path().vertices

        # Keep everything above y == 0. You can define this mask however
        # you need, if you want to be more careful in your selection.
        keep = vertices[:, 1] > 0

        # Construct new polygon from these "good" vertices
        new_poly = plt.Polygon(
            vertices[keep],
            closed=False,
            fill=False,
            edgecolor=color,
            linewidth=poly.get_linewidth(),
        )
        poly.set_visible(False)
        ax.add_artist(new_poly)
        plt.draw()
        values.append(
            [
                label,
                np.mean(paths_dict["L2_measures"]),
                np.mean(paths_dict["dist_cond_measures"]),
                # np.mean(paths_dict["measures_dist_cond"]),
                # paths_dict["samples_per_sec"],
            ]
        )
    plt.title("Recall curve (CDF) for L2 metric")
    plt.xlabel("L2 value")
    plt.ylabel("Percentage of images")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(out_dir, "cum_distrib_L2_dist"))
    plt.close()

    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    values = []
    i_ddpm = 0
    i_gan = 0
    colors = []
    for path_log in path_logs:
        paths_dict, label = move_plots(path_log, out_dir)
        if "ddpm" in path_log:
            color = cm_ddpm[i_ddpm]
            i_ddpm += 1
        else:
            color = cm_gan[i_gan]
            i_gan += 1
        axis = ax.hist(
            paths_dict["L1_measures"],
            bins=100,
            density=True,
            histtype="step",
            cumulative=True,
        )
        colors.append(
            plt.plot(
                [],
                color=color,
                label=LABELS[label],
            )[0]
        )
        poly = axis[2][0]
        print(poly)
        vertices = poly.get_path().vertices

        # Keep everything above y == 0. You can define this mask however
        # you need, if you want to be more careful in your selection.
        keep = vertices[:, 1] > 0

        # Construct new polygon from these "good" vertices
        new_poly = plt.Polygon(
            vertices[keep],
            closed=False,
            fill=False,
            edgecolor=color,
            linewidth=poly.get_linewidth(),
        )
        poly.set_visible(False)
        ax.add_artist(new_poly)
        plt.draw()
        values.append(
            [
                label,
                np.mean(paths_dict["L1_measures"]),
                np.mean(paths_dict["dist_cond_measures"]),
                # np.mean(paths_dict["measures_dist_cond"]),
                # paths_dict["samples_per_sec"],
            ]
        )
    plt.title("Recall curve (CDF) for L1 metric")
    plt.xlabel("L1 value")
    plt.ylabel("Percentage of images")
    ax.legend(handles=colors, loc="center left", bbox_to_anchor=(1, 0.5))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.savefig(os.path.join(out_dir, "cum_distrib_L1_dist"))
    plt.close()

    fields = ["model", "avg L2 metric", "avg L2 cond", "nb samples /sec"]

    with open(os.path.join(out_dir, "table.csv"), "w") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_config",
        default="configs_visualize/visualize.yaml"
        required=False,
    )

    args = parser.parse_args()

    plot(args.path_config)
