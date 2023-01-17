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

from collections import defaultdict

LABELS = {
    "ore_maps_ddpm_100": "DDPM - 100",
    "ore_maps_ddpm_150": "DDPM - 150",
    "ore_maps_ddpm_200": "DDPM - 200",
    "ore_maps_ddpm_250": "DDPM - 250",
    "ore_maps_ddpm_250_smaller": "DDPM - 250 smaller",
    "ore_maps_ddpm_500": "DDPM - 500",
    "ore_maps_ddpm_500_smaller": "DDPM - 500 smaller",
    "ore_maps_ddpm_1000": "DDPM - 1000",
    "ore_maps_gan_2inject_wgp": "WGP-GAN - condition injected twice",
    "ore_maps_gan_fullSN_wgp": "WGP-GAN - condition injected at every layer, Spectral Norm",
    "ore_maps_gan_fullinject_noT": "WGP-GAN - condition injected at every layer, no Transposed Conv",
    "ore_maps_gan_1inject_noT": "WGP-GAN - condition injected once, no Transposed Convolution",
    "gan_ore_maps_fullinject_2conv64": "W-GAN - fullinject 2chan_wconv, latent dim 64",
    "gan_ore_maps_fullinject_8conv64": "W-GAN - fullinject 8chan_wconv, latent dim 64",
    "gan_ore_maps_fullinject_conv": "W-GAN - fullinject 2chan_wconv, latent dim 128",
    "gan_ore_maps_fullinject_8conv": "W-GAN - fullinject 8chan_wconv, latent dim 128",
    "gan_ore_maps_halfinject_conv": "W-GAN - halfinject 2chan_wconv, latent dim 128",
    "gan_ore_maps_halfinject_8conv": "W-GAN - halfinject 8chan_wconv, latent dim 128",
}


def move_plots(path_log: str, out_dir: str) -> tuple[dict, str]:
    config = read_yaml_config_file(os.path.join(path_log, "config.yaml"))
    out_plots_dir = os.path.join(path_log, config["name_experiment"])
    for f in os.listdir(path_log):
        if ".png" in f:
            shutil.copyfile(os.path.join(path_log, f), out_plots_dir)
    with open(os.path.join(path_log, "metrics_img_path.json")) as f:
        paths_dict = json.load(f)
    for k, v in paths_dict["paths"].items():
        for k1, v1 in v.items():
            tmp_out_dir = os.path.join(
                out_dir, config["name_experiment"], k, f"{v1:.4f}"
            )
            os.makedirs(tmp_out_dir, exist_ok=True)
            shutil.copy(k1, tmp_out_dir)
            path_gt = os.path.join("/".join(k1.split("/")[:-1]), "ground_truth.png")
            shutil.copy(path_gt, tmp_out_dir)
    paths_dict.pop("paths")

    return paths_dict, config["name_experiment"]

    # for k1, v1 in paths_dict.items():
    #     if k1 not in [
    #         "L2_measures",
    #         "dist_cond_measures",
    #         "samples_per_sec",
    #         "L1_measures",
    #         "dist_cond_measures",
    #         "mean_L1",
    #         "mean_L2",
    #         "mean_dist_cond",
    #     ]:
    #         if isinstance(v1, dict):
    #             for k, v in v1.items():
    #                 local_out_dir = os.path.join(
    #                     out_dir, config["name_experiment"], f"{v:.3f}"
    #                 )
    #                 os.makedirs(local_out_dir, exist_ok=True)
    #                 shutil.copy(k, local_out_dir)
    #                 path_gt = os.path.join(
    #                     "/".join(k.split("/")[:-1]), "ground_truth.png"
    #                 )
    #                 shutil.copy(path_gt, local_out_dir)
    #         else:
    #             local_out_dir = os.path.join(
    #                 out_dir, config["name_experiment"], f"{v1:.3f}"
    #             )
    #             os.makedirs(local_out_dir, exist_ok=True)
    #             shutil.copy(k1, local_out_dir)
    #             path_gt = os.path.join("/".join(k1.split("/")[:-1]), "ground_truth.png")
    #             shutil.copy(path_gt, local_out_dir)
    # return paths_dict, config["name_experiment"]


def main_plot(path_logs: list, out_dir: str) -> None:
    measures_dicts = []
    labels = []
    all_metrics = []
    values_table = defaultdict(list)
    for path_log in path_logs:
        tmp_measures, tmp_label = move_plots(path_log, out_dir)
        measures_dicts.append(tmp_measures)
        labels.append(tmp_label)
        tmp_metrics = set([k[:-5] for k in tmp_measures.keys() if "mean" in k])
        all_metrics.append(tmp_metrics)
        values_table["model"].append(LABELS[tmp_label])
    metrics = set.intersection(*all_metrics)

    for metric in set.union(*all_metrics):
        for meas_dict in measures_dicts:
            if metric in meas_dict:
                values_table[metric].append(f"{measures_dict[metric+'_mean']:.4f}")

    cm_ddpm = pl.cm.Reds(
        np.linspace(0.5, 1, len([path for path in path_logs if "ddpm" in path]))
    )
    cm_gan = pl.cm.Greens(
        np.linspace(0.5, 1, len([path for path in path_logs if "gan" in path]))
    )

    for metric in metrics:
        i_ddpm = 0
        i_gan = 0
        colors = []
        fig = plt.figure(figsize=(20, 10))
        ax = plt.subplot(111)
        values = []
        for measures_dict, label in zip(measures_dicts, labels):
            if "ddpm" in label:
                color = cm_ddpm[i_ddpm]
                i_ddpm += 1
            else:
                color = cm_gan[i_gan]
                i_gan += 1
            axis = ax.hist(
                measures_dict[f"{metric}_measures"],
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

            keep = vertices[:, 1] > 0

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
                    np.mean(measures_dict["L2_measures"]),
                    np.mean(measures_dict["dist_cond_measures"]),
                ]
            )
        plt.title(f"Recall curve (CDF) for {metric} metric")
        plt.xlabel(f"{metric} value")
        plt.ylabel("Percentage of images")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(out_dir, f"cum_distrib_{metric}_dist"))
        plt.close()

    with open(os.path.join(out_dir, "table.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=values_table.keys())
        writer.writeheader()
        writer.writerow(values_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_config",
        default="configs_visualize/visualize.yaml",
        required=False,
    )

    args = parser.parse_args()

    config = read_yaml_config_file(args.path_config)
    os.makedirs(config["path_output"], exist_ok=True)
    main_plot(config["paths_logs"], config["path_output"])
