import sys
import shutil

sys.path.insert(0, ".")
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

import matplotlib.pyplot as plt
import argparse
import os
import json
import csv
import numpy as np
from src.utils import read_yaml_config_file
import matplotlib.pylab as pl

import tikzplotlib

from collections import defaultdict


def move_plots(path_log: str, out_dir: str) -> tuple[dict, str]:
    config = read_yaml_config_file(os.path.join(path_log, "config.yaml"))
    out_plots_dir = os.path.join(path_log, config["name_experiment"])
    for f in os.listdir(path_log):
        if ".png" in f:
            shutil.copyfile(os.path.join(path_log, f), out_plots_dir)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    shutil.copy(
        os.path.join(path_log, "metrics_img_path.json"),
        os.path.join(out_dir, "metrics", config["name_experiment"] + ".json"),
    )
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


def main_plot(path_logs: list, out_dir: str) -> None:
    measures_dicts = []
    labels = []
    all_metrics = []
    values_table = defaultdict(list)
    for path_log in path_logs:
        if isinstance(path_log, list):
            tmp_label = path_log[1]
            path_log = path_log[0]
            tmp_measures, _ = move_plots(path_log, out_dir)
        else:
            tmp_measures, tmp_label = move_plots(path_log, out_dir)
        measures_dicts.append(tmp_measures)
        labels.append(tmp_label)
        tmp_metrics = set([k for k in tmp_measures.keys() if "measures" not in k])
        all_metrics.append(tmp_metrics)
        values_table["model"].append(tmp_label)
    metrics = set.intersection(*all_metrics)

    for metric in set.union(*all_metrics):
        for meas_dict in measures_dicts:
            if metric in meas_dict:
                values_table[metric].append(f"{meas_dict[metric]:.4f}")

    cm_ddpm = pl.cm.Reds(
        np.linspace(0.5, 1, len([path for path in path_logs if "ddpm" in path[0]]))
    )
    cm_gan = pl.cm.Greens(
        np.linspace(0.5, 1, len([path for path in path_logs if "gan" in path[0]]))
    )

    # metric = list(metrics)[0]
    # data, label = measures_dicts[0], labels[0]
    # plt.figure()
    # hist = np.histogram(
    #     data[f"{metric[:-5]}_measures"][:-1],
    #     bins=100,
    #     density=True,
    # )
    # hist = (np.cumsum(hist[0]) * (hist[1][1] - hist[1][0]), hist[1])
    # num_bins = 100
    # bin_edges = np.linspace(
    #     min(data[f"{metric[:-5]}_measures"]),
    #     max(data[f"{metric[:-5]}_measures"]),
    #     num_bins + 1,
    # )

    # Compute the histogram
    # bin_counts, _ = np.histogram(data[f"{metric[:-5]}_measures"], bins=bin_edges)
    # bin_counts = np.cumsum(bin_counts)
    # bin_counts = bin_counts / bin_counts[-1]
    # Compute the bar positions and widths
    # bar_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    # bar_widths = bin_edges[1:] - bin_edges[:-1]

    # Plot the bars
    # plt.plot(bar_centers, bin_counts, drawstyle="steps-mid", color="blue", linewidth=1)
    # plt.stairs(*hist)
    # plt.plot(hist[0], hist[1][:-1])
    # tikzplotlib.save(os.path.join(out_dir, f"zztest.tex"))
    # plt.savefig(os.path.join(out_dir, f"zztest.png"))

    for metric in metrics:
        i_ddpm = 0
        i_gan = 0
        colors = []
        fig = plt.figure(figsize=(20, 10))
        ax = plt.subplot(111)
        values = []
        for measures_dict, label in zip(measures_dicts, labels):
            if "DDPM" in label:
                color = cm_ddpm[i_ddpm]
                i_ddpm += 1
            else:
                color = cm_gan[i_gan]
                i_gan += 1

            num_bins = 100
            bin_edges = np.linspace(
                min(measures_dict[f"{metric[:-5]}_measures"]),
                max(measures_dict[f"{metric[:-5]}_measures"]),
                num_bins + 1,
            )
            bin_counts, _ = np.histogram(
                measures_dict[f"{metric[:-5]}_measures"], bins=bin_edges
            )
            bin_counts = np.cumsum(bin_counts)
            bin_counts = bin_counts / bin_counts[-1]
            bar_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            bar_widths = bin_edges[1:] - bin_edges[:-1]
            print(label)
            plt.plot(
                bar_centers,
                bin_counts,
                drawstyle="steps-mid",
                color=color,
                linewidth=1,
                label=label,
            )
            # axis = ax.hist(
            #     measures_dict[f"{metric[:-5]}_measures"],
            #     bins=100,
            #     density=True,
            #     histtype="step",
            #     cumulative=True,
            # )
            # colors.append(
            #     plt.plot(
            #         [],
            #         color=color,
            #         label=label,
            #     )[0]
            # )
            # poly = axis[2][0]
            # vertices = poly.get_path().vertices

            # keep = vertices[:, 1] > 0

            # new_poly = plt.Polygon(
            #     vertices[keep],
            #     closed=False,
            #     fill=False,
            #     edgecolor=color,
            #     linewidth=poly.get_linewidth(),
            # )
            # poly.set_visible(False)
            # ax.add_artist(new_poly)
            # plt.draw()
            values.append(
                [
                    label,
                    np.mean(measures_dict["L2_min_measures"]),
                    np.mean(measures_dict["dist_cond_mean_measures"]),
                ]
            )
        plt.title(f"Recall curve (CDF) for {metric} metric")
        plt.xlabel(f"{metric} value")
        plt.ylabel("Percentage of images")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        tikzplotlib.clean_figure()
        tikzplotlib.save(
            os.path.join(out_dir, f"cum_distrib_{metric}_dist.tex"),
            strict=True,
        )
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(out_dir, f"cum_distrib_{metric}_dist.png"))
        plt.savefig(os.path.join(out_dir, f"cum_distrib_{metric}_dist") + ".pgf")
        # tikzplotlib.save(os.path.join(out_dir, f"cum_distrib_{metric}_dist") + ".tex")
        plt.close()

    with open(os.path.join(out_dir, "table.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(values_table.keys())
        writer.writerows(zip(*values_table.values()))


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
