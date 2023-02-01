import sys

sys.path.insert(0, "/home/abigeard/RA_CCS/DeepGenerativeModelsCCS")
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.data.datamodule.datamodule import MyDataModule
from src.utils import read_yaml_config_file, plot_fig_2D
from src.main_utils import instantiate_lit_model
import torch
import argparse


def cond_test_model(config, datamodule, path_ckpt, n_obs, path_output):
    config["trainer"]["devices"] = [3]
    lit_model = instantiate_lit_model(config)
    lit_model = lit_model.load_from_checkpoint(path_ckpt).cuda()

    tsboard_logger = TensorBoardLogger(
        config.get("tensorboard_logs")["save_dir"],
        config["name_experiment"],
    )

    log_dir = os.path.join(
        tsboard_logger.save_dir,
        config["name_experiment"],
        f"version_{tsboard_logger.version}",
    )

    lit_model.log_dir = log_dir
    lit_model.n_obs = n_obs
    datamodule.setup("test")
    return lit_model.test_model(
        datamodule,
        n_obs,
        path_output,
    )


def main_test(path_config):
    main_test_config = read_yaml_config_file(path_config)

    save = main_test_config.get("save")
    n_obs = main_test_config.get("n_obs")
    path_logs = main_test_config.get("path_logs")
    path_output = main_test_config.get("path_output")
    path_saved_tests = main_test_config.get("path_saved_tests")

    update_conf_datamodule = main_test_config.get("datamodule")
    update_conf_datamodule = (
        {} if update_conf_datamodule is None else update_conf_datamodule
    )
    update_conf_lit_model = main_test_config.get("lit_model")
    update_conf_lit_model = (
        {} if update_conf_lit_model is None else update_conf_lit_model
    )

    os.makedirs(path_output, exist_ok=True)
    metrics_samples = []
    ground_truths = []
    y_1_idxs = []
    metrics_measures = []
    for log in path_logs:
        # conf = os.path.join(log, "config.yaml")
        if os.path.exists(os.path.join(log, "config.yaml")):
            conf = os.path.join(log, "config.yaml")
        else:
            name_config = [
                f for f in os.listdir(os.path.join(log)) if f[-5:] == ".yaml"
            ][0]
            conf = os.path.join(log, name_config)
        config = read_yaml_config_file(conf)
        ckpts = os.listdir(os.path.join(log, "checkpoints"))

        path_ckpt = os.path.join(log, "checkpoints", ckpts[0])
        for ckpt in ckpts:
            if "best" in ckpt:
                path_ckpt = os.path.join(log, "checkpoints", ckpt)

        conf_datamodule = config.get("datamodule")
        conf_datamodule.update(update_conf_datamodule)
        config["lit_model"].update(update_conf_lit_model)
        datamodule = MyDataModule(**conf_datamodule)
        datamodule.setup(stage="test")

        path_exp = os.path.join(
            path_output,
            config["name_experiment"],
        )
        if path_saved_tests is None:
            x, tmp_y_1_idx, tmp_metrics_samples, tmp_metrics_measures = cond_test_model(
                config,
                datamodule,
                path_ckpt,
                n_obs,
                path_exp,
            )
            if save:
                torch.save(x, os.path.join(path_exp, "x.pt"))
                torch.save(tmp_y_1_idx, os.path.join(path_exp, "y_1_idx.pt"))
                torch.save(
                    tmp_metrics_samples, os.path.join(path_exp, "metrics_samples.pt")
                )
                torch.save(
                    tmp_metrics_measures, os.path.join(path_exp, "metrics_measures.pt")
                )

        else:
            path_exp = os.path.join(path_saved_tests, config["name_experiment"])
            x, tmp_y_1_idx, tmp_metrics_samples, tmp_metrics_measures = (
                torch.load(os.path.join(path_exp, "x.pt")),
                torch.load(os.path.join(path_exp, "y_1_idx.pt")),
                torch.load(os.path.join(path_exp, "metrics_samples.pt")),
                torch.load(os.path.join(path_exp, "metrics_measures.pt")),
            )

        metrics_samples.append(tmp_metrics_samples)
        ground_truths.append(x)
        y_1_idxs.append(tmp_y_1_idx)
        metrics_measures.append(tmp_metrics_measures)

    name_metrics = [m for m in metrics_samples[0].keys() if m != "time_inference"]

    for name_metric in name_metrics:
        fig, axs = plt.subplots(
            len(n_obs),
            len(path_logs) + 1,
            figsize=(50, 50),
        )
        fig.suptitle(f"Results for {name_metric} metric")
        for i in range(len(n_obs)):
            plot_fig_2D(
                axs[i][0],
                ground_truths[0][i],
                y_1_idxs[0],
                i,
            )
            for j in range(1, len(path_logs) + 1):
                plot_fig_2D(
                    axs[i][j],
                    metrics_samples[j - 1][name_metric][i],
                    y_1_idxs[j - 1],
                    i,
                )
                axs[i, j].set_title(
                    f"{name_metric}: {metrics_measures[j - 1][name_metric+'_min'][i]:.4f}",
                )
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.savefig(os.path.join(path_output, f"table_results_{name_metric}.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_config",
        help="config path that contains config to test the models on various conditions.",
        default="configs_cond_tests/test_specific.yaml",
        required=False,
    )
    args = parser.parse_args()

    main_test(args.path_config)
