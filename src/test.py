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
import matplotlib.cm as cm


def test(config, datamodule, path_ckpt, n_obs, path_output):
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

    # trainer.predict(lit_model, datamodule)


if __name__ == "__main__":
    n_obs = [1, 2, 3, 5, 8, 12, 16, 20]
    path_logs = [
        "logs/ore_maps_gan_fullSN_wgp/version_0",
        "logs/ore_maps_gan_2inject_wgp/version_0",
        "logs/ore_maps_ddpm_100/version_0",
        "logs/ore_maps_ddpm_250/version_0",
        "logs/ore_maps_ddpm_500/version_0",
        "logs/ore_maps_ddpm_1000/version_0",
    ]
    path_confs = [
        "configs_runs/gan_ore_maps_fullSN_wgp.yaml",
        "configs_runs/gan_ore_maps_2inject_wgp.yaml",
        "configs_runs/ddpm_ore_maps_100.yaml",
        "configs_runs/ddpm_ore_maps_250.yaml",
        "configs_runs/ddpm_ore_maps_500.yaml",
        "configs_runs/ddpm_ore_maps_1000.yaml",
    ]
    path_output = "logs/test"

    os.makedirs(path_output, exist_ok=True)
    # figs, paths = []
    metrics_samples = []
    ground_truths = []
    y_1_idxs = []
    metrics_measures = []
    for log, conf in zip(path_logs, path_confs):
        # config = read_yaml_config_file(os.path.join(log, "config.yaml"))
        config = read_yaml_config_file(conf)
        ckpts = os.listdir(os.path.join(log, "checkpoints"))

        path_ckpt = os.path.join(log, "checkpoints", ckpts[0])
        for ckpt in ckpts:
            if "best" in ckpt:
                path_ckpt = os.path.join(log, "checkpoints", ckpt)

        datamodule = MyDataModule(**config.get("datamodule"))
        datamodule.setup(stage="test")

        x, tmp_y_1_idx, tmp_metrics_samples, tmp_metrics_measures = test(
            config,
            datamodule,
            path_ckpt,
            n_obs,
            os.path.join(
                path_output,
                config["name_experiment"],
            ),
        )

        torch.cuda.empty_cache()

        metrics_samples.append(tmp_metrics_samples)
        ground_truths.append(x)
        y_1_idxs.append(tmp_y_1_idx)
        metrics_measures.append(tmp_metrics_measures)

    for name_metric in metrics_samples[0].keys():
        fig, axs = plt.subplots(
            len(n_obs),
            len(path_confs) + 1,
            figsize=(50, 50),
            # gridspec_kw={
            #     # "width_ratios": [20] * (len(path_confs) + 1),
            #     # "height_ratios": [20] * len(n_obs),
            #     "wspace": 0,
            #     "hspace": 0.1,
            # },
        )
        # fig.tight_layout()
        fig.suptitle(f"Results for {name_metric} metric")
        for i in range(len(n_obs)):
            plot_fig_2D(
                axs[i][0],
                ground_truths[0][i],
                y_1_idxs[0],
                i,
            )
            # sample = ground_truths[0][i]
            # y_1_idx = y_1_idxs[0]
            # cmap = cm.viridis
            # i = 0
            # axs[i, 0].imshow(sample.squeeze().detach().cpu(), cmap=cmap)
            # for j in range(len(y_1_idx[0][i])):
            #     observation_pt = (y_1_idx[0][i][j].cpu(), y_1_idx[1][i][j].cpu())
            #     axs[i, 0].scatter(
            #         [observation_pt[0][1]],
            #         [observation_pt[0][0]],
            #         color=cmap(observation_pt[1]),
            #         marker=".",
            #         s=150,
            #         edgecolors="black",
            #     )
            for j in range(1, len(path_confs) + 1):
                plot_fig_2D(
                    axs[i][j],
                    metrics_samples[j - 1][name_metric][i],
                    y_1_idxs[j - 1],
                    i,
                )
                axs[i, j].set_title(path_confs[j - 1].split("/")[1])
                # sample = ground_truths[0][i]
                # y_1_idx = y_1_idxs[0]
                # cmap = cm.viridis
                # i = 0
                # axs[i, 0].imshow(sample.squeeze().detach().cpu(), cmap=cmap)
                # for j in range(len(y_1_idx[0][i])):
                #     observation_pt = (y_1_idx[0][i][j].cpu(), y_1_idx[1][i][j].cpu())
                #     axs[i, 0].scatter(
                #         [observation_pt[0][1]],
                #         [observation_pt[0][0]],
                #         color=cmap(observation_pt[1]),
                #         marker=".",
                #         s=150,
                #         edgecolors="black",
                #     )
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.savefig(os.path.join(path_output, f"table_results_{name_metric}.jpg"))
