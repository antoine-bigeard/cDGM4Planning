import sys

sys.path.insert(0, "/home/abigeard/RA_CCS/DeepGenerativeModelsCCS")
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.data.datamodule.datamodule import MyDataModule
from src.utils import read_yaml_config_file
from src.main_utils import instantiate_lit_model


def test(config, datamodule, path_ckpt, n_obs, path_output):
    config["trainer"]["devices"] = [3]
    lit_model = instantiate_lit_model(config)
    lit_model = lit_model.load_from_checkpoint(path_ckpt)

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
    trainer = pl.Trainer(
        logger=tsboard_logger,
        **config.get("trainer"),
        enable_progress_bar=True,
    )
    datamodule.setup("test")
    figs, paths = lit_model.test_model(
        datamodule,
        n_obs,
        path_output,
    )

    return figs, paths
    # trainer.predict(lit_model, datamodule)


if __name__ == "__main__":
    n_obs = [1, 2, 3, 5, 8, 12, 15, 18, 20]
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
    figs, paths = []
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

        tmp_figs, tmp_paths = test(
            config,
            datamodule,
            path_ckpt,
            n_obs,
            os.path.join(
                path_output,
                config["name_experiment"],
            ),
        )

        figs.append(tmp_figs)
        paths.append(tmp_paths)
