import sys

sys.path.insert(0, ".")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from src.data.datamodule.datamodule import MyDataModule
from src.utils import read_yaml_config_file, update
from src.main_utils import instantiate_lit_model

import argparse
import os
import yaml
import torch


def main(config):

    mode = config["mode"]
    checkpoint_path = config.get("checkpoint_path")
    conf_datamodule = config.get("datamodule")
    conf_trainer = config.get("trainer")
    conf_ts_board = config.get("tensorboard_logs")
    conf_checkpoint_callback = config.get("checkpoint_callback")

    seed = config["seed"]
    if seed is not None:
        torch.manual_seed(seed)

    tsboard_logger = TensorBoardLogger(
        conf_ts_board["save_dir"],
        config["name_experiment"],
    )

    logs_folder = os.path.join(
        tsboard_logger.save_dir,
        config["name_experiment"],
        f"version_{tsboard_logger.version}",
    )
    os.makedirs(logs_folder, exist_ok=True)

    with open(os.path.join(logs_folder, "config.yaml"), "w") as dst:
        yaml.dump(config, dst)

    # configure data module
    datamodule = MyDataModule(**conf_datamodule)

    lit_model = instantiate_lit_model(config, logs_folder=logs_folder)

    # load trained model if checkpoutint is given
    if checkpoint_path is not None:
        print("Logging checkpoint found at: ", checkpoint_path)
        try:
            lit_model = lit_model.load_from_checkpoint(
                checkpoint_path, **lit_model.hparams
            )
        except:
            lit_model = lit_model.load_from_checkpoint(checkpoint_path)
        lit_model.log_dir = logs_folder

    early_stop_callback = EarlyStopping(
        monitor="train/loss",
        min_delta=0.01,
        patience=10,
        verbose=False,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logs_folder, "checkpoints"),
        verbose=True,
        **conf_checkpoint_callback,
    )

    trainer = pl.Trainer(
        logger=tsboard_logger,
        callbacks=[checkpoint_callback],
        **conf_trainer,
        enable_progress_bar=True,
    )

    if mode == "fit":
        trainer.fit(lit_model, datamodule)

    elif mode == "test":
        trainer.test(lit_model, datamodule)

    elif mode == "predict":
        trainer.predict(lit_model, datamodule)

    else:
        raise ValueError("Please give a valid mode: fit, test or predict")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_config",
        help="config path that contains config for data, models, training.",
        default="configs_runs/ddpms/ddpm_ore_maps_100.yaml",
        required=False,
    )

    args = parser.parse_args()

    config = read_yaml_config_file(args.path_config)

    main(config)
