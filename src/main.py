import sys

sys.path.insert(0, "/home/abigeard/RA_CCS/DeepGenerativeModelsCCS")

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
    conf_lit_model = config.get("lit_model")
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

    # os.system(f"cp src/model/model/DCGAN.py {os.path.join(logs_folder, 'DCGAN.py')}")
    # os.system(f"cp src/model/model/blocks.py {os.path.join(logs_folder, 'blocks.py')}")

    # configure data module
    datamodule = MyDataModule(**conf_datamodule)

    # configure lit model
    # if "encoding_layer" in conf_lit_model:
    #     conf_lit_model["encoding_layer"] = eval(conf_lit_model["encoding_layer"])
    # if config["lit_model_type"] in ["LitDCGAN", "LitDCGAN2d"]:
    #     conf_lit_model["generator"] = eval(conf_lit_model["generator"])
    #     conf_lit_model["discriminator"] = eval(conf_lit_model["discriminator"])
    # elif config["lit_model_type"] == "LitVAE":
    #     conf_lit_model["vae"] = eval(conf_lit_model["vae"])
    #     conf_lit_model["conf_vae"]["encoder_model"] = eval(
    #         conf_lit_model["conf_vae"]["encoder_model"]
    #     )
    #     conf_lit_model["conf_vae"]["decoder_model"] = eval(
    #         conf_lit_model["conf_vae"]["decoder_model"]
    #     )
    # elif config["lit_model_type"] in ["LitDDPM", "LitDDPM2d"]:
    #     conf_lit_model["ema"] = eval(conf_lit_model["ema"])
    #     conf_lit_model["diffusion"] = eval(conf_lit_model["diffusion"])
    #     conf_lit_model["ddpm"] = eval(conf_lit_model["ddpm"])
    # lit_model = eval(config["lit_model_type"])(log_dir=logs_folder, **conf_lit_model)

    lit_model = instantiate_lit_model(config, logs_folder=logs_folder)

    # load trained model if checkpoutint is given
    if checkpoint_path is not None:
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

    # parser.add_argument(
    #     "--path_config",
    #     help="config path that contains config for data, models, training.",
    #     default=[
    #         # "/home/abigeard/RA_CCS/DeepGenerativeModelsCCS/config/ddpm_ore_maps.yaml",
    #         # "/home/abigeard/RA_CCS/DeepGenerativeModelsCCS/config/gan_ore_maps.yaml",
    #         # "configs_runs/gan_ore_maps_fullinject.yaml",
    #         "configs_runs/ddpm_ore_maps_1000.yaml"
    #     ],
    #     required=False,
    # )
    parser.add_argument(
        "--path_config",
        help="config path that contains config for data, models, training.",
        default="configs_runs/style-gan_simple.yaml",
        required=False,
    )
    parser.add_argument(
        "--parallel",
        help="mode fit, test or predict",
        default=0,
        required=False,
    )

    args = parser.parse_args()

    config = read_yaml_config_file(args.path_config)

    main(config)
    # if args.parallel:
    #     pool = multiprocessing.Pool()
    #     outs = pool.map(main, args.path_config)
    # else:
    #     for path in args.path_config:
    #         main(path)
