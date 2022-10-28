import sys
from pympler.asizeof import asizeof
from torch import manual_seed

sys.path.insert(0, "/home/abigeard/RA_CCS/DeepGenerativeModelsCCS")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from src.data.datamodule.datamodule import MyDataModule
from src.model.lit_model.lit_models import *
from src.model.model.DCGAN import *
from src.model.model.CVAE import *
from src.model.model.DDPM import *
from src.model.model.modules_diffusion import *
from src.utils import read_yaml_config_file

import argparse
import os
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_config",
        help="config path that contains config for data, models, training.",
        default="/home/abigeard/RA_CCS/DeepGenerativeModelsCCS/config/ddpm_test.yaml",
        required=False,
    )
    parser.add_argument(
        "--mode",
        help="mode fit, test or predict",
        default="fit",
        required=False,
    )

    args = parser.parse_args()

    path_config = args.path_config
    mode = args.mode

    config = read_yaml_config_file(path_config)
    checkpoint_path = config.get("checkpoint_path")
    conf_datamodule = config.get("datamodule")
    conf_lit_model = config.get("lit_model")
    conf_trainer = config.get("trainer")
    name_exp = config.get("name_experiment")
    conf_ts_board = config.get("tensorboard_logs")
    conf_checkpoint_callback = config.get("checkpoint_callback")

    seed = config["seed"]
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

    os.system(f"cp src/model/model/DCGAN.py {os.path.join(logs_folder, 'DCGAN.py')}")
    os.system(f"cp src/model/model/blocks.py {os.path.join(logs_folder, 'blocks.py')}")

    # configure data module
    datamodule = MyDataModule(**conf_datamodule)

    # configure lit model
    if config["lit_model_type"] == "LitDCGAN":
        conf_lit_model["generator"] = eval(conf_lit_model["generator"])
        conf_lit_model["discriminator"] = eval(conf_lit_model["discriminator"])
    elif config["lit_model_type"] == "LitVAE":
        conf_lit_model["vae"] = eval(conf_lit_model["vae"])
        conf_lit_model["conf_vae"]["encoder_model"] = eval(
            conf_lit_model["conf_vae"]["encoder_model"]
        )
        conf_lit_model["conf_vae"]["decoder_model"] = eval(
            conf_lit_model["conf_vae"]["decoder_model"]
        )
    elif config["lit_model_type"] == "LitDDPM":
        conf_lit_model["ema"] = eval(conf_lit_model["ema"])
        conf_lit_model["diffusion"] = eval(conf_lit_model["diffusion"])
        conf_lit_model["ddpm"] = eval(conf_lit_model["ddpm"])
    lit_model = eval(config["lit_model_type"])(log_dir=logs_folder, **conf_lit_model)

    # load trained model if checkpoutint is given
    if checkpoint_path is not None:
        lit_model.load_from_checkpoint(checkpoint_path)

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
        # callbacks=[early_stop_callback],
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
