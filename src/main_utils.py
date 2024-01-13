import pytorch_lightning as pl

from src.model.lit_model.lit_models2d import LitDCGAN2d, LitDDPM2d
from src.model.model.DCGAN2d import LargeGeneratorInject2d, LargerDiscriminator2d
from src.model.model.DDPM2d import Diffusion2d
from src.model.model.modules_diffusion2d import UNet_conditional2d, EMA2d

from src.model.model.stylegan_simple import Generator, Discriminator

from src.model.lit_model.lit_model_cvae import LitModelVAE, LitDDPMGrav
from src.model.model.vae_geoph import Encoder, Decoder


def instantiate_lit_model(config, logs_folder=None) -> pl.LightningModule:
    conf_lit_model = config.get("lit_model")
    if "encoding_layer" in conf_lit_model:
        conf_lit_model["encoding_layer"] = eval(conf_lit_model["encoding_layer"])
    if config["lit_model_type"] in ["LitDCGAN", "LitDCGAN2d"]:
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
    elif config["lit_model_type"] in ["LitDDPM", "LitDDPM2d", "LitDDPMGrav"]:
        conf_lit_model["ema"] = eval(conf_lit_model["ema"])
        conf_lit_model["diffusion"] = eval(conf_lit_model["diffusion"])
        conf_lit_model["ddpm"] = eval(conf_lit_model["ddpm"])
    elif (
        config["lit_model_type"] == "LitModelVAE"
        and "encoder" in conf_lit_model
        and "decoder" in conf_lit_model
    ):
        conf_lit_model["encoder"] = eval(conf_lit_model["encoder"])
        conf_lit_model["decoder"] = eval(conf_lit_model["decoder"])
    return eval(config["lit_model_type"])(log_dir=logs_folder, **conf_lit_model)
