import pytorch_lightning as pl

from src.model.lit_model.lit_models import LitDCGAN, LitDDPM, LitVAE
from src.model.model.DCGAN import LargeGeneratorInject, LargerDiscriminator
from src.model.lit_model.lit_models2d import LitDCGAN2d, LitDDPM2d
from src.model.model.DCGAN2d import LargeGeneratorInject2d, LargerDiscriminator2d
from src.model.model.DDPM import Diffusion
from src.model.model.modules_diffusion import UNet_conditional, EMA
from src.model.model.DDPM2d import Diffusion2d
from src.model.model.modules_diffusion2d import UNet_conditional2d, EMA2d

# from src.model.model.stylegan2 import Generator, Discriminator
# from src.model.model.stylegan_2_ludicrains import Generator, Discriminator
from src.model.model.stylegan_simple import Generator, Discriminator
from src.model.model.gan_resnet import GeneratorResNet, DiscriminatorResNet


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
    elif config["lit_model_type"] in ["LitDDPM", "LitDDPM2d"]:
        conf_lit_model["ema"] = eval(conf_lit_model["ema"])
        conf_lit_model["diffusion"] = eval(conf_lit_model["diffusion"])
        conf_lit_model["ddpm"] = eval(conf_lit_model["ddpm"])
    return eval(config["lit_model_type"])(log_dir=logs_folder, **conf_lit_model)
