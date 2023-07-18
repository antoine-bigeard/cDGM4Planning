import pytorch_lightning as pl

from src.model.lit_model.lit_models2d import LitDCGAN2d, LitDDPM2d
from src.model.lit_model.lit_models1d import (
    LitDDPM1d,
    LitDDPM1dSeq2Seq,
    LitDDPM1dSeq2Seq2,
    LitTransformer,
    LitCrossAttentionDDPM,
)
from src.model.model.DCGAN2d import LargeGeneratorInject2d, LargerDiscriminator2d
from src.model.model.DDPM2d import Diffusion2d
from src.model.model.modules_diffusion2d import UNet_conditional2d, EMA2d
from src.model.model.DDPM1d import (
    Diffusion,
    DiffusionTransformer,
    DiffusionTransformer2,
)
from src.model.model.modules_diffusion1d import UNet_conditional, EMA
from src.model.stable_diffusion.diffusion import DiffusionUNet

from src.model.model.stylegan_simple import Generator, Discriminator

from src.model.model.transformer import (
    Transformer4Input,
    Transformer4DDPM,
    TransformerAlone,
    Transformer4DDPM2,
    Transformer4DDPM3,
    Transformer4DDPM4,
    TestModel,
    Transformer4Diffusion,
)

from src.model.model.diffusion_transformer import DiT

from src.model.lit_model.lit_models1d_fullddpm import LitDDPM1dFull

from stable_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion, DDPM
from stable_diffusion_2d.ldm.models.diffusion.ddpm import LatentDiffusion2d, DDPM2d


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
    elif config["lit_model_type"] in [
        "LitDDPM1d",
        "LitDDPM2d",
        "LitDDPM1dSeq2Seq",
        "LitDDPM1dSeq2Seq2",
        "LitCrossAttentionDDPM",
        "LitDDPM1dFull",
    ]:
        conf_lit_model["ema"] = eval(conf_lit_model["ema"])
        conf_lit_model["diffusion"] = eval(conf_lit_model["diffusion"])
        conf_lit_model["ddpm"] = eval(conf_lit_model["ddpm"])
    elif config["lit_model_type"] in ["LitTransformer"]:
        conf_lit_model["transformer"] = eval(conf_lit_model["transformer"])
    elif config["lit_model_type"] in [
        "LatentDiffusion",
        "DDPM",
        "LatentDiffusion2d",
        "DDPM2d",
    ]:
        lit_model = eval(config["lit_model_type"])(
            log_dir=logs_folder,
            **conf_lit_model,
        )
        lit_model.learning_rate = config["learning_rate"]
        return lit_model
    return eval(config["lit_model_type"])(
        log_dir=logs_folder,
        **conf_lit_model,
        # cuda_device_idx=config.get("trainer")["devices"][0],
    )
