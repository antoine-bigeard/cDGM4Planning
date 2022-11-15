from matplotlib import scale
import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
from src.utils import read_yaml_config_file
import torch.nn as nn

from torch.nn.utils.parametrizations import spectral_norm

from src.model.model.blocks import *


class LargeGeneratorInject2d(nn.Module):
    def __init__(
        self,
        latent_dim,
        layers=[
            {"Cv": (128, 128, 4, 2, 1), "BN": None, "LR": None},
            {"CvT": (128, 256, 4, 2, 1), "BN": None, "LR": None, "D": (0.25,)},
            {"CvT": (256, 512, 4, 2, 1), "BN": None, "LR": None},
            {"Cv": (512, 256, 4, 2, 1), "LR": None},
            {"Cv": (256, 128, 3, 1, 1), "LR": None},
            {"Cv": (128, 128, 3, 1, 1), "LR": None},
            {"Cv": (128, 1, 3, 1, 1), "LR": None},
        ],
        injections=[
            (True, 0, 0.5),  # (use y, out_ch_y, scale_fact)
            (False, 2, 2),
            (False, 2, 2),
            (False, 2, 0.5),
            (True, 0, 1),
            (False, 2, 1),
            (False, 2, 1),
        ],  # (True to inject y, number of channels to use for convolution before injection (if 0 no convolution used))
        encoding_layer=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.injections = injections
        self.layers = layers
        self.encoding_layer = encoding_layer

        self.latent = nn.Sequential(
            nn.ConvTranspose2d(1, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common = nn.Sequential(
            *[
                BasicBlock(layer_params, self.injections[i])
                for i, layer_params in enumerate(self.layers)
            ]
        )

        self.trans_y = nn.Sequential(
            *[BasicBlockY2d(inject) for inject in self.injections]
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        y = y.cuda().float()
        z = z.float()
        if self.encoding_layer is not None:
            y = self.encoding_layer(y).cuda()
        z = self.latent(z.cuda())
        out = z
        for i in range(len(self.injections)):
            if self.injections[i][0]:
                new_y = self.trans_y[i](y)
                out = torch.cat([out, new_y], dim=1)
                out = self.common[i](out)
            else:
                out = self.common[i](out)
        return out


class LargerDiscriminator2d(nn.Module):
    def __init__(
        self,
        layers=[
            {"Cv": (3, 128, 3, 2, 1)},
            {"LR": (0.2,), "D": (0.4,)},
            {"Cv": (128, 256, 3, 2, 1)},
            {"LR": (0.2,), "D": (0.4,), "LR": (0.2,)},
            {"Cv": (256, 256, 3, 2, 1)},
            {"D": (0.4,)},
            {"F": None},
            {"L": (2048, 1)},
        ],
        spectral_norm=[False, False, False, False, False, False, False, False],
        sequential_cond: bool = False,
        encoding_layer=None,
    ):
        super().__init__()
        self.layers = layers
        self.spectral_norm = spectral_norm
        self.encoding_layer = encoding_layer

        self.common = nn.Sequential(
            *[
                SpectralNorm(BasicBlock(layer_params))
                if self.spectral_norm[i]
                else BasicBlock(layer_params)
                for i, layer_params in enumerate(self.layers)
            ]
        )

    def forward(self, x, y):
        x = x.float()
        y = y.float()
        if self.encoding_layer is not None:
            y = self.encoding_layer(y).cuda()
        input_disc = torch.cat([x, y], dim=1)
        return self.common(input_disc)
