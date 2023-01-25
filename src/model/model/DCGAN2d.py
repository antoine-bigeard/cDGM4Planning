from matplotlib import scale
import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
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
        latent_1d=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.injections = injections
        self.layers = layers
        self.encoding_layer = encoding_layer
        self.latent_1d = latent_1d

        self.latent = (
            nn.Sequential(
                nn.Linear(self.latent_dim, 32 * 32 * 32),
                View([32, 32, 32]),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(32, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
            )
            if self.latent_1d
            else nn.Sequential(
                nn.Conv2d(1, 256, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=False),
            )
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

    def inference(self, y: torch.Tensor, latent_dim=20):
        with torch.inference_mode():
            if self.latent_1d:
                z = torch.randn(y.shape[0], 1, latent_dim).cuda()
            else:
                z = torch.randn(y.shape[0], 1, latent_dim, latent_dim).cuda()
            return self.forward(z, y)

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
        no_inject: bool = False,
    ):
        super().__init__()
        self.layers = layers
        self.spectral_norm = spectral_norm
        self.encoding_layer = encoding_layer
        self.no_inject = no_inject

        self.common = nn.Sequential(
            *[
                BasicBlock(
                    layer_params, injection=None, spectral_norm=self.spectral_norm[i]
                )
                for i, layer_params in enumerate(self.layers)
            ]
        )

    def forward(self, x, y):
        x = x.float()
        y = y.float()
        if self.encoding_layer is not None:
            y = self.encoding_layer(y).cuda()
        if self.no_inject:
            input_disc = x
        else:
            input_disc = torch.cat([x, y], dim=1)
        return self.common(input_disc)
