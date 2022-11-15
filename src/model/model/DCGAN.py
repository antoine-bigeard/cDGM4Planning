from matplotlib import scale
import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
from src.utils import read_yaml_config_file
import torch.nn as nn

from torch.nn.utils.parametrizations import spectral_norm

from src.model.model.blocks import *


class LargeGeneratorInject(nn.Module):
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
            nn.Linear(self.latent_dim, 8192),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common = nn.Sequential(
            *[
                BasicBlock(layer_params, self.injections[i])
                for i, layer_params in enumerate(self.layers)
            ]
        )

        self.trans_y = nn.Sequential(
            *[BasicBlockY(inject) for inject in self.injections]
        )

    def inference(self, y: torch.Tensor, latent_dim=20):
        z = torch.randn(y.shape[0], latent_dim).cuda()
        return self.forward(z, y)

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        y = y.cuda()
        z = z
        if self.encoding_layer is not None:
            y = self.encoding_layer(y).cuda()
        z = self.latent(z.cuda()).view([z.size(0), 128, 64])
        out = z
        for i in range(len(self.injections)):
            if self.injections[i][0]:
                new_y = self.trans_y[i](y)
                out = torch.cat([out, new_y], dim=1)
                out = self.common[i](out)
            else:
                out = self.common[i](out)
        return out


class LargerDiscriminator(nn.Module):
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
        # self.common = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=3,
        #         out_channels=128,
        #         kernel_size=3,
        #         stride=2,
        #         padding=1,
        #     ),
        #     nn.LeakyReLU(0.2, inplace=False),
        #     nn.Dropout(0.4, inplace=False),
        #     nn.Conv1d(
        #         in_channels=128,
        #         out_channels=256,
        #         kernel_size=3,
        #         stride=2,
        #         padding=1,
        #     ),
        #     nn.LeakyReLU(0.2, inplace=False),
        #     nn.Dropout(0.4, inplace=False),
        #     nn.LeakyReLU(0.2, inplace=False),
        #     nn.Conv1d(
        #         in_channels=256,
        #         out_channels=256,
        #         kernel_size=3,
        #         stride=2,
        #         padding=1,
        #     ),
        #     nn.Dropout(0.4, inplace=False),
        #     Flatten(),
        #     spectral_norm(nn.Linear(2048, 1))
        #     if self.spec_norm_lin
        #     else nn.Linear(2048, 1),
        # )

    def forward(self, x, y):
        if self.encoding_layer is not None:
            y = self.encoding_layer(y).cuda()
        input_disc = torch.cat([x, y], dim=1)
        return self.common(input_disc)

        # self.common = nn.Sequential(
        #     nn.Sequential(
        #         nn.Conv1d(
        #             in_channels=128 + test0(self.injections[0][1], 2)
        #             if self.injections[0][0]
        #             else 128,
        #             out_channels=128,
        #             stride=2,
        #             kernel_size=4,
        #             padding=1,
        #         ),
        #         nn.BatchNorm1d(128),
        #         nn.LeakyReLU(),
        #     ),
        #     nn.Sequential(
        #         nn.ConvTranspose1d(
        #             in_channels=128 + test0(self.injections[1][1], 2)
        #             if self.injections[1][0]
        #             else 128,
        #             out_channels=256,
        #             stride=2,
        #             kernel_size=4,
        #             padding=1,
        #         ),
        #         nn.BatchNorm1d(256),
        #         nn.LeakyReLU(),
        #         nn.Dropout(0.25, inplace=False),
        #     ),
        #     nn.Sequential(
        #         nn.ConvTranspose1d(
        #             in_channels=256 + test0(self.injections[2][1], 2)
        #             if self.injections[2][0]
        #             else 256,
        #             out_channels=512,
        #             stride=2,
        #             kernel_size=4,
        #             padding=1,
        #         ),
        #         nn.BatchNorm1d(512),
        #         nn.LeakyReLU(),
        #     ),
        #     nn.Sequential(
        #         nn.Conv1d(
        #             in_channels=512 + test0(self.injections[3][1], 2)
        #             if self.injections[3][0]
        #             else 512,
        #             out_channels=256,
        #             stride=2,
        #             kernel_size=4,
        #             padding=1,
        #         ),
        #         nn.LeakyReLU(),
        #     ),
        #     nn.Sequential(
        #         nn.Conv1d(
        #             in_channels=256 + test0(self.injections[4][1], 2)
        #             if self.injections[4][0]
        #             else 256,
        #             out_channels=128,
        #             stride=1,
        #             kernel_size=3,
        #             padding=1,
        #         ),
        #         nn.LeakyReLU(),
        #     ),
        #     nn.Sequential(
        #         nn.Conv1d(
        #             in_channels=128 + test0(self.injections[5][1], 2)
        #             if self.injections[5][0]
        #             else 128,
        #             out_channels=128,
        #             stride=1,
        #             kernel_size=3,
        #             padding=1,
        #         ),
        #         nn.LeakyReLU(),
        #     ),
        #     nn.Sequential(
        #         nn.Conv1d(
        #             in_channels=128 + test0(self.injections[6][1], 2)
        #             if self.injections[6][0]
        #             else 128,
        #             out_channels=1,
        #             stride=1,
        #             kernel_size=3,
        #             padding=1,
        #         ),
        #     ),
        # )

        # self.trans_y = nn.Sequential(
        #     nn.Conv1d(2, self.injections[0][1], kernel_size=3, stride=1, padding=1)
        #     if self.injections[0][0] and self.injections[0][1] > 0
        #     else nn.Identity(),
        #     nn.Conv1d(2, self.injections[1][1], kernel_size=4, stride=2, padding=1)
        #     if self.injections[1][0] and self.injections[1][1] > 0
        #     else nn.Identity(),
        #     nn.Conv1d(2, self.injections[2][1], kernel_size=3, stride=1, padding=1)
        #     if self.injections[2][0] and self.injections[2][1] > 0
        #     else nn.Identity(),
        #     nn.ConvTranspose1d(
        #         2, self.injections[3][1], kernel_size=4, stride=2, padding=1
        #     )
        #     if self.injections[3][0] and self.injections[3][1] > 0
        #     else nn.Identity(),
        #     nn.Conv1d(2, self.injections[4][1], kernel_size=3, stride=1, padding=1)
        #     if self.injections[4][0] and self.injections[4][1] > 0
        #     else nn.Identity(),
        #     nn.Conv1d(2, self.injections[5][1], kernel_size=3, stride=1, padding=1)
        #     if self.injections[5][0] and self.injections[5][1] > 0
        #     else nn.Identity(),
        #     nn.Conv1d(2, self.injections[6][1], kernel_size=3, stride=1, padding=1)
        #     if self.injections[6][0] and self.injections[6][1] > 0
        #     else nn.Identity(),
        # )
