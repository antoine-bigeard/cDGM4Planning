import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
from src.utils import read_yaml_config_file
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 6528),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=130, out_channels=128, stride=2, kernel_size=4
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(
                in_channels=128, out_channels=64, stride=2, kernel_size=4
            ),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.25, inplace=False),
            nn.Conv1d(
                in_channels=64,
                out_channels=32,
                stride=2,
                kernel_size=4,
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=1,
                stride=2,
                kernel_size=4,
            ),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        z = self.latent(z).view([z.size(0), 128, 51])
        input_gen = torch.cat([z, y], dim=1)
        return self.common(input_gen)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.common = nn.Sequential(
            nn.Conv1d(
                in_channels=3,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.4, inplace=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.Dropout(0.4, inplace=False),
            Flatten(),
            nn.Linear(1664, 1),
        )

    def forward(self, x, y):
        input_disc = torch.cat([x, y], dim=1)
        return self.common(input_disc)


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


"""
OTHER VERSION THAT SEEMS TO ADVANTAGEOUS FOR DISCRIMINATOR

import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
from src.utils import read_yaml_config_file
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 6528),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.common = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=130, out_channels=128, stride=2, kernel_size=4
            ),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.ConvTranspose1d(
                in_channels=128, out_channels=64, stride=2, kernel_size=4
            ),
            nn.BatchNorm1d(64),
            # nn.Dropout(0.25, inplace=False),
            nn.Conv1d(
                in_channels=64,
                out_channels=32,
                stride=2,
                kernel_size=4,
            ),
            # nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=1,
                stride=2,
                kernel_size=4,
            ),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        z = self.latent(z).view([z.size(0), 128, 51])
        input_gen = torch.cat([z, y], dim=1)
        return self.common(input_gen)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.common = nn.Sequential(
            nn.Conv1d(
                in_channels=3,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.4, inplace=False),
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=False),
            Flatten(),
            nn.Dropout(0.4, inplace=False),
            nn.Linear(1664, 1),
        )

    def forward(self, x, y):
        input_disc = torch.cat([x, y], dim=1)
        return self.common(input_disc)


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

"""
