import pytorch_lightning as pl
import torch
from random import randrange, random
import torch.nn.functional as F
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
    MetricCollection,
)
import numpy as np
import torchmetrics.functional as tm_metrics
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

from src.model.model.DCGAN import Generator, Discriminator

from src.utils import random_observation

from src.utils import categories


class LitDCGAN(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        g_lr: float = 0.0002,
        d_lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 256,
        validation_shape: np.ndarray = [5, 51, 1],
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.b1 = (b1,)
        self.b2 = (b2,)
        self.batch_size = batch_size
        self.validation_shape = validation_shape

        # networks
        self.generator = Generator(
            latent_dim=self.latent_dim,
        )
        self.discriminator = Discriminator()

        self.validation_z = torch.randn(self.validation_shape[0], self.latent_dim)
        self.validation_y = random_observation(self.validation_shape)

    def forward(self, z, y):
        return self.generator(z, y)

    def g_criterion(self):
        pass

    def d_criterion(self):
        pass

    def generator_step(self, x):
        z = torch.randn(x.shape[0], self.latent_dim)
        y = random_observation(x.shape)

        generated_surface = self(z, y)

        d_output = torch.squeeze(self.discriminator(generated_surface, y))

        g_loss = self.g_criterion(d_output, torch.ones(x.size(0)))

        return g_loss

    def discriminator_step(self, x, y):
        # Loss for the real samples
        d_output = torch.squeeze(self.discriminator(x, y))
        loss_real = self.d_criterion(d_output, torch.ones(x.size(0)))

        # Loss for the generated samples
        z = torch.randn(x.size(0), self.latent_dim)
        y = random_observation(x.shape)
        generated_surfaces = self(z, y)
        d_output = torch.squeeze(self.discriminator(generated_surfaces, y))
        loss_fake = self.d_criterion(d_output, torch.zeros(x.size(0)))

        return loss_real + loss_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        # train generator
        if optimizer_idx == 0:
            g_loss = self.generator_step(x)
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            d_loss = self.generator_step(x, y)
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.g_lr, betas=(self.b1, self.b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.d_lr, betas=(self.b1, self.b2)
        )
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):

        # log sampled images
        sample_surfaces = self(self.validation_z, self.validation_y)

        fig = plt.figure()
        for i, surface in enumerate(sample_surfaces):
            plt.plot(surface)
            observation_pt = 
            plt.plot([self.validation_y[i]])

        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
