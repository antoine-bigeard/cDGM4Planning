import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from src.model.model.DCGAN import Generator, Discriminator

from src.utils import random_observation


class LitDCGAN(pl.LightningModule):
    def __init__(
        self,
        generator: nn.Module = Generator,
        discriminator: nn.Module = Discriminator,
        conf_generator: dict = {"latent_dim": 100},
        conf_discriminator: dict = {},
        g_lr: float = 0.0002,
        d_lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        validation_x_shape: np.ndarray = [5, 51, 1],
        log_dir: str = "",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # instantiate models
        self.conf_generator = conf_generator
        self.conf_discriminator = conf_discriminator
        self.generator = generator(**self.conf_generator)
        self.discriminator = discriminator(**self.conf_discriminator)

        self.latent_dim = self.conf_generator["latent_dim"]
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.b1 = b1
        self.b2 = b2
        self.validation_x_shape = validation_x_shape
        self.log_dir = log_dir

        self.validation_z = torch.randn(
            self.validation_x_shape[0], self.latent_dim
        ).cuda()
        rd_obs = random_observation(self.validation_x_shape, return_1_idx=True)
        self.validation_y, self.y_1_idxs = rd_obs[0].cuda(), rd_obs[1]

    def forward(self, z, y):
        return self.generator(z, y)

    def g_criterion(self, pred, target):
        return F.mse_loss(F.sigmoid(pred), target)

    def d_criterion(self, pred, target):
        return F.mse_loss(F.sigmoid(pred), target)

    def generator_step(self, y, x_shape):
        z = torch.randn(x_shape[0], self.latent_dim).cuda()
        y = random_observation(x_shape).cuda()

        generated_surface = self(z, y)

        d_output = torch.squeeze(self.discriminator(generated_surface, y))

        g_loss = self.g_criterion(d_output, torch.ones(x_shape[0]).cuda())

        return g_loss

    def discriminator_step(self, x, y):
        # Loss for the real samples
        d_output = torch.squeeze(self.discriminator(x, y))
        loss_real = self.d_criterion(d_output, torch.ones(x.size(0)).cuda())

        # Loss for the generated samples
        z = torch.randn(x.size(0), self.latent_dim).cuda()
        y = random_observation(x.shape).cuda()
        generated_surfaces = self(z, y)
        d_output = torch.squeeze(self.discriminator(generated_surfaces, y))
        loss_fake = self.d_criterion(d_output, torch.zeros(x.size(0)).cuda())

        return loss_real + loss_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        # train generator
        if optimizer_idx == 0:
            g_loss = self.generator_step(
                y, x_shape=(y.size(0), *self.validation_x_shape[1:])
            )
            self.log("train/g_loss", g_loss, prog_bar=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            d_loss = self.discriminator_step(x, y)
            self.log("train/d_loss", d_loss, prog_bar=True)
            return d_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # train generator
        g_loss = self.generator_step(
            y, x_shape=(y.size(0), *self.validation_x_shape[1:])
        )
        self.log("val/g_loss", g_loss, prog_bar=True)

        # train discriminator
        d_loss = self.discriminator_step(x, y)
        self.log("val/d_loss", d_loss, prog_bar=True)
        return {"g_loss": g_loss, "d_loss": d_loss}

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.g_lr, betas=(self.b1, self.b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.d_lr, betas=(self.b1, self.b2)
        )
        return [opt_g, opt_d], []

    def on_train_epoch_start(self) -> None:
        sample_surfaces = self(self.validation_z, self.validation_y)
        for i, surface in enumerate(sample_surfaces):
            fig = plt.figure()
            plt.plot(surface.squeeze().detach().cpu())
            observation_pt = (
                self.y_1_idxs[i],
                self.validation_y[i, :, self.y_1_idxs[i]][1].cpu(),
            )
            # plt.plot(observation_pt, "ro", markersize=8)
            plt.savefig(os.path.join(self.log_dir, f"test_{i}"))

            self.logger.experiment.add_figure(
                f"generated_image_test", fig, self.current_epoch
            )

    def on_validation_epoch_end(self):

        # log sampled images
        sample_surfaces = self(self.validation_z, self.validation_y)

        for i, surface in enumerate(sample_surfaces):
            fig = plt.figure()
            plt.plot(surface.squeeze().cpu())
            observation_pt = (
                self.y_1_idxs[i],
                self.validation_y[i, :, self.y_1_idxs[i]][1].cpu(),
            )
            plt.plot(observation_pt, "ro", markersize=8)

            self.logger.experiment.add_figure(
                f"generated_image_{i}", fig, self.current_epoch
            )
