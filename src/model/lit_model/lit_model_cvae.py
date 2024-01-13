from typing import Any
import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy

from collections import defaultdict
from src.model.model.vae_geoph import Encoder, Decoder, VAE, ConvVAE
from src.utils import calculate_gravity_matrix, plot_density_maps
from src.model.model.DDPM2d import Diffusion2d
from src.model.model.modules_diffusion2d import UNet_conditional2d, EMA2d
from src.model.model.blocks import SimpleEncodingGrav


def ConvBlock(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def DeconvBlock(in_channels, out_channels, kernel_size, last=False):
    if not last:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
        ),
        nn.Tanh(),
    )


class Stack(nn.Module):
    def __init__(self, channels, height, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)


class LitModelVAE(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        lr: float,
        batch_size: int,
        n_sample_for_metric: int,
        log_dir: str = "",
        conditional=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.latent_dim = latent_dim
        self.conditional = conditional

        self.vae = ConvVAE(z_dim=latent_dim, conditional=conditional).cuda()

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # self.gravity_matrix = self.gravity_matrix.to(self.device)
        # self.train_test_batch = self.trainer.datamodule.train_dataset[:5]
        self.train_test_batch = [
            self.trainer.datamodule.val_dataset.__getitem__(i) for i in range(5)
        ]
        train_test_batch = torch.cat(
            [torch.Tensor(d).unsqueeze(0) for d, _ in self.train_test_batch]
        ), torch.cat([torch.Tensor(d).unsqueeze(0) for _, d in self.train_test_batch])
        self.train_test_batch = (
            train_test_batch
            if len(train_test_batch[0].shape) == 4
            else (
                train_test_batch[0].unsqueeze(1),
                train_test_batch[1],
            )
        )
        # self.train_test_batch = torch.cat(
        #     [d for d, _ in self.train_test_batch], dim=0
        # ), torch.tensor([d for _, d in self.train_test_batch])

    def forward(self, x):
        return self.vae(x)

    def loss_function(self, recon_x, x, mu, log_var):
        # BCE = F.binary_cross_entropy(
        #     recon_x.view(-1, 784), x.view(-1, 784), reduction="sum"
        # )
        BCE = F.mse_loss(recon_x.view(-1, 784), x.view(-1, 784), reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def training_step(self, batch, batch_idx):
        x, y = batch
        if len(x.shape) < 4:
            x = x.float().unsqueeze(1)

        recon_batch, mu, log_var = self(x)
        loss = self.loss_function(recon_batch, x, mu, log_var) / x.shape[0]

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if len(x.shape) < 4:
            x = x.float().unsqueeze(1)

        recon_batch, mu, log_var = self(x)
        loss = self.loss_function(recon_batch, x, mu, log_var) / x.shape[0]

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_validation_epoch_end(self) -> None:
        x, y = self.train_test_batch
        x = torch.Tensor(x).float().to(self.device)
        y = torch.Tensor(y).float().to(self.device)
        l = [None] * len(x)
        x = x.float()
        y = y.float()

        x_hat, mu, log_var = self(x)

        plot_density_maps(
            x.squeeze().detach().cpu().numpy(),
            l,
            32,
            log_dir=self.log_dir,
            curr_epoch=self.current_epoch,
            suffix="real",
        )

        plot_density_maps(
            x_hat.squeeze().detach().cpu().numpy(),
            l,
            32,
            log_dir=self.log_dir,
            curr_epoch=self.current_epoch,
            suffix="gen",
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=(self.lr or self.learning_rate)
        )
        return {
            "optimizer": optimizer,
            "monitor": "val/loss",
        }


class LitModelConvVAE(LitModelVAE):
    def __init__(
        self,
        latent_dim: int,
        input_height: int,
        input_width: int,
        input_channels: int,
        lr: float,
        batch_size: int,
        n_sample_for_metric: int,
    ):
        LATENT_DIM = 1024
        self.latent_dim = LATENT_DIM

        self.save_hyperparameters()
        self.lr = lr

        self.batch_size = batch_size

        assert input_height == input_width

        final_height = 32
        self.final_height = final_height
        final_width = 32
        self.final_width = final_width

        self.encoder = nn.Sequential(
            ConvBlock(input_channels, 32, 4),
            ConvBlock(32, 64, 4),
            ConvBlock(64, 128, 4),
            ConvBlock(128, 256, 4),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            Stack(1024, 1, 1),
            DeconvBlock(1024, 128, 5),
            DeconvBlock(128, 64, 5),
            DeconvBlock(64, 32, 6),
            DeconvBlock(32, input_channels, 6, last=True),
        )

        self.hidden2mu = nn.Linear(256 * final_height**2, 256 * final_height**2)
        self.hidden2log_var = nn.Linear(
            256 * final_height**2, 256 * final_height**2
        )

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))


"""
class LitModelVAE(pl.LightningModule):
    def __init__(
        self,
        encoder: Encoder,
        conf_encoder: dict,
        decoder: Decoder,
        conf_decoder: dict,
        lr: float = 1e-3,
        b1: float = 0.5,
        b2: float = 0.999,
        w_recon: float = 1.0,
        w_kl: float = 1.0,
        log_dir: str = "",
        batch_size: int = 32,
        n_sample_for_metric: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder(**conf_encoder)
        self.decoder = decoder(**conf_decoder)

        self.gravity_matrix = torch.Tensor(calculate_gravity_matrix()).to(self.device)

        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.w_recon = w_recon
        self.w_kl = w_kl
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.n_sample_for_metric = n_sample_for_metric
        self.conf_encoder = conf_encoder
        self.conf_decoder = conf_decoder

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.gravity_matrix = self.gravity_matrix.to(self.device)
        self.train_test_batch = self.trainer.datamodule.train_dataset[:5]

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        return self.encoder(x)

    def loss_fn(self, x, x_hat, y, mu, logvar):
        g = self.gravity_matrix.repeat(x.shape[0], 1, 1).float()
        x_flat = x.view(x.shape[0], -1).unsqueeze(-1).float()
        recon_loss = F.mse_loss(x_hat, x)  # + F.mse_loss(
        # torch.bmm(g, x_flat).squeeze(), y
        # )
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kl_loss

    def on_validation_epoch_end(self) -> None:
        x, y = self.train_test_batch
        x = torch.Tensor(x).float().to(self.device)
        y = torch.Tensor(y).float().to(self.device)
        l = [None] * len(x)
        x = x.float()
        y = y.float()
        mu, logvar = self.forward(x)

        sample = self.reparametrize(mu, logvar)

        # q = torch.cat((sample, y), dim=1)
        q = sample
        q = torch.randn_like(q)
        x_hat = self.decoder(q).squeeze()

        plot_density_maps(
            x.detach().cpu().numpy(),
            l,
            32,
            log_dir=self.log_dir,
            curr_epoch=self.current_epoch,
            suffix="real",
        )

        plot_density_maps(
            x_hat.detach().cpu().numpy(),
            l,
            32,
            log_dir=self.log_dir,
            curr_epoch=self.current_epoch,
            suffix="gen",
        )

    def inference(self, y):
        z = torch.randn(self.n_sample_for_metric, self.hparams.nz).to(self.device)
        q = torch.cat((z, y.repeat(self.n_sample_for_metric, 1)), dim=1)
        x_hat = self.decoder(q)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        mu, logvar = self.forward(x)

        sample = self.reparametrize(mu, logvar)

        # q = torch.cat((sample, y), dim=1)
        q = sample
        x_hat = self.decoder(q)

        recon_loss, kl_loss = self.loss_fn(x, x_hat, y, mu, logvar)
        loss = self.w_recon * recon_loss + self.w_kl * kl_loss

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/recon_loss", recon_loss, prog_bar=True)
        self.log("train/kl_loss", kl_loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        mu, logvar = self.forward(x)

        sample = self.reparametrize(mu, logvar)

        # q = torch.cat((sample, y), dim=1)
        q = sample
        x_hat = self.decoder(q).squeeze()

        recon_loss, kl_loss = self.loss_fn(x, x_hat, y, mu, logvar)
        loss = self.w_recon * recon_loss + self.w_kl * kl_loss

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/recon_loss", recon_loss, prog_bar=True)
        self.log("val/kl_loss", kl_loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        mu, logvar = self.forward(x)

        sample = self.reparametrize(mu, logvar)

        q = torch.cat((sample, y), dim=1)
        x_hat = self.decoder(q)

        recon_loss, kl_loss = self.loss_fn(x, x_hat, y, mu, logvar)
        loss = self.w_recon * recon_loss + self.w_kl * kl_loss

        self.log("test/loss", loss, prog_bar=True)
        self.log("test/recon_loss", recon_loss, prog_bar=True)
        self.log("test/kl_loss", kl_loss, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        return optimizer
"""


class LitDDPMGrav(pl.LightningModule):
    def __init__(
        self,
        ddpm=UNet_conditional2d,
        conf_ddpm={
            "c_in": 3,
            "c_out": 128,
            "time_dim": 256,
        },
        diffusion=Diffusion2d,
        conf_diffusion: dict = {
            "noise_steps": 1000,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "surf_size": 256,
        },
        ema=EMA2d,
        conf_ema={"beta": 0.995},
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        validation_x_shape: np.ndarray = [5, 51, 1],
        log_dir: str = "",
        batch_size=512,
        n_sample_for_metric: int = 100,
        cfg_scale: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gravity_matrix = torch.Tensor(calculate_gravity_matrix()).to(self.device)

        self.enc_layer = SimpleEncodingGrav()
        self.conf_ddpm = conf_ddpm
        self.model = ddpm(**self.conf_ddpm, encoding_layer=self.enc_layer)
        self.conf_diffusion = conf_diffusion
        self.diffusion = diffusion(**self.conf_diffusion)
        self.conf_ema = conf_ema
        self.ema = ema(**self.conf_ema)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.cfg_scale = cfg_scale

        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.validation_x_shape = validation_x_shape
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.n_sample_for_metric = n_sample_for_metric

        self.inference_model = lambda labels: self.diffusion.sample(
            self.model,
            labels=labels,
            cfg_scale=self.cfg_scale,
        )

        self.current_training_step = 0
        self.current_validation_step = 0

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.gravity_matrix = self.gravity_matrix.to(self.device)
        self.train_test_batch = self.trainer.datamodule.train_dataset[:5]

    def loss_fn(self, recon_x, x):
        return F.mse_loss(recon_x, x)

    def inference(self, conditions):
        with torch.inference_mode():
            return self.diffusion.sample(
                self.model,
                labels=conditions,
                cfg_scale=self.cfg_scale,
            )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.unsqueeze(1)
        x = x.float()
        y = y.float()
        # x = torch.zeros_like(x)
        # y = torch.zeros_like(y)

        t = self.diffusion.sample_timesteps(x.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(x, t)
        if self.cfg_scale > 0 and np.random.random() < 0.1:
            y = None
        predicted_noise = self.model(x_t, t, y)
        loss = F.mse_loss(noise, predicted_noise)

        self.ema.step_ema(self.ema_model, self.model)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.unsqueeze(1)
        x = x.float()
        y = y.float()
        # x = torch.zeros_like(x)
        # y = torch.zeros_like(y)

        t = self.diffusion.sample_timesteps(x.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(x, t)
        if self.cfg_scale > 0 and np.random.random() < 0.1:
            y = None
        predicted_noise = self.model(x_t, t, y)
        loss = F.mse_loss(noise, predicted_noise)
        self.log("val/loss", loss, prog_bar=True)

        self.ema.step_ema(self.ema_model, self.model)

        self.current_validation_step += 1
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        return opt

    def on_validation_epoch_end(self) -> None:
        x, y = self.train_test_batch
        x = torch.Tensor(x).float().to(self.device)
        y = torch.Tensor(y).float().to(self.device)
        l = [None] * len(x)
        x = x.float()
        y = y.float()

        x_hat = self.inference(y)

        plot_density_maps(
            x.squeeze().detach().cpu().numpy(),
            l,
            32,
            log_dir=self.log_dir,
            curr_epoch=self.current_epoch,
            suffix="real",
        )

        plot_density_maps(
            x_hat.squeeze().detach().cpu().numpy(),
            l,
            32,
            log_dir=self.log_dir,
            curr_epoch=self.current_epoch,
            suffix="gen",
        )
