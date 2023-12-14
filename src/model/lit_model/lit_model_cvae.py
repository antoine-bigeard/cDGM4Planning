from typing import Any
import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from src.model.model.vae_geoph import Encoder, Decoder


class LitModelVAE(pl.LightningModule):
    def __init__(
        self,
        encoder: Encoder,
        conf_encoder: dict,
        decoder: Decoder,
        conf_decoder: dict,
        gravity_model,
        lr: float = 1e-3,
        b1: float = 0.5,
        b2: float = 0.999,
        w_recon: float = 1.0,
        w_kl: float = 1.0,
        log_dir: str = "",
        batch_size: int = 32,
        n_sample_for_metrics: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder(**conf_encoder)
        self.decoder = decoder(**conf_decoder)
        self.gravity_model = gravity_model

        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.w_recon = w_recon
        self.w_kl = w_kl
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.n_sample_for_metrics = n_sample_for_metrics
        self.conf_encoder = conf_encoder
        self.conf_decoder = conf_decoder

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.train_test_batch = next(iter(self.train_dataloader()))

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).sqrt()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        return self.encoder(x)

    def loss_fn(self, x, x_hat, mu, logvar):
        recon_loss = F.mse_loss(x_hat, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kl_loss

    def inference(self, y):
        z = torch.randn(self.n_sample_for_metrics, self.hparams.nz).to(self.device)
        q = torch.cat((z, y.repeat(self.n_sample_for_metrics, 1)), dim=1)
        x_hat = self.decoder(q)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, logvar = self.forward(x)

        sample = self.reparametrize(mu, logvar)

        q = torch.cat((sample, y), dim=1)
        x_hat = self.decoder(q)

        recon_loss, kl_loss = self.loss_fn(x, x_hat, mu, logvar)
        loss = self.w_recon * recon_loss + self.w_kl * kl_loss

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/recon_loss", recon_loss, prog_bar=True)
        self.log("train/kl_loss", kl_loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mu, logvar = self.forward(x)

        sample = self.reparametrize(mu, logvar)

        q = torch.cat((sample, y), dim=1)
        x_hat = self.decoder(q)

        recon_loss, kl_loss = self.loss_fn(x, x_hat, mu, logvar)
        loss = self.w_recon * recon_loss + self.w_kl * kl_loss

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/recon_loss", recon_loss, prog_bar=True)
        self.log("val/kl_loss", kl_loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        mu, logvar = self.forward(x)

        sample = self.reparametrize(mu, logvar)

        q = torch.cat((sample, y), dim=1)
        x_hat = self.decoder(q)

        recon_loss, kl_loss = self.loss_fn(x, x_hat, mu, logvar)
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
