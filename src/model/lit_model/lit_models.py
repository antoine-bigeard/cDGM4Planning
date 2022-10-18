import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics import MetricCollection, PearsonCorrCoef
from src.model.model.DCGAN import *
from src.model.lit_model.metrics import *

from src.utils import *

import copy


class LitDCGAN(pl.LightningModule):
    def __init__(
        self,
        generator: nn.Module = SmallGenerator,
        discriminator: nn.Module = SmallDiscriminator,
        conf_generator: dict = {"latent_dim": 100},
        conf_discriminator: dict = {},
        g_lr: float = 0.0002,
        d_lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        validation_x_shape: np.ndarray = [5, 51, 1],
        log_dir: str = "",
        use_rd_y: bool = True,
        batch_size=512,
        wasserstein_gp_loss=False,
        n_sample_for_metric: int = 100,
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
        self.use_rd_y = use_rd_y
        self.batch_size = batch_size
        self.w_gp_loss = wasserstein_gp_loss
        self.n_sample_for_metric = n_sample_for_metric

        self.validation_z = torch.randn(
            self.validation_x_shape[0], self.latent_dim, device=self.device
        )
        self.val_z_best_metrics = torch.randn(
            self.n_sample_for_metric, self.latent_dim
        ).to(self.device)

        rd_obs = random_observation(
            self.validation_x_shape, return_1_idx=True, random=False
        )
        self.validation_y, self.y_1_idxs = rd_obs[0].to(self.device), rd_obs[1]

        self.current_training_step = 0
        self.current_validation_step = 0

    def forward(self, z, y):
        return self.generator(z.cuda(), y.cuda())

    def g_criterion(self, pred, target):
        return F.mse_loss(F.sigmoid(pred), target)

    def d_criterion(self, pred, target):
        return F.mse_loss(F.sigmoid(pred), target)

    def generator_step(self, y, x_shape):
        z = torch.randn(x_shape[0], self.latent_dim).to(self.device)
        if self.use_rd_y:
            y, idxs_1 = random_observation(x_shape, return_1_idx=True)
            y = y.to(self.device)

        generated_surface = self(z, y)

        d_output = torch.squeeze(self.discriminator(generated_surface, y.cuda()))

        if self.w_gp_loss:
            g_loss = -torch.mean(d_output)
        else:
            g_loss = self.g_criterion(
                d_output, torch.ones(x_shape[0]).to(self.device)
            )  # + F.mse_loss(generated_surface[:, :, idxs_1], y[:, 1, idxs_1])
            # g_loss = d_output.mean()

        return {"g_loss": g_loss, "preds": generated_surface, "condition": y}

    def discriminator_step(self, x, y):
        # Loss for the real samples
        d_output_real = torch.squeeze(self.discriminator(x.cuda(), y.cuda()))
        # loss_real = d_output.mean()

        # Loss for the generated samples
        z = torch.randn(x.size(0), self.latent_dim).to(self.device)
        if self.use_rd_y:
            y = random_observation(x.shape).to(self.device)
        generated_surfaces = self(z, y)
        d_output_fake = torch.squeeze(self.discriminator(generated_surfaces, y.cuda()))

        if self.w_gp_loss:
            gradient_penalty = calculate_gradient_penalty(
                self.discriminator, x.data, generated_surfaces.data, y, self.device
            )

            d_loss = (
                -torch.mean(d_output_real)
                + torch.mean(d_output_fake)
                + 10 * gradient_penalty
            )

        else:
            loss_real = self.d_criterion(
                d_output_real, torch.ones(x.size(0)).to(self.device)
            )
            loss_fake = self.d_criterion(
                d_output_fake, torch.zeros(x.size(0)).to(self.device)
            )
            d_loss = loss_real + loss_fake
        # loss_fake = d_output.mean()

        return (
            {"d_loss": d_loss, "gradient_penalty": gradient_penalty}
            if self.w_gp_loss
            else {"d_loss": d_loss}
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        # train generator
        if optimizer_idx == 0:
            # if self.current_training_step % 2 > 0 or self.current_training_step == 0:
            gen_dict = self.generator_step(
                y, x_shape=(y.size(0), *self.validation_x_shape[1:])
            )
            self.log("train/g_loss", gen_dict["g_loss"], prog_bar=True)

            self.current_training_step += 1
            return gen_dict["g_loss"]

        # train discriminator
        if optimizer_idx == 1:
            # if self.current_training_step % 2 == 0 and self.current_training_step > 0:
            dis_dict = self.discriminator_step(x, y)
            self.log("train/d_loss", dis_dict["d_loss"], prog_bar=True)
            if self.w_gp_loss:
                self.log(
                    "train/gradient_penalty",
                    dis_dict["gradient_penalty"],
                    prog_bar=True,
                )
            self.current_training_step += 1
            return dis_dict["d_loss"]

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # train generator
        gen_dict = self.generator_step(
            y, x_shape=(y.size(0), *self.validation_x_shape[1:])
        )
        self.log("val/g_loss", gen_dict["g_loss"], prog_bar=True)

        # train discriminator
        dis_dict = self.discriminator_step(x, y)
        self.log("val/d_loss", dis_dict["d_loss"], prog_bar=True)
        if self.w_gp_loss:
            self.log(
                "val/gradient_penalty",
                dis_dict["gradient_penalty"],
                prog_bar=True,
            )
        if self.current_validation_step % 10 == 0:
            best_y, best_L2, best_sample_y, best_sample_L2 = self.metrics(
                x[:10], self.validation_y
            )
            for k in best_y.keys():
                self.log(f"val/dist_y/{k}", best_y[k])
                self.log(f"val/L2/{k}", best_L2[k])
            if self.current_validation_step == 0:
                img_dir = os.path.join(self.log_dir, f"val_epoch_{self.current_epoch}")
                figs = create_figs_best_metrics(
                    [(best_sample_L2, "best_L2"), (best_sample_y, "best_y")],
                    self.y_1_idxs,
                    x[0],
                    img_dir,
                    save=True,
                )
                for i, fig in enumerate(figs):
                    self.logger.experiment.add_figure(
                        f"best_metrics_{i}", fig, self.current_epoch
                    )

        self.current_validation_step += 1

        return {"g_loss": gen_dict["g_loss"], "d_loss": dis_dict["d_loss"]}

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.g_lr, betas=(self.b1, self.b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.d_lr, betas=(self.b1, self.b2)
        )
        return [opt_g, opt_d], []

    def on_train_epoch_start(self) -> None:
        img_dir = os.path.join(self.log_dir, f"epoch_{self.current_epoch}")
        os.makedirs(img_dir, exist_ok=True)
        # log sampled images
        sample_surfaces = []
        for s in range(self.validation_z.size(0)):
            sample_surfaces.append(
                self(
                    self.validation_z,
                    torch.cat(
                        [self.validation_y[s].unsqueeze(0)] * self.validation_z.size(0),
                        0,
                    ),
                )
            )

        figs = create_figs(
            sample_surfaces,
            self.y_1_idxs,
            self.validation_y,
            img_dir=img_dir,
            save=True,
        )
        for i, fig in enumerate(figs):
            self.logger.experiment.add_figure(
                f"generated_image_{i}", fig, self.current_epoch
            )

    def on_validation_epoch_end(self):

        # log sampled images
        sample_surfaces = []
        for s in range(self.validation_z.size(0)):
            sample_surfaces.append(
                self(
                    self.validation_z,
                    torch.cat(
                        [self.validation_y[s].unsqueeze(0)] * self.validation_z.size(0),
                        0,
                    ),
                )
            )

        figs = create_figs(sample_surfaces, self.y_1_idxs, self.validation_y)
        for i, fig in enumerate(figs):
            self.logger.experiment.add_figure(
                f"generated_image_{i}", fig, self.current_epoch
            )

    def metrics(self, x, y):
        best_dist_y = {}
        best_L2 = {}
        best_samp_y = {}
        best_samp_L2 = {}
        for label in y:
            new_vals_metric_y = []
            new_vals_metric_L2 = []
            for i, surf in enumerate(x):
                idx, new_y, new_L2, new_samp_y, new_samp_L2 = compute_best_pred(
                    self, self.val_z_best_metrics, surf, label
                )
                new_vals_metric_y.append(new_y)
                new_vals_metric_L2.append(new_L2)
                if i == 0:
                    best_samp_y[idx] = new_samp_y
                    best_samp_L2[idx] = new_samp_L2
            best_dist_y[idx] = sum(new_vals_metric_y) / len(x)
            best_L2[idx] = sum(new_vals_metric_L2) / len(x)
        return best_dist_y, best_L2, best_samp_y, best_samp_L2

    def on_validation_epoch_start(self) -> None:
        self.current_validation_step = 0
        return super().on_validation_epoch_start()


class LitVAE(pl.LightningModule):
    def __init__(
        self,
        vae: nn.Module = SmallGenerator,
        conf_vae: dict = {"latent_dim": 100},
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        validation_x_shape: np.ndarray = [5, 51, 1],
        log_dir: str = "",
        use_rd_y: bool = True,
        batch_size=512,
        n_sample_for_metric: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        # instantiate models
        self.conf_vae = conf_vae
        self.vae = vae(**self.conf_vae)

        self.latent_dim = self.conf_vae["latent_dim"]
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.validation_x_shape = validation_x_shape
        self.log_dir = log_dir
        self.use_rd_y = use_rd_y
        self.batch_size = batch_size
        self.n_sample_for_metric = n_sample_for_metric

        self.validation_z = torch.randn(
            self.validation_x_shape[0], self.latent_dim, device=self.device
        )
        self.val_z_best_metrics = torch.randn(
            self.n_sample_for_metric, self.latent_dim
        ).to(self.device)
        rd_obs = random_observation(
            self.validation_x_shape, return_1_idx=True, random=False
        )
        self.validation_y, self.y_1_idxs = rd_obs[0].to(self.device), rd_obs[1]

        self.current_training_step = 0
        self.current_validation_step = 0

    def forward(self, z, y):
        return self.vae(z.cuda(), y.cuda())

    def loss_fn(self, recon_x, x):
        # BCE = F.binary_cross_entropy(recon_x.squeeze(), x.squeeze(), reduction="sum")
        # KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # return (BCE + KLD) / x.size(0)

        # return L2_metric(recon_x, x, p=2).mean().squeeze()
        return F.mse_loss(recon_x, x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_shape = (y.size(0), *self.validation_x_shape[1:])
        if self.use_rd_y:
            y, idxs_1 = random_observation(x_shape, return_1_idx=True)
            y = y.to(self.device)

        recon_x, z = self(x, y)

        loss = self.loss_fn(recon_x, x)

        self.log("train/loss", loss, prog_bar=True)
        self.current_training_step += 1

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_shape = (y.size(0), *self.validation_x_shape[1:])
        if self.use_rd_y:
            y, idxs_1 = random_observation(x_shape, return_1_idx=True)
            y = y.to(self.device)

        recon_x, z = self(x, y)

        loss = self.loss_fn(recon_x, x)

        self.log("val/loss", loss, prog_bar=True)

        if self.current_validation_step % 10 == 0:
            best_y, best_L2, best_sample_y, best_sample_L2 = self.metrics(
                x[:10], self.validation_y
            )
            for k in best_y.keys():
                self.log(f"val/dist_y/{k}", best_y[k])
                self.log(f"val/L2/{k}", best_L2[k])
            if self.current_validation_step == 0:
                img_dir = os.path.join(self.log_dir, f"val_epoch_{self.current_epoch}")
                figs = create_figs_best_metrics(
                    [(best_sample_L2, "best_L2"), (best_sample_y, "best_y")],
                    self.y_1_idxs,
                    x[0],
                    img_dir,
                    save=True,
                )
                for i, fig in enumerate(figs):
                    self.logger.experiment.add_figure(
                        f"best_metrics_{i}", fig, self.current_epoch
                    )

        self.current_validation_step += 1

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.vae.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        return opt

    def on_validation_epoch_start(self) -> None:
        self.current_validation_step = 0
        return super().on_validation_epoch_start()

    def on_train_epoch_start(self) -> None:
        img_dir = os.path.join(self.log_dir, f"epoch_{self.current_epoch}")
        os.makedirs(img_dir, exist_ok=True)
        # log sampled images
        sample_surfaces = []
        for s in range(self.validation_z.size(0)):
            sample_surfaces.append(
                self.vae.inference(
                    self.validation_z,
                    torch.cat(
                        [self.validation_y[s].unsqueeze(0)] * self.validation_z.size(0),
                        0,
                    ),
                )
            )

        figs = create_figs(
            sample_surfaces, self.y_1_idxs, self.validation_y, img_dir, save=True
        )
        for i, fig in enumerate(figs):
            self.logger.experiment.add_figure(
                f"generated_image_{i}", fig, self.current_epoch
            )

    def on_validation_epoch_end(self):

        # log sampled images
        sample_surfaces = []
        for s in range(self.validation_z.size(0)):
            sample_surfaces.append(
                self.vae.inference(
                    self.validation_z,
                    torch.cat(
                        [self.validation_y[s].unsqueeze(0)] * self.validation_z.size(0),
                        0,
                    ),
                )
            )

        figs = create_figs(sample_surfaces, self.y_1_idxs, self.validation_y)
        for i, fig in enumerate(figs):
            self.logger.experiment.add_figure(
                f"generated_image_{i}", fig, self.current_epoch
            )

    def metrics(self, x, y):
        best_dist_y = {}
        best_L2 = {}
        best_samp_y = {}
        best_samp_L2 = {}
        for label in y:
            new_vals_metric_y = []
            new_vals_metric_L2 = []
            for i, surf in enumerate(x):
                idx, new_y, new_L2, new_samp_y, new_samp_L2 = compute_best_pred(
                    self.vae.inference, self.val_z_best_metrics, surf, label
                )
                new_vals_metric_y.append(new_y)
                new_vals_metric_L2.append(new_L2)
                if i == 0:
                    best_samp_y[idx] = new_samp_y
                    best_samp_L2[idx] = new_samp_L2
            best_dist_y[idx] = sum(new_vals_metric_y) / len(x)
            best_L2[idx] = sum(new_vals_metric_L2) / len(x)
        return best_dist_y, best_L2, best_samp_y, best_samp_L2


class LitDDPM(pl.LightningModule):
    def __init__(
        self,
        ema,
        conf_ema,
        diffusion,
        conf_diffusion: dict = {},
        ddpm: nn.Module = SmallGenerator,
        conf_ddpm: dict = {},
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        validation_x_shape: np.ndarray = [5, 51, 1],
        log_dir: str = "",
        use_rd_y: bool = True,
        batch_size=512,
        n_sample_for_metric: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        # instantiate models
        self.conf_ema = conf_ema
        self.ema = self.ema(**self.conf_ema)
        self.conf_diffusion = conf_diffusion
        self.diffusion = diffusion(**self.conf_diffusion)
        self.conf_ddpm = conf_ddpm
        self.ddpm = ddpm(**self.conf_ddpm)
        self.ema_model = copy.deepcopy(self.ddpm).eval().requires_grad_(False)

        self.latent_dim = self.conf_ddpm["latent_dim"]
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.validation_x_shape = validation_x_shape
        self.log_dir = log_dir
        self.use_rd_y = use_rd_y
        self.batch_size = batch_size
        self.n_sample_for_metric = n_sample_for_metric

        self.validation_z = torch.randn(
            self.validation_x_shape[0], self.latent_dim, device=self.device
        )
        self.val_z_best_metrics = torch.randn(
            self.n_sample_for_metric, self.latent_dim
        ).to(self.device)
        rd_obs = random_observation(
            self.validation_x_shape, return_1_idx=True, random=False
        )
        self.validation_y, self.y_1_idxs = rd_obs[0].to(self.device), rd_obs[1]

        self.current_training_step = 0
        self.current_validation_step = 0

    def forward(self, z, y):
        return self.vae(z.cuda(), y.cuda())

    def loss_fn(self, recon_x, x):
        return F.mse_loss(recon_x, x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.use_rd_y:
            y, idxs_1 = random_observation(x_shape, return_1_idx=True)
            y = y.to(self.device)
        t = self.diffusion.sample_timesteps(x.size(0)).to(self.device)
        x_t, noise = self.diffusion.noise_images(x, t)
        if np.random.random() < 0.1:
            y = None
        predicted_noise = self.ddpm(x_t, t, y)

        loss = F.mse_loss(noise, predicted_noise)

        self.log("train/loss", loss, prog_bar=True)
        self.current_training_step += 1

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_shape = (y.size(0), *self.validation_x_shape[1:])
        if self.use_rd_y:
            y, idxs_1 = random_observation(x_shape, return_1_idx=True)
            y = y.to(self.device)

        samples_x = self.diffusion.sample(self.ddpm, n=y.size(0), labels=y)
        ema_sampled_x = self.diffusion.sample(self.ema_model, n=y.size(0), labels=y)

        # if self.current_validation_step % 10 == 0:
        #     best_y, best_L2, best_sample_y, best_sample_L2 = self.metrics(
        #         x[:10], self.validation_y
        #     )
        #     for k in best_y.keys():
        #         self.log(f"val/dist_y/{k}", best_y[k])
        #         self.log(f"val/L2/{k}", best_L2[k])
        #     if self.current_validation_step == 0:
        #         img_dir = os.path.join(self.log_dir, f"val_epoch_{self.current_epoch}")
        #         figs = create_figs_best_metrics(
        #             [(best_sample_L2, "best_L2"), (best_sample_y, "best_y")],
        #             self.y_1_idxs,
        #             x[0],
        #             img_dir,
        #             save=True,
        #         )
        #         for i, fig in enumerate(figs):
        #             self.logger.experiment.add_figure(
        #                 f"best_metrics_{i}", fig, self.current_epoch
        #             )

        self.current_validation_step += 1

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.vae.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        return opt

    # def on_validation_epoch_start(self) -> None:
    #     self.current_validation_step = 0
    #     return super().on_validation_epoch_start()

    # def on_train_epoch_start(self) -> None:
    #     img_dir = os.path.join(self.log_dir, f"epoch_{self.current_epoch}")
    #     os.makedirs(img_dir, exist_ok=True)
    #     # log sampled images
    #     sample_surfaces = []
    #     for s in range(self.validation_z.size(0)):
    #         sample_surfaces.append(
    #             self.vae.inference(
    #                 self.validation_z,
    #                 torch.cat(
    #                     [self.validation_y[s].unsqueeze(0)] * self.validation_z.size(0),
    #                     0,
    #                 ),
    #             )
    #         )

    #     figs = create_figs(
    #         sample_surfaces, self.y_1_idxs, self.validation_y, img_dir, save=True
    #     )
    #     for i, fig in enumerate(figs):
    #         self.logger.experiment.add_figure(
    #             f"generated_image_{i}", fig, self.current_epoch
    #         )

    # def on_validation_epoch_end(self):

    #     # log sampled images
    #     sample_surfaces = []
    #     for s in range(self.validation_z.size(0)):
    #         sample_surfaces.append(
    #             self.vae.inference(
    #                 self.validation_z,
    #                 torch.cat(
    #                     [self.validation_y[s].unsqueeze(0)] * self.validation_z.size(0),
    #                     0,
    #                 ),
    #             )
    #         )

    #     figs = create_figs(sample_surfaces, self.y_1_idxs, self.validation_y)
    #     for i, fig in enumerate(figs):
    #         self.logger.experiment.add_figure(
    #             f"generated_image_{i}", fig, self.current_epoch
    #         )

    def metrics(self, x, y):
        best_dist_y = {}
        best_L2 = {}
        best_samp_y = {}
        best_samp_L2 = {}
        for label in y:
            new_vals_metric_y = []
            new_vals_metric_L2 = []
            for i, surf in enumerate(x):
                idx, new_y, new_L2, new_samp_y, new_samp_L2 = compute_best_pred(
                    self.vae.inference, self.val_z_best_metrics, surf, label
                )
                new_vals_metric_y.append(new_y)
                new_vals_metric_L2.append(new_L2)
                if i == 0:
                    best_samp_y[idx] = new_samp_y
                    best_samp_L2[idx] = new_samp_L2
            best_dist_y[idx] = sum(new_vals_metric_y) / len(x)
            best_L2[idx] = sum(new_vals_metric_L2) / len(x)
        return best_dist_y, best_L2, best_samp_y, best_samp_L2
