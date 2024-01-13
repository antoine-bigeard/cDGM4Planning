import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics import MetricCollection, PearsonCorrCoef
from src.model.model.DCGAN2d import *
from src.model.lit_model.metrics import *
from src.model.lit_model.lit_model_utils import *
from src.model.model.modules_diffusion2d import *
from src.model.model.DDPM2d import *
from src.model.model.vae_geoph import ConvVAE

from src.utils import *
from src.model.lit_model.metrics import compute_cond_dist

import copy
import json


class LitModel2d(pl.LightningModule):
    def __init__(
        self,
        n_samples_hist: int = 4,
        metrics: list = ["L2", "L1", "dist_cond"],
    ) -> None:
        super().__init__()
        self.n_samples_hist = n_samples_hist
        self.metrics = metrics

    def on_fit_start(self) -> None:
        val_fig_batch = [
            self.trainer.datamodule.val_dataset[i]
            for i in range(self.validation_x_shape[0])
        ]
        self.validation_x, self.validation_y = (
            torch.cat(
                [
                    torch.tensor(val_fig_batch[i][0]).unsqueeze(0)
                    for i in range(len(val_fig_batch))
                ]
            ),
            torch.cat(
                [
                    torch.tensor(val_fig_batch[i][1]).unsqueeze(0)
                    for i in range(len(val_fig_batch))
                ]
            ),
        )
        self.y_1_idxs = get_idx_val_2D(self.validation_y)

    def on_test_start(self):
        self.dict_metrics_paths = defaultdict(dict)
        self.metrics_measures = defaultdict(list)

    def predict_step(self, batch, batch_idx: int):
        pass

    def test_model(self, datamodule, n_obs: list, path_output: str):
        with torch.inference_mode():
            x = torch.stack(
                [torch.Tensor(datamodule.test_dataset[i][0]) for i in range(len(n_obs))]
            ).to(self.device)

            labels = torch.cat(
                [
                    random_observation_ore_maps(
                        x[i].unsqueeze(0), lambda: n_obs[i], seed=True
                    )
                    for i in range(len(n_obs))
                ],
                dim=0,
            ).to(self.device)

            metrics_measures, metrics_samples = measure_metrics(
                self.inference_model,
                x,
                labels,
                self.n_sample_for_metric,
                self.metrics,
                no_batch=True,
            )
            figs, paths = create_figs_best_metrics_2D(
                metrics_samples,
                get_idx_val_2D(labels),
                x,
                os.path.join(path_output),
                save=True,
                sequential_cond=self.sequential_cond,
            )

            out_file = os.path.join(os.path.join(path_output), "metrics.json")
            for k, v in metrics_measures.items():
                metrics_measures[k] = [float(m) for m in v]
            with open(out_file, "w") as f:
                json.dump(metrics_measures, f)
            return x, get_idx_val_2D(labels), metrics_samples, metrics_measures

    def on_test_end(self) -> None:
        img_dir = self.log_dir
        new_dict_metrics = defaultdict(dict)
        new_dict_metrics["paths"] = defaultdict(dict)
        path_dict = os.path.join(img_dir, "metrics_img_path.json")
        for k, v in self.metrics_measures.items():
            fig_hist = plt.figure()
            plt.hist(
                v,
                bins=100,
                density=True,
                histtype="step",
                cumulative=True,
                label=f"cum_distrib_{k}",
            )
            plt.savefig(os.path.join(img_dir, f"cum_distrib_{k}"))
            fig_cum = plt.figure()
            plt.hist(
                v,
                bins=250,
                density=True,
                histtype="step",
                cumulative=False,
                label=f"histogram_{k}",
            )
            plt.savefig(os.path.join(img_dir, f"histogram_{k}"))
            vals_to_keep = keep_samples(torch.Tensor(v), n=4)
            for p, m in self.dict_metrics_paths[k].items():
                if m in vals_to_keep:
                    new_dict_metrics["paths"][k][p] = float(m)
                # else:
                #     os.remove(k)

            new_dict_metrics[f"{k}_mean"] = float(np.mean(v))
            new_dict_metrics[f"{k}_measures"] = list(map(lambda x: float(x), list(v)))
            # new_dict_metrics["samples_per_sec"] = np.mean()
            self.logger.experiment.add_figure(f"histogram_{k}", fig_hist)
            self.logger.experiment.add_figure(f"cumulative_distribution_{k}", fig_cum)
        with open(path_dict, "w") as f:
            json.dump(new_dict_metrics, f)

    def test_step(self, batch, batch_idx):
        img_dir = os.path.join(self.log_dir, "images", f"test_step_{batch_idx}")
        os.makedirs(img_dir, exist_ok=True)
        x, y = batch
        y_1_idxs = get_idx_val_2D(y)
        metrics_measures, metrics_samples = measure_metrics(
            self.inference_model,
            x,
            y,
            self.n_sample_for_metric,
            self.metrics,
            no_batch=False,
        )
        figs, paths = create_figs_best_metrics_2D(
            metrics_samples,
            y_1_idxs,
            x,
            img_dir,
            save=True,
            sequential_cond=self.sequential_cond,
        )
        for k, v in metrics_measures.items():
            self.metrics_measures[k] += v
        for k, ps in paths.items():
            if k not in ["ground_truth", "time_inference"]:
                for i, p in enumerate(ps):
                    self.dict_metrics_paths[k + "_min"][p] = self.metrics_measures[
                        k + "_min"
                    ][i]


class LitDCGAN2d(LitModel2d):
    def __init__(
        self,
        generator: nn.Module = LargeGeneratorInject2d,
        discriminator: nn.Module = LargerDiscriminator2d,
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
        sequential_cond: bool = False,
        latent_1d: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # instantiate models
        self.sequential_cond = sequential_cond
        self.encoding_layer_gen = None
        self.encoding_layer_dis = None
        if self.sequential_cond:
            self.encoding_layer_gen = ConditionEncoding(
                input_size=128, hidden_size=128, num_layers=1, batch_first=True
            ).cuda()
            self.encoding_layer_dis = ConditionEncoding(
                input_size=128, hidden_size=128, num_layers=1, batch_first=True
            ).cuda()
            # self.encoding_layer = SequentialCat()

        self.conf_generator = conf_generator
        self.conf_discriminator = conf_discriminator
        self.generator = generator(
            **self.conf_generator,
            encoding_layer=self.encoding_layer_gen,
            latent_1d=latent_1d,
        )
        self.discriminator = discriminator(
            **self.conf_discriminator, encoding_layer=self.encoding_layer_dis
        )

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
        self.latent_1d = latent_1d

        self.inference_model = lambda labels: self.generator.inference(
            labels, self.latent_dim
        )

        self.validation_z = (
            torch.randn(
                self.validation_x_shape[0],
                1,
                self.latent_dim,
                device=self.device,
            )
            if self.latent_1d
            else torch.randn(
                self.validation_x_shape[0],
                1,
                self.latent_dim,
                self.latent_dim,
                device=self.device,
            )
        )
        self.val_z_best_metrics = (
            torch.randn(self.n_sample_for_metric, 1, self.latent_dim).to(self.device)
            if self.latent_1d
            else torch.randn(
                self.n_sample_for_metric, 1, self.latent_dim, self.latent_dim
            ).to(self.device)
        )

        self.current_training_step = 0
        self.current_validation_step = 0

    def forward(self, z, y):
        return self.generator(z.cuda(), y.cuda())

    def g_criterion(self, pred, target):
        # return F.mse_loss(F.sigmoid(pred), target)
        return F.binary_cross_entropy(F.sigmoid(pred), target)

    def d_criterion(self, pred, target):
        # return F.mse_loss(F.sigmoid(pred), target)
        return F.binary_cross_entropy(F.sigmoid(pred), target)

    def generator_step(self, y, x):
        z = (
            torch.randn(x.shape[0], 1, self.latent_dim).to(self.device)
            if self.latent_1d
            else torch.randn(x.shape[0], 1, self.latent_dim, self.latent_dim).to(
                self.device
            )
        )
        if self.use_rd_y:
            y = random_observation_ore_maps(x).to(self.device)
        y_1_idxs = get_idx_val_2D(y)

        generated_surfaces = self(z, y)

        d_output = torch.squeeze(self.discriminator(generated_surfaces, y.cuda()))

        if self.w_gp_loss in ["wgp", "w"]:
            g_loss = -torch.mean(d_output)
        else:
            g_loss = self.g_criterion(d_output, torch.ones(x.shape[0]).to(self.device))

        return {"g_loss": g_loss, "preds": generated_surfaces, "condition": y}

    def discriminator_step(self, x, y):
        # Loss for the real samples
        d_output_real = torch.squeeze(self.discriminator(x.cuda(), y.cuda()))

        # Loss for the generated samples
        z = (
            torch.randn(x.shape[0], 1, self.latent_dim).to(self.device)
            if self.latent_1d
            else torch.randn(x.shape[0], 1, self.latent_dim, self.latent_dim).to(
                self.device
            )
        )
        if self.use_rd_y:
            y = random_observation_ore_maps(x).to(self.device)
        y_1_idxs = get_idx_val_2D(y)
        generated_surfaces = self(z, y)
        d_output_fake = torch.squeeze(self.discriminator(generated_surfaces, y.cuda()))

        if self.w_gp_loss == "wgp":
            gradient_penalty = calculate_gradient_penalty(
                self.discriminator, x.data, generated_surfaces.data, y, self.device
            )

            d_loss = (
                -torch.mean(d_output_real)
                + torch.mean(d_output_fake)
                + 10 * gradient_penalty
            )

            return {"d_loss": d_loss, "gradient_penalty": gradient_penalty}
        elif self.w_gp_loss == "w":
            d_loss = torch.mean(-d_output_real + d_output_fake)
            return {"d_loss": d_loss}

        else:
            loss_real = self.d_criterion(
                d_output_real, torch.ones(x.size(0)).to(self.device)
            )
            loss_fake = self.d_criterion(
                d_output_fake, torch.zeros(x.size(0)).to(self.device)
            )
            d_loss = (loss_real + loss_fake) / 2
            return {"d_loss": d_loss}

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        # train generator
        if optimizer_idx == 0:
            gen_dict = self.generator_step(y, x)
            self.log("train/g_loss", gen_dict["g_loss"], prog_bar=True)

            self.current_training_step += 1
            return gen_dict["g_loss"]

        # train discriminator
        if optimizer_idx == 1:
            dis_dict = self.discriminator_step(x, y)
            self.log("train/d_loss", dis_dict["d_loss"], prog_bar=True)
            if self.w_gp_loss == "wgp":
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
        gen_dict = self.generator_step(y, x)
        self.log("val/g_loss", gen_dict["g_loss"], prog_bar=True)

        # train discriminator
        dis_dict = self.discriminator_step(x, y)
        self.log("val/d_loss", dis_dict["d_loss"], prog_bar=True)
        if self.w_gp_loss == "wgp":
            self.log(
                "val/gradient_penalty",
                dis_dict["gradient_penalty"],
                prog_bar=True,
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

        figs = create_figs_2D(
            sample_surfaces,
            self.y_1_idxs,
            img_dir=img_dir,
            save=True,
            sequential_cond=self.sequential_cond,
        )
        for i, fig in enumerate(figs):
            self.logger.experiment.add_figure(
                f"generated_image_{i}", fig, self.current_epoch
            )

    def on_validation_epoch_start(self) -> None:
        self.current_validation_step = 0
        return super().on_validation_epoch_start()

    def on_train_start(self) -> None:
        os.makedirs(os.path.join(self.log_dir, "ckpts"), exist_ok=True)

    def inference(self, conditions):
        with torch.inference_mode():
            return self.generator.inference(conditions, latent_dim=self.latent_dim)


class LitDDPM2d(LitModel2d):
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
        use_rd_y: bool = True,
        batch_size=512,
        n_sample_for_metric: int = 100,
        latent_dim=20,
        cfg_scale: int = 3,
        sequential_cond: bool = False,
        encoding_layer=ConditionEncoding,
        conf_encoding_layer={},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.sequential_cond = sequential_cond
        self.encoding_layer = None
        if self.sequential_cond:
            self.conf_encoding_layer = conf_encoding_layer
            self.encoding_layer = encoding_layer(**conf_encoding_layer).cuda()
            self.encoding_layer = ConditionEncoding(
                input_size=128, hidden_size=128, num_layers=1, batch_first=True
            ).cuda()

        # instantiate models
        self.conf_ddpm = conf_ddpm
        self.model = ddpm(**self.conf_ddpm, encoding_layer=self.encoding_layer)
        self.conf_diffusion = conf_diffusion
        self.diffusion = diffusion(**self.conf_diffusion)
        self.conf_ema = conf_ema
        self.ema = ema(**self.conf_ema)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.cfg_scale = cfg_scale

        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.validation_x_shape = validation_x_shape
        self.log_dir = log_dir
        self.use_rd_y = use_rd_y
        self.batch_size = batch_size
        self.n_sample_for_metric = n_sample_for_metric

        self.inference_model = lambda labels: self.diffusion.sample(
            self.model,
            labels=labels,
            cfg_scale=self.cfg_scale,
        )

        self.validation_z = torch.randn(
            self.validation_x_shape[0], self.latent_dim, device=self.device
        )

        self.current_training_step = 0
        self.current_validation_step = 0

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
        if self.use_rd_y:
            y = random_observation_ore_maps(x).to(self.device)

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
        if self.use_rd_y:
            y = random_observation_ore_maps(x).to(self.device)

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

    def on_train_epoch_start(self) -> None:
        img_dir = os.path.join(self.log_dir, f"epoch_{self.current_epoch}")
        os.makedirs(img_dir, exist_ok=True)
        # log sampled images
        sample_surfaces = []
        for s in range(self.validation_x.size(0)):
            sample_surfaces.append(
                self.diffusion.sample(
                    self.model,
                    labels=torch.cat(
                        [self.validation_y[s].unsqueeze(0)] * self.validation_x.size(0),
                        0,
                    ).cuda(),
                    cfg_scale=self.cfg_scale,
                )
            )

        figs = create_figs_2D(
            sample_surfaces,
            self.y_1_idxs,
            img_dir,
            save=True,
            sequential_cond=self.sequential_cond,
        )
        for i, fig in enumerate(figs):
            self.logger.experiment.add_figure(
                f"generated_image_{i}", fig, self.current_epoch
            )


class LitModelVAEOre(LitModel2d):
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
        BCE = F.mse_loss(recon_x.view(-1, 1024), x.view(-1, 1024), reduction="sum")
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
