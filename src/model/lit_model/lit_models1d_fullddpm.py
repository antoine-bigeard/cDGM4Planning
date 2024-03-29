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
from src.model.model.transformer import TransformerAlone

from src.utils import *
from src.model.lit_model.metrics import compute_cond_dist

import copy
import json


class LitModel1d(pl.LightningModule):
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
            self.trainer.datamodule.val_dataset[i + 5]
            for i in range(self.validation_x_shape[0])
        ]
        self.validation_x = torch.stack([x[0][0] for x in val_fig_batch]).to(
            self.device
        ), torch.stack([x[0][1] for x in val_fig_batch]).to(self.device)
        self.validation_y = torch.stack([x[1][0] for x in val_fig_batch]).to(
            self.device
        ), torch.stack([x[1][1] for x in val_fig_batch]).to(self.device)

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


class LitDDPM1dFull(LitModel1d):
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
        cuda_device_idx: int = 1,
    ):
        super().__init__()
        self.cuda_device_idx = cuda_device_idx
        self.save_hyperparameters()
        self.sequential_cond = sequential_cond
        self.encoding_layer = None
        if self.sequential_cond:
            self.conf_encoding_layer = conf_encoding_layer
            self.encoding_layer = encoding_layer(**conf_encoding_layer).cuda(
                self.cuda_device_idx
            )
            # self.encoding_layer = ConditionEncoding(
            #     input_size=128, hidden_size=128, num_layers=1, batch_first=True
            # ).cuda()

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

        self.mse_loss = nn.MSELoss(reduction="none")

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
        (surfaces, surfaces_padding_mask), (
            observations,
            observations_padding_mask,
        ) = batch
        if self.use_rd_y:
            y = random_observation_ore_maps(surfaces).to(self.device)

        t = self.diffusion.sample_timesteps(surfaces.shape[0]).to(self.device)
        surfaces_t, noise = self.diffusion.noise_images(surfaces, t)
        if self.cfg_scale > 0 and np.random.random() < 0.1:
            y = None
        predicted_noise = self.model(surfaces_t, t, observations)

        noise = noise.reshape(
            noise.size(0), noise.size(1), noise.size(2) // 32, 32
        ).swapaxes(1, 2)
        predicted_noise = predicted_noise.reshape(
            predicted_noise.size(0),
            predicted_noise.size(1),
            predicted_noise.size(2) // 32,
            32,
        ).swapaxes(1, 2)

        loss = self.mse_loss(noise, predicted_noise)
        loss = (
            loss * (1 - surfaces_padding_mask).unsqueeze(-1).unsqueeze(-1)
        ).sum() / (1 - surfaces_padding_mask).sum()
        loss = F.mse_loss(noise, predicted_noise)

        self.ema.step_ema(self.ema_model, self.model)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (surfaces, surfaces_padding_mask), (
            observations,
            observations_padding_mask,
        ) = batch
        if self.use_rd_y:
            y = random_observation_ore_maps(surfaces).to(self.device)

        t = self.diffusion.sample_timesteps(surfaces.shape[0]).to(self.device)
        surfaces_t, noise = self.diffusion.noise_images(surfaces, t)
        if self.cfg_scale > 0 and np.random.random() < 0.1:
            y = None
        predicted_noise = self.model(surfaces_t, t, observations)

        noise = noise.reshape(
            noise.size(0), noise.size(1), noise.size(2) // 32, 32
        ).swapaxes(1, 2)
        predicted_noise = predicted_noise.reshape(
            predicted_noise.size(0),
            predicted_noise.size(1),
            predicted_noise.size(2) // 32,
            32,
        ).swapaxes(1, 2)

        loss = self.mse_loss(noise, predicted_noise)
        loss = (
            loss * (1 - surfaces_padding_mask).unsqueeze(-1).unsqueeze(-1)
        ).sum() / (1 - surfaces_padding_mask).sum()
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
        for s in range(self.validation_x[0].size(0)):
            sample_surfaces.append(
                self.diffusion.sample(
                    self.model,
                    observations=torch.cat(
                        [self.validation_y[0][s].unsqueeze(0)]
                        * self.validation_x[0].size(0),
                        0,
                    ).to(self.device),
                    cfg_scale=self.cfg_scale,
                ).reshape(self.validation_x[0].size(0), 5, 166, 32)
            )

        sample_surfaces = [s.moveaxis(2, 0) for s in sample_surfaces]

        figs = create_figs_1D_seq_spillpoint(
            sample_surfaces,
            conditions=torch.cat(
                [
                    self.validation_y[0]
                    .reshape(3, 8, 166, 32)
                    .moveaxis(2, 1)
                    .unsqueeze(2)
                ]
                * len(sample_surfaces),
                dim=2,
            ),
            img_dir=img_dir,
            save=True,
            pad_to_block=self.validation_y[1],
        )
