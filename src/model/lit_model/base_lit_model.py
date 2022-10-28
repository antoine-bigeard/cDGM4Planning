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
from src.model.lit_model.lit_model_utils import *
from src.model.model.modules_diffusion import *
from src.model.model.DDPM import *

from src.utils import *

import copy


class BaseLitModel(pl.LightningModule):
    def __init__(
        self,
        validation_x_shape: np.ndarray = [5, 51, 1],
        log_dir: str = "",
        use_rd_y: bool = True,
        batch_size=512,
        n_sample_for_metric: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # instantiate models
        self.validation_x_shape = validation_x_shape
        self.log_dir = log_dir
        self.use_rd_y = use_rd_y
        self.batch_size = batch_size
        self.n_sample_for_metric = n_sample_for_metric
        self.kwargs = kwargs

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
        pass

    def inference(self, z, y):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def on_train_epoch_start(self) -> None:
        img_dir = os.path.join(self.log_dir, f"epoch_{self.current_epoch}")
        os.makedirs(img_dir, exist_ok=True)
        # log sampled images
        sample_surfaces = []
        for s in range(self.validation_z.size(0)):
            sample_surfaces.append(
                self.inference(
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
