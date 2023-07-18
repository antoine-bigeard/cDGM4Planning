import torch
from tqdm import tqdm

import math

import torch
from torch import nn
import torch.nn.functional as F

from src.utils import create_figs_1D_seq_spillpoint
import os

import matplotlib.pyplot as plt


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        surf_size=64,
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.surf_size = surf_size

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        t = t.to(self.alpha.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None].to(x.device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None].to(
            x.device
        )
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(
        self, model, labels, cfg_scale=3, labels_padding_masks=None, log_dir=None
    ):
        n = labels.shape[0]
        model.eval()
        if log_dir is not None:
            img_dir = os.path.join(log_dir, "diffusion")
            os.makedirs(img_dir, exist_ok=True)
            all_samples = []
        with torch.inference_mode():
            x = torch.randn((n, model.c_out, self.surf_size)).cuda()
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(x.device)
                predicted_noise = model(
                    x, t, labels, src_padding_mask=labels_padding_masks
                )
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                alpha = self.alpha[t.to(self.alpha.device)][:, None, None].to(x.device)
                alpha_hat = self.alpha_hat[t.to(self.alpha.device)][:, None, None].to(
                    x.device
                )
                beta = self.beta[t.to(self.alpha.device)][:, None, None].to(x.device)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
                if log_dir is not None:
                    all_samples.append(x.detach().cpu().numpy())
        model.train()
        if log_dir is not None:
            for idx, s in enumerate(all_samples):
                plt.figure()
                plt.plot(s[0, 0])
                plt.savefig(os.path.join(img_dir, f"{idx}.png"))
                plt.close()
        return x


class DiffusionTransformer:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        surf_size=64,
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.surf_size = surf_size

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        t = t.to(self.alpha.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to(x.device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ].to(x.device)
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(
        self, transformer, observations, cfg_scale=3, log_dir=None, test_diff=False
    ):
        n = observations.shape[0]
        transformer.eval()
        if log_dir is not None:
            img_dir = os.path.join(log_dir, "diffusion")
            os.makedirs(img_dir, exist_ok=True)
            all_samples = []
        with torch.inference_mode():
            x = torch.randn(
                (n, observations.size(1), transformer.output_channels, self.surf_size)
            ).cuda()
            if test_diff:
                x = observations.unsqueeze(0)
                n = 1
                # self.noise_steps = 500
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(x.device)
                # predicted_noise = transformer.generate_out_sequence(x, observations, t)
                predicted_noise = transformer(x, t, observations, None, None)
                alpha = self.alpha[t.to(self.alpha.device)][:, None, None, None].to(
                    x.device
                )
                alpha_hat = self.alpha_hat[t.to(self.alpha.device)][
                    :, None, None, None
                ].to(x.device)
                beta = self.beta[t.to(self.alpha.device)][:, None, None, None].to(
                    x.device
                )
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
                if log_dir is not None:
                    all_samples.append(x.cpu())
        if log_dir is not None:
            # plot all first elements of the samples and save them
            for idx, s in enumerate(all_samples):
                plt.figure()
                plt.plot(s[0, 0, 0].numpy())
                plt.savefig(os.path.join(img_dir, f"{idx}.png"))
                plt.close()
        transformer.train()
        return x

        # n = observations.shape[0]
        # transformer.eval()
        # with torch.inference_mode():
        #     surfaces = torch.randn(
        #         (n, observations.size(1), transformer.output_channels, self.surf_size)
        #     ).cuda()
        #     for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
        #         t = (torch.ones(n) * i).long().to(surfaces.device)
        #         predicted_noise = transformer.generate_out_sequence(
        #             surfaces, observations.cuda(), t
        #         )
        #         alpha = self.alpha[t.to(self.alpha.device)][:, None, None, None].cuda()
        #         alpha_hat = self.alpha_hat[t.to(self.alpha.device)][
        #             :, None, None, None
        #         ].cuda()
        #         beta = self.beta[t.to(self.alpha.device)][:, None, None, None].cuda()
        #         if i > 1:
        #             noise = torch.randn_like(surfaces)
        #         else:
        #             noise = torch.zeros_like(surfaces)
        #         surfaces = (
        #             1
        #             / torch.sqrt(alpha)
        #             * (
        #                 surfaces
        #                 - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
        #             )
        #             + torch.sqrt(beta) * noise
        #         )
        #         if log_dir is not None:
        #             plt.figure()
        #             plt.plot(surfaces[0, 0, 0, :].cpu().numpy())
        #             os.makedirs(os.path.join(log_dir, "validation"), exist_ok=True)
        #             plt.savefig(
        #                 os.path.join(
        #                     log_dir,
        #                     "validation",
        #                     f"denosing_step_{t.item()}.png",
        #                 )
        #             )
        #             plt.close()
        #         # if log_dir is not None and t.item() % 20 == 0:
        #         #     create_figs_1D_seq_spillpoint(
        #         #         surfaces.unsqueeze(2),
        #         #         conditions=None,
        #         #         img_dir=os.path.join(
        #         #             log_dir,
        #         #             "validation",
        #         #             f"denosing_step_{t.item()}",
        #         #             "surfaces_t",
        #         #         ),
        #         #         save=True,
        #         #         test=True,
        #         #     )
        # transformer.train()
        # return surfaces


class DiffusionTransformer2:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        surf_size=64,
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.surf_size = surf_size

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        t = t.to(self.alpha.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None].to(x.device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None].to(
            x.device
        )
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, observations, cfg_scale=3):
        n = observations.shape[0]
        with torch.inference_mode():
            surfaces = torch.randn((n, model.c_out, observations.size(2))).cuda()
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(surfaces.device)
                predicted_noise = model(surfaces, t, observations.cuda())
                alpha = self.alpha[t.to(self.alpha.device)][:, None, None].cuda()
                alpha_hat = self.alpha_hat[t.to(self.alpha.device)][
                    :, None, None
                ].cuda()
                beta = self.beta[t.to(self.alpha.device)][:, None, None].cuda()
                if i > 1:
                    noise = torch.randn_like(surfaces)
                else:
                    noise = torch.zeros_like(surfaces)
                surfaces = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        surfaces
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
        return surfaces
