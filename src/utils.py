import yaml
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.cm as cm
from random import randrange
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random as rd


# from src.model.lit_model.metrics import *


def read_yaml_config_file(path_config: str):
    with open(path_config) as conf:
        return yaml.load(conf, yaml.FullLoader)


def random_observation(shape, return_1_idx=False, random=True):
    y = torch.ones(shape[0], 2, shape[2])
    if random:
        y_ones_idx = [randrange(start=0, stop=shape[2]) for i in range(shape[0])]
    else:
        y_ones_idx = range(8, shape[2], int(shape[2] / shape[0]))
    for i in range(shape[0]):
        single_one = torch.tensor(
            [1 if j == y_ones_idx[i] else 0 for j in range(shape[2])]
        )
        y[i, 0, :] = single_one
        if random:
            y[i, 1, :] = torch.rand(shape[2]) * single_one
        else:
            y[i, 1, :] = X0 * single_one
    if return_1_idx:
        return y, y_ones_idx
    return y


def calculate_gradient_penalty(
    discriminator, real_samples, fake_samples, labels, device
):
    torch.set_grad_enabled(True)
    # Random weight term for interpolation between real and fake samples
    if real_samples.dim() == 3:
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1))).to(device)
    elif real_samples.dim() == 4:
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(
            device
        )
    labels = torch.Tensor(labels).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = discriminator(interpolates, labels)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(device)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def create_figs(
    sample_surfaces,
    y_1_idxs,
    img_dir=None,
    save=False,
    sequential_cond=False,
):
    figs = []
    for i, surface in enumerate(sample_surfaces):
        fig = plt.figure()
        for s in surface:
            plt.plot(s.squeeze().detach().cpu(), color="blue")

        if not sequential_cond:
            # observation_pt = (y_1_idxs[i], validation_y[i, 1, y_1_idxs[i]].cpu())
            observation_pt = (y_1_idxs[0][i].cpu(), y_1_idxs[1][i].cpu())
            plt.scatter([observation_pt[0]], [observation_pt[1]], s=25, c="r")
        else:
            for j in range(len(y_1_idxs[0][i])):
                if y_1_idxs[0][i][j] != -1:
                    observation_pt = (
                        y_1_idxs[0][i][j].cpu(),
                        y_1_idxs[1][i][j].cpu(),
                    )
                    plt.scatter([observation_pt[0]], [observation_pt[1]], s=25, c="r")
        # plt.ylim((-3, -3))
        if save:
            plt.savefig(os.path.join(img_dir, f"test_{i}"))
        figs.append(fig)
    return figs


def test0(a, b):
    return b if a == 0 else a


def create_figs_2D(
    sample_surfaces,
    y_1_idxs,
    img_dir=None,
    save=False,
    sequential_cond=False,
):
    figs = []
    for i, surface in enumerate(sample_surfaces):
        cmap = cm.viridis
        for idx_s, s in enumerate(surface):
            sample_img_dir = os.path.join(img_dir, f"test_{i}")
            os.makedirs(sample_img_dir, exist_ok=True)
            fig = plt.figure()
            plt.imshow(s.squeeze().detach().cpu(), cmap=cmap)
            if not sequential_cond:
                for j in range(len(y_1_idxs[0][i])):
                    observation_pt = (y_1_idxs[0][i][j].cpu(), y_1_idxs[1][i][j].cpu())
                    plt.scatter(
                        [observation_pt[0][1]],
                        [observation_pt[0][0]],
                        color=cmap(observation_pt[1]),
                        marker=".",
                        s=150,
                        edgecolors="black",
                    )
            else:
                for j in range(len(y_1_idxs[0][i])):
                    if y_1_idxs[0][i][j] != -1:
                        observation_pt = (
                            y_1_idxs[0][i][j].cpu(),
                            y_1_idxs[1][i][j].cpu(),
                        )
                        plt.scatter(
                            [observation_pt[0]], [observation_pt[1]], s=25, c="r"
                        )
            if save:
                plt.savefig(os.path.join(sample_img_dir, f"sample_{idx_s}"))
            figs.append(fig)
            plt.close()
    return figs


def create_figs_best_metrics(
    best_samples,
    y_1_idx,
    validation_x,
    img_dir=None,
    save=False,
    sequential_cond=False,
):
    paths = []
    figs = []
    for i in range(y_1_idx[0].shape[0]):
        fig = plt.figure()
        for (name, sample) in best_samples.items():
            s = sample[i].squeeze().detach().cpu()
            if name == "std":
                samp = best_samples["best_L2"][i].squeeze().detach().cpu()
                down = samp - 2 * s
                up = samp + 2 * s
                plt.fill_between(
                    np.array([i for i in range(samp.shape[-1])]), down, up, alpha=0.2
                )
            else:
                plt.plot(s, label=name)
        plt.plot(validation_x[i].squeeze().detach().cpu(), label="real_obs")
        if not sequential_cond:
            observation_pt = (y_1_idx[0][i].cpu(), y_1_idx[1][i].cpu())
            plt.scatter([observation_pt[0]], [observation_pt[1]], s=25, c="r")
        else:
            for j in range(len(y_1_idx[0][i])):
                if y_1_idx[0][i][j] != -1:
                    observation_pt = (
                        y_1_idx[0][i][j].cpu(),
                        y_1_idx[1][i][j].cpu(),
                    )
                    plt.scatter([observation_pt[0]], [observation_pt[1]], s=25, c="r")
        plt.legend(loc="upper left")
        figs.append(fig)
        if save:
            os.makedirs(img_dir, exist_ok=True)
            path_img = os.path.join(img_dir, f"test_best_metric_{i}.png")
            plt.savefig(path_img)
            paths.append(path_img)
    return figs, paths


def plot_fig_2D(ax, sample, y_1_idx, i):
    cmap = cm.viridis
    ax.imshow(sample.squeeze().detach().cpu(), cmap=cmap)
    for j in range(len(y_1_idx[0][i])):
        observation_pt = (y_1_idx[0][i][j].cpu(), y_1_idx[1][i][j].cpu())
        ax.scatter(
            [observation_pt[0][1]],
            [observation_pt[0][0]],
            color=cmap(observation_pt[1]),
            marker=".",
            s=150,
            edgecolors="black",
        )


def create_figs_best_metrics_2D(
    best_samples,
    y_1_idx,
    validation_x,
    img_dir=None,
    save=False,
    sequential_cond=False,
):
    paths = defaultdict(list)
    figs = []
    for i in range(len(y_1_idx[0])):
        figs.append({})
        sample_img_dir = os.path.join(img_dir, f"test_best_metric_{i}")
        os.makedirs(sample_img_dir, exist_ok=True)
        cmap = cm.viridis
        for (name, sample) in best_samples.items():
            fig = plt.figure()
            plt.imshow(sample[i].squeeze().detach().cpu(), cmap=cmap)
            for j in range(len(y_1_idx[0][i])):
                observation_pt = (y_1_idx[0][i][j].cpu(), y_1_idx[1][i][j].cpu())
                plt.scatter(
                    [observation_pt[0][1]],
                    [observation_pt[0][0]],
                    color=cmap(observation_pt[1]),
                    marker=".",
                    s=150,
                    edgecolors="black",
                )
            if save:
                plt.savefig(os.path.join(sample_img_dir, f"{name}_sample"))
                paths[name].append(os.path.join(sample_img_dir, f"{name}_sample.png"))
            figs[-1][name] = fig
            plt.close()

        fig = plt.figure()
        plt.imshow(validation_x[i].squeeze().detach().cpu(), label="real_obs")
        for j in range(len(y_1_idx[0][i])):
            observation_pt = (y_1_idx[0][i][j].cpu(), y_1_idx[1][i][j].cpu())
            plt.scatter(
                [observation_pt[0][1]],
                [observation_pt[0][0]],
                color=cmap(observation_pt[1]),
                marker=".",
                s=150,
                edgecolors="black",
            )
        if save:
            plt.savefig(os.path.join(sample_img_dir, f"ground_truth"))
            paths["ground_truth"].append(os.path.join(sample_img_dir, f"ground_truth"))
        plt.close()
        figs[-1]["ground_truth"] = fig
    return figs, paths


def get_idx_val(y, multiple_obs=False):
    # size [2, 64]
    if y.dim() == 2:
        nonzero_idx = torch.where(y[0, :])
        return nonzero_idx, y[1, nonzero_idx[1]]
    # size [B, 2, 64]
    if y.dim() == 3 and not multiple_obs:
        nonzero_idx = torch.where(y[:, 0, :])
        return nonzero_idx[1], y[nonzero_idx[0], 1, nonzero_idx[1]]
    if y.dim() == 3 and multiple_obs:
        nonzero_idx = torch.where(y[:, 0, :])
        return nonzero_idx, y[nonzero_idx[0], 1, nonzero_idx[1]]
    # size [B, Seq_size, 2, 64]
    if y.dim() == 4:
        nonzero_idx = torch.where(y[:, :, 0, :] == 1)
        idxs[nonzero_idx[0], nonzero_idx[1]] = nonzero_idx[2]
        values[nonzero_idx[0], nonzero_idx[1]] = y[
            nonzero_idx[0], nonzero_idx[1], 1, nonzero_idx[2]
        ]
        return idxs, values


def get_idx_val_2D(y):
    # size [2, 64, 64]
    if y.dim() == 3:
        nonzero_idx = torch.where(y[0, :, :])
        return nonzero_idx, y[1, nonzero_idx[1], nonzero_idx[2]]
    # size [B, 2, 64, 64]
    if y.dim() == 4:
        idxs, values = [], []
        for i in range(y.shape[0]):
            nonzero_idx = torch.where(y[[i], 0, :, :])
            idxs.append(
                torch.cat(
                    [nonzero_idx[1].unsqueeze(-1), nonzero_idx[2].unsqueeze(-1)], dim=-1
                )
            )
            values.append(
                y[nonzero_idx[0] + i, 1, nonzero_idx[1], nonzero_idx[2]],
            )
        return idxs, values
    # size [B, Seq_size, 2, 64, 64]
    if y.dim() == 5:
        idxs = torch.full((y.shape[0], y.shape[1], 2), -1, device=y.device)
        values = torch.full(
            (y.shape[0], y.shape[1]), -1, dtype=y.dtype, device=y.device
        )
        nonzero_idx = torch.where(y[:, :, 0, :, :] == 1)
        idxs[nonzero_idx[0], nonzero_idx[1]] = torch.stack(
            [nonzero_idx[2], nonzero_idx[3]]
        )
        values[nonzero_idx[0], nonzero_idx[1], nonzero_idx[2]] = y[
            nonzero_idx[0], nonzero_idx[1], 1, nonzero_idx[2], nonzero_idx[3]
        ]
        return idxs, values


def exp_distrib(lbda=0.2):
    return lambda: int(np.random.exponential(scale=1 / lbda)) + 1


def random_observation_ore_maps(
    x: torch.Tensor,
    distribution=exp_distrib(lbda=0.2),
    seed=False,
):  # x of shape [B, 1, h, w] or [B, 1, w]
    if seed:
        rd.seed(1234)
    if x.dim() == 4:
        n, m = x.shape[2], x.shape[3]
        xy_pos = []
        for i in range(n):
            for j in range(m):
                xy_pos.append((i, j))
        observations = []
        n, m = x.shape[2], x.shape[3]
        n_obs = distribution()
        for i in range(x.shape[0]):
            new_obs = np.zeros((2, n, m))
            temp_xy = rd.sample(xy_pos, n_obs)
            for k, l in temp_xy:
                new_obs[0, k, l] = 1
                new_obs[1, k, l] = x[i, 0, k, l]
            observations.append(new_obs)
        return torch.Tensor(np.stack(observations))


def keep_samples(measures: torch.Tensor, n: int):
    minn = measures.min()
    inter = (measures.max() - minn) / n
    return [measures[measures <= i * inter + minn].max() for i in range(n)]
