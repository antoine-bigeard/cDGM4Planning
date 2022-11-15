import yaml
import numpy as np
import pandas as pd
import torch
from PIL import Image
from random import randrange
import matplotlib.pyplot as plt
import os
import seaborn as sns

# from src.model.lit_model.metrics import *

# fmt: off
X0 = torch.Tensor([ 1.2862, -1.5020, -1.1767, -0.8801, -0.6243, -0.4191, -0.2709, -0.1829,
         -0.1547, -0.1822, -0.2578, -0.3714, -0.5105, -0.6610, -0.8087, -0.9394,
         -1.0402, -1.1003, -1.1115, -1.0692, -0.9719, -0.8219, -0.6251, -0.3905,
         -0.1296,  0.1442,  0.4162,  0.6718,  0.8971,  1.0796,  1.2095,  1.2795,
          1.2862,  1.2295,  1.1130,  0.9437,  0.7318,  0.4895,  0.2308, -0.0296,
         -0.2772, -0.4985, -0.6820, -0.8186, -0.9026, -0.9316, -0.9070, -0.8336,
         -0.7195, -0.5755, -0.4145, -0.2506, -0.0982,  0.0287,  0.1177,  0.1585,
          0.1436,  0.0690, -0.0659, -0.2578, -0.5003, -0.7836, -1.0955,  1.2862])
# fmt: on


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


def calculate_gradient_penalty(D, real_samples, fake_samples, labels, device):
    torch.set_grad_enabled(True)
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1))).to(device)
    labels = torch.Tensor(labels).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates, labels)
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
        fig = plt.figure()
        for s in surface:
            sns.heatmap(s.squeeze().detach().cpu(), cmap="hot", interpolation="nearest")

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


def create_figs_best_metrics(
    best_samples,
    y_1_idx,
    validation_x,
    img_dir=None,
    save=False,
    sequential_cond=False,
):
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
            plt.savefig(os.path.join(img_dir, f"test_best_metric_{i}"))

    return figs


def get_idx_val(y):
    # size [2, 64]
    if y.dim() == 2:
        nonzero_idx = torch.where(y[0, :])
        return nonzero_idx, y[1, nonzero_idx[1]]
    # size [B, 2, 64]
    if y.dim() == 3:
        nonzero_idx = torch.where(y[:, 0, :])
        return nonzero_idx[1], y[nonzero_idx[0], 1, nonzero_idx[1]]
    # size [B, Seq_size, 2, 64]
    if y.dim() == 4:
        idxs = torch.full((y.shape[0], y.shape[1]), -1, device=y.device)
        values = torch.full(
            (y.shape[0], y.shape[1]), -1, dtype=y.dtype, device=y.device
        )
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
        nonzero_idx = torch.where(y[:, 0, :, :])
        return nonzero_idx[1], y[nonzero_idx[0], 1, nonzero_idx[1], nonzero_idx[2]]
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
