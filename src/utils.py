import yaml
import numpy as np
import pandas as pd
import torch
from PIL import Image
from random import randrange
import matplotlib.pyplot as plt
import os
from src.model.lit_model.metrics import *

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
            # torch.manual_seed(123)
            # y[i, 1, :] = torch.rand(shape[2]) * single_one
            y[i, 1, :] = X0 * single_one
    if return_1_idx:
        return y, y_ones_idx
    return y


def calculate_gradient_penalty(D, real_samples, fake_samples, labels, device):
    torch.set_grad_enabled(True)
    """Calculates the gradient penalty loss for WGAN GP.
    Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
    the interpolated real and fake samples, as in the WGAN GP paper.
    """
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


def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot


def create_figs(
    sample_surfaces,
    y_1_idxs,
    validation_y,
    img_dir=None,
    save=False,
):
    figs = []
    for i, surface in enumerate(sample_surfaces):
        fig = plt.figure()
        for s in surface:
            plt.plot(s.squeeze().detach().cpu(), color="blue")

        observation_pt = (
            y_1_idxs[i],
            validation_y[i, :, y_1_idxs[i]][1].cpu(),
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
):
    figs = []
    for idx in y_1_idx:
        fig = plt.figure()
        for (sample, name) in best_samples:
            plt.plot(sample[idx].squeeze().detach().cpu(), label=name)
        plt.plot(validation_x.squeeze().detach().cpu(), label="x")
        observation_pt = (
            idx,
            validation_x[0, idx].cpu(),
        )
        plt.scatter([observation_pt[0]], [observation_pt[1]], s=25, c="r")
        plt.legend(loc="upper left")
        figs.append(fig)
        if save:
            os.makedirs(img_dir, exist_ok=True)
            plt.savefig(os.path.join(img_dir, f"test_best_metric_{idx}"))

    return figs


def compute_best_pred(model, val_z_best_metrics, x, y):
    z = val_z_best_metrics.cuda()
    input_y = y.clone().cuda()
    idx = torch.where(input_y[0] == 1)
    input_y[1, idx] = x[0, idx]
    input_y = torch.cat([input_y.unsqueeze(0) for i in range(z.size(0))])
    samples = model(z, input_y)
    l2_metric = L2_metric(
        samples,
        (
            torch.cat([x.unsqueeze(0) for i in range(z.size(0))], dim=0),
            input_y,
        ),
        p=2,
    )
    dist_y = dist_to_y_metric(
        samples,
        (
            torch.cat([x.unsqueeze(0) for i in range(z.size(0))], dim=0),
            input_y,
        ),
    )
    best_l2_metric_idx = l2_metric.argmin(dim=0, keepdim=True)
    best_dist_y_idx = dist_y.argmin(dim=0, keepdim=True)

    return (
        int(idx[0].squeeze()),
        dist_y[best_dist_y_idx].squeeze(),
        l2_metric[best_l2_metric_idx].squeeze(),
        samples[best_dist_y_idx.squeeze()],
        samples[best_l2_metric_idx.squeeze()],
    )
