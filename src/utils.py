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
import inspect
import collections.abc


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def read_yaml_config_file(path_config: str):
    with open(path_config) as conf:
        return yaml.load(conf, yaml.FullLoader)


def calculate_gradient_penalty(
    discriminator, real_samples, fake_samples, labels, device
):
    torch.set_grad_enabled(True)
    if real_samples.dim() == 3:
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1))).to(device)
    elif real_samples.dim() == 4:
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(
            device
        )
    labels = torch.Tensor(labels).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = discriminator(interpolates, labels)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(device)
    fake.requires_grad = False
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
                        edgecolors="white",
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
        for name, sample in best_samples.items():
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
            edgecolors="white",
        )


def create_figs_best_metrics_2D(
    best_samples,
    y_1_idx,
    validation_x,
    img_dir=None,
    save=False,
    sequential_cond=False,
):
    img_dir = os.path.join(img_dir, "images")
    paths = defaultdict(list)
    figs = []
    for i in range(len(y_1_idx[0])):
        figs.append({})
        sample_img_dir = os.path.join(img_dir, f"test_best_metric_{i}")
        os.makedirs(sample_img_dir, exist_ok=True)
        cmap = cm.viridis
        for name, sample in best_samples.items():
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
                    edgecolors="white",
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
                edgecolors="white",
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


def calculate_gravity_matrix(grid_size=32, cell_size=25, observation_height=50):
    G = np.zeros((grid_size, grid_size**2))

    for row in range(grid_size):
        for col in range(grid_size):
            cell_center_x = (col + 0.5) * cell_size
            cell_center_y = -(row + 0.5) * cell_size

            for obs_col in range(grid_size):
                obs_point_x = (obs_col + 0.5) * cell_size
                distance = np.sqrt(
                    (cell_center_x - obs_point_x) ** 2
                    + (cell_center_y - observation_height) ** 2
                )

                angle = np.arctan2(
                    observation_height - cell_center_y, abs(obs_point_x - cell_center_x)
                )

                vertical_component = np.sin(angle) / distance

                # vertical_component = (observation_height - cell_center_y) / distance

                G[obs_col, row * grid_size + col] = vertical_component

    return G


def plot_density_maps(
    models,
    y,
    num_cells,
    log_dir=None,
    curr_epoch=None,
    suffix=None,
    pred_grav_data=None,
):
    num_models = models.shape[0]
    if log_dir:
        os.makedirs(os.path.join(log_dir, str(curr_epoch)), exist_ok=True)

    for i, model in enumerate(models):
        # plot y (condition) on the side of density map on the sample plot
        plt.figure(figsize=(10, 10))

        plt.subplot(1, 2, 1)
        plt.plot(y[i])
        plt.title(f"Condition")

        plt.subplot(1, 2, 2)
        plt.imshow(
            model,
            cmap="viridis",
            extent=[0, num_cells, num_cells, 0],
            vmin=0,
            vmax=1,
        )
        plt.colorbar(label="Density")
        plt.title(f"Density Map")
        plt.xlabel("Cell")
        plt.ylabel("Depth")
        plt.tight_layout()

        if pred_grav_data is not None:
            plt.subplot(1, 2, 1)
            plt.plot(pred_grav_data[i])
            plt.title(f"Predicted Gravity Data")

        # plt.imshow(
        #     model,
        #     cmap="viridis",
        #     extent=[0, num_cells, num_cells, 0],
        #     # vmin=0,
        #     # vmax=1,
        #     # vmin=1.95,
        #     # vmax=2.85,
        # )

        # plt.colorbar(label="Density")
        # plt.title(f"Density Map")
        # plt.xlabel("Cell")
        # plt.ylabel("Depth")
        # plt.tight_layout()
        if log_dir:
            curr_path = os.path.join(
                log_dir,
                str(curr_epoch),
                f"density_map_{i}_{suffix}.png" if suffix else f"density_map_{i}.png",
            )
        plt.savefig(curr_path)
        plt.close()


def normalize_df(df, columns=["surfaces", "observations"]):
    for col in columns:
        df[col] = df[col].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df


def generate_G():
    # gravitational_constant = 6.67430e-11
    gravitational_constant = 1
    cell_size = 25
    nd = 32
    measurement_height = 50
    start_measurement = 200
    end_measurement = 600

    def compute_distance(cell_center, measurement_point):
        return np.sqrt(
            (cell_center[0] - measurement_point[0]) ** 2
            + (cell_center[1] - measurement_point[1]) ** 2
        )

    G = np.zeros((nd, nd**2))
    cell_centers = [
        (i * cell_size + cell_size / 2, j * cell_size + cell_size / 2)
        for i in range(nd)
        for j in range(nd)
    ]
    measurement_points = [
        (np.linspace(start_measurement, end_measurement, nd)[i], measurement_height)
        for i in range(nd)
    ]

    for i, measurement_point in enumerate(measurement_points):
        for j, cell_center in enumerate(cell_centers):
            distance = compute_distance(cell_center, measurement_point)
            G[i, j] = gravitational_constant * cell_size**2 * distance**-2

    return torch.Tensor(G)


def get_variable_name(var):
    frame = inspect.currentframe().f_back
    for name, value in frame.f_locals.items():
        if value is var:
            return name
