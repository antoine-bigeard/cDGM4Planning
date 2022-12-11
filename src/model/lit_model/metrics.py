from collections import defaultdict

from torchmetrics import Metric
import torch
import torch.nn as nn
from torchmetrics.functional import pearson_corrcoef
from src.utils import get_idx_val, get_idx_val_2D
from time import time


def compute_L2(preds, targets, p=2, y_1_idxs=None):
    # return (preds-targets).square().sum(dim=(1,2,3)).sqrt()
    if preds.dim() == 3:
        return torch.cdist(preds, targets, p=p).cpu().float()
    elif preds.dim() == 4:
        return (
            torch.cdist(
                preds.view(
                    preds.shape[0], preds.shape[1], preds.shape[2] * preds.shape[3]
                ),
                targets.view(
                    preds.shape[0], preds.shape[1], preds.shape[2] * preds.shape[3]
                ),
                p=p,
            )
            .cpu()
            .float()
        )


def compute_cond_dist(preds, targets, y_1_idxs):

    # valid_obs = y_1_idxs[0][y_1_idxs[0] != -1]
    dists = (
        (preds[:, 0, y_1_idxs[0]] - targets[:, 0, y_1_idxs[0]])
        .abs()
        .mean(dim=-1)
        .cpu()
        .float()
    )

    return dists


METRICS = {
    "L2": compute_L2,
    "dist_cond": compute_cond_dist,
    "L1": lambda preds, targets, y_1_idxs: compute_L2(
        preds, targets, p=1, y_1_idxs=y_1_idxs
    ),
}


def compute_metrics(samples, real_x, y, metrics_names, n_sample_for_metric):
    initial_samples_shape = samples.shape
    if samples.dim() == 4:
        samples = samples.view(
            samples.shape[0], samples.shape[1], samples.shape[2] * samples.shape[3]
        )
        real_x = real_x.view(
            real_x.shape[0], real_x.shape[1], real_x.shape[2] * real_x.shape[3]
        )
        y = y.view(y.shape[0], y.shape[1], y.shape[2] * y.shape[3])

    metrics = defaultdict(list)
    for name in metrics_names:
        # batched_measures = METRICS[name](samples, real_x, y_1_idxs=y_1_idxs).squeeze()
        n_batch = real_x.shape[0] // n_sample_for_metric
        for j in range(n_batch):
            measures = METRICS[name](
                samples[[j + n_batch * k for k in range(n_sample_for_metric)]],
                real_x[[j + n_batch * k for k in range(n_sample_for_metric)]],
                y_1_idxs=get_idx_val(
                    y[[j + n_batch * k for k in range(n_sample_for_metric)]]
                ),
            ).squeeze()
            min_metric = measures.min(dim=0)
            metrics[name].append(
                (
                    min_metric.values.squeeze(),
                    samples[min_metric.indices.squeeze()].view(
                        initial_samples_shape[1],
                        initial_samples_shape[2],
                        initial_samples_shape[3],
                    )
                    if len(initial_samples_shape) == 4
                    else samples[min_metric.indices.squeeze()],
                )
            )
    # if "std" in metrics_names:
    #     std_dim0 = torch.std(samples, dim=0)
    #     metrics["std"] = (std_dim0.mean().squeeze().cpu().float(), std_dim0)

    return metrics


# def compute_metrics(samples, real_x, y):
#     initial_samples_shape = samples.shape
#     if samples.dim() == 4:
#         samples = samples.view(
#             samples.shape[0], samples.shape[1], samples.shape[2] * samples.shape[3]
#         )
#         real_x = real_x.view(real_x.shape[0], real_x.shape[1] * real_x.shape[2])
#         y = y.view(y.shape[0], y.shape[1] * y.shape[2])
#     metrics = {}
#     y_1_idxs = get_idx_val(y.unsqueeze(0))
#     batched_real = torch.cat([real_x.unsqueeze(0) for i in range(samples.shape[0])])
#     # best L2
#     min_L2 = compute_L2(samples, batched_real).min(dim=0)
#     metrics["best_L2"] = (
#         min_L2.values.squeeze(),
#         samples[min_L2.indices.squeeze()].view(
#             initial_samples_shape[1], initial_samples_shape[2], initial_samples_shape[3]
#         )
#         if len(initial_samples_shape) == 4
#         else samples[min_cond_dist.indices.squeeze()],
#     )

#     # best cond
#     cond_dists = compute_cond_dist(
#         samples,
#         batched_real,
#         (
#             torch.stack([y_1_idxs[0]] * samples.shape[0]),
#             torch.stack([y_1_idxs[1]] * samples.shape[0]),
#         ),
#     )
#     min_cond_dist = cond_dists.min(dim=0)
#     metrics["best_cond_dist"] = (
#         min_cond_dist.values.squeeze(),
#         samples[min_cond_dist.indices.squeeze()].view(
#             initial_samples_shape[1], initial_samples_shape[2], initial_samples_shape[3]
#         )
#         if len(initial_samples_shape) == 4
#         else samples[min_cond_dist.indices.squeeze()],
#     )
#     metrics["mean_cond_dist"] = cond_dists.mean(dim=0)

#     # variance
#     std_dim0 = torch.std(samples, dim=0)
#     metrics["std"] = (std_dim0.mean().squeeze().cpu().float(), std_dim0)

#     return metrics


def compute_val_metric(self, x):
    z = torch.randn(self.n_sample_for_metric, self.latent_dim).to(self.device)
    best_dist_y = {}
    best_L2 = {}
    for label, idx in zip(self.validation_y, self.y_1_idxs):
        for surf in x[:10]:
            input_y = label.clone()
            input_y[1, idx] = surf[:, idx]
            input_y = torch.cat([input_y.unsqueeze(0) for i in range(z.size(0))])
            samples = self(z, input_y)
            l2_metric = L2_metric(
                samples,
                (
                    torch.cat([surf.unsqueeze(0) for i in range(z.size(0))], dim=0),
                    input_y,
                ),
                p=2,
            ).min()
            dist_y = dist_to_y_metric(
                samples,
                (
                    torch.cat([surf.unsqueeze(0) for i in range(z.size(0))], dim=0),
                    input_y,
                ),
                p=2,
            ).min()

            if idx in best_dist_y:
                best_dist_y[idx] += dist_y
                best_L2[idx] += l2_metric
            else:
                best_dist_y[idx] = dist_y
                best_L2[idx] = l2_metric
        best_dist_y[idx] = best_dist_y[idx] / 10
        best_L2[idx] = best_L2[idx] / 10
    return best_dist_y, best_L2


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


def measure_metrics(
    inference_model,
    x: torch.Tensor,
    y: torch.Tensor,
    n_sample_for_metric,
    metrics_names,
    no_batch=False,
):
    if no_batch:
        metrics_measures = defaultdict(list)
        metrics_samples = defaultdict(list)
        for real_x, label in zip(x, y):
            samples = inference_model(
                labels=torch.cat(
                    [label.unsqueeze(0)] * n_sample_for_metric,
                    dim=0,
                ).cuda()
            )
            # metrics_measures["samples_per_sec"].append(n_sample_for_metric / (end - start))
            new_metrics = compute_metrics(
                samples,
                real_x=torch.stack([real_x] * n_sample_for_metric),
                y=torch.stack([label] * n_sample_for_metric),
                metrics_names=metrics_names,
                n_sample_for_metric=n_sample_for_metric,
            )
            for k, v in new_metrics.items():
                metrics_measures[k] += [item[0] for item in v]
                metrics_samples[k] += [item[1] for item in v]
                # else:
                #     metrics_measures[k].append(v)
        return metrics_measures, metrics_samples
    else:
        metrics_measures = defaultdict(list)
        metrics_samples = defaultdict(list)
        real_x = torch.cat([x] * n_sample_for_metric, dim=0)
        labels = torch.cat([y] * n_sample_for_metric, dim=0)
        samples = inference_model(labels=labels)
        # metrics_measures["samples_per_sec"].append(n_sample_for_metric / (end - start))
        new_metrics = compute_metrics(
            samples,
            real_x=real_x,
            y=labels,
            metrics_names=metrics_names,
            n_sample_for_metric=n_sample_for_metric,
        )
        for k, v in new_metrics.items():
            metrics_measures[k] += [item[0] for item in v]
            metrics_samples[k] += [item[1] for item in v]
            # else:
            #     metrics_measures[k].append(v)
        return metrics_measures, metrics_samples


# def measure_metrics(inference_model, x, y, n_sample_for_metric):
#     metrics = {
#         "best_L2": 0,
#         "best_cond_dist": 0,
#         "mean_cond_dist": 0,
#         "std": 0,
#     }
#     best_L2_measures = []
#     best_L2_sample = []
#     best_cond_dist_sample = []
#     best_cond_dist_measures = []
#     std = []
#     samples_per_sec = []
#     for i, (real, label) in enumerate(zip(x, y)):
#         start = time()
#         samples = inference_model(
#             labels=torch.cat(
#                 [label.unsqueeze(0)] * n_sample_for_metric,
#                 dim=0,
#             ).cuda()
#         )
#         end = time()
#         samples_per_sec.append(n_sample_for_metric / (end - start))
#         new_metrics = compute_metrics(samples, real_x=real, y=label)
#         for k, v in new_metrics.items():
#             if k in ["best_L2", "best_cond_dist", "std"]:
#                 metrics[k] += v[0]
#             else:
#                 metrics[k] += v
#         best_L2_measures.append(new_metrics["best_L2"][0].squeeze().detach().cpu())
#         best_L2_sample.append(new_metrics["best_L2"][1])
#         best_cond_dist_sample.append(new_metrics["best_cond_dist"][1])
#         best_cond_dist_measures.append(
#             new_metrics["best_cond_dist"][0].squeeze().detach().cpu()
#         )
#         std.append(new_metrics["std"][1])
#     metrics = {k: v / x.shape[0] for k, v in metrics.items()}
#     return (
#         metrics,
#         torch.stack(best_L2_sample),
#         torch.stack(best_cond_dist_sample),
#         torch.stack(std),
#         best_L2_measures,
#         best_cond_dist_measures,
#         samples_per_sec,
#     )


def histogram(
    L2_measures: torch.Tensor, bins=50, std_fact=3, density=False
) -> torch.Tensor:
    std = L2_measures.std()
    mean = L2_measures.mean()
    range = (mean - std_fact * std, mean + std_fact * std)
    return torch.histogram(L2_measures, bins=bins, range=range, density=density)


def cum_density(L2_measures: torch.Tensor, bins=50, std_fact=3) -> torch.Tensor:
    density = histogram(L2_measures, bins, std_fact, density=True)
    return density.cumsum()
