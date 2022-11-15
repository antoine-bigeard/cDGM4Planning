from torchmetrics import Metric
import torch
import torch.nn as nn
from torchmetrics.functional import pearson_corrcoef
from src.utils import get_idx_val


def L2_metric(preds, target, p):
    return torch.cdist(preds, target[0], p=p).cpu().float()


def dist_to_y_metric(preds, target):
    idx1 = torch.where(target[1][:, 0, :])
    return (
        (preds[idx1[0], :, idx1[1]] - target[0][idx1[0], :, idx1[1]])
        .abs()
        .cpu()
        .float()
    )


class L2Metric(Metric):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
        self.add_state("distance", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        self.distance = (
            self.distance
            + (
                (torch.cdist(preds, target[0], p=self.p))
                / (torch.max(target[0]) - torch.min(target[0]))
            )
            .mean()
            .cpu()
            .float()
        )

    def compute(self):
        return self.distance.float()


class DistanceToY(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("distance_to_y", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        idx1 = target[1][:, 0, :].nonzero()
        self.distance_to_y = (
            self.distance_to_y
            + (
                (
                    preds[idx1[:, 0], :, idx1[:, 1]]
                    - target[0][idx1[:, 0], :, idx1[:, 1]]
                )
                / (torch.max(preds, dim=-1)[0] - torch.min(preds, dim=-1)[0])
            ).mean()
        )

    def compute(self):
        return self.distance_to_y.float()


class Pearson(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("pearson_corr", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.pearson_corr = (
            self.pearson_corr
            + pearson_corrcoef(preds.squeeze().T, target=target[0].squeeze().T).mean()
        )

    def compute(self):
        return self.pearson_corr.float()


class BestL2(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("best_L2_metric", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.best_L2_metric = self.best_L2_metric

    def compute(self):
        return self.pearson_corr.float()


def compute_L2(preds, targets, p=2):
    return torch.cdist(preds, targets, p=p).cpu().float()


def compute_cond_dist(preds, targets, y_1_idxs):
    valid_obs = y_1_idxs[0][y_1_idxs[0] != -1]
    dists = (
        (preds[:, 0, valid_obs] - targets[:, 0, valid_obs])
        .abs()
        .mean(dim=-1)
        .cpu()
        .float()
    )

    return dists


def compute_metrics(samples, real_x, y):
    metrics = {}
    y_1_idxs = get_idx_val(y.unsqueeze(0))
    batched_real = torch.cat([real_x.unsqueeze(0) for i in range(samples.shape[0])])
    # best L2
    min_L2 = compute_L2(samples, batched_real).min(dim=0)
    metrics["best_L2"] = (min_L2.values, samples[min_L2.indices.squeeze()])

    # best cond
    cond_dists = compute_cond_dist(
        samples,
        batched_real,
        (
            torch.stack([y_1_idxs[0]] * samples.shape[0]),
            torch.stack([y_1_idxs[1]] * samples.shape[0]),
        ),
    )
    min_cond_dist = cond_dists.min(dim=0)
    metrics["best_cond_dist"] = (
        min_cond_dist.values,
        samples[min_cond_dist.indices.squeeze()],
    )
    metrics["mean_cond_dist"] = cond_dists.mean(dim=0)

    # variance
    std_dim0 = torch.std(samples, dim=0)
    metrics["std"] = (std_dim0.mean().cpu().float(), std_dim0)

    return metrics


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


def measure_metrics(inference_model, x, y, n_sample_for_metric):
    metrics = {
        "best_L2": 0,
        "best_cond_dist": 0,
        "mean_cond_dist": 0,
        "std": 0,
    }
    best_L2_measures = []
    best_L2_sample = []
    best_cond_dist_sample = []
    std = []
    for i, (real, label) in enumerate(zip(x, y)):
        samples = inference_model(
            labels=torch.cat(
                [label.unsqueeze(0)] * n_sample_for_metric,
                dim=0,
            ).cuda()
        )
        new_metrics = compute_metrics(samples, real_x=real, y=label)
        for k, v in new_metrics.items():
            if k in ["best_L2", "best_cond_dist", "std"]:
                metrics[k] += v[0]
            else:
                metrics[k] += v
        best_L2_measures.append(new_metrics["best_L2"][0].squeeze().detach().cpu())
        best_L2_sample.append(new_metrics["best_L2"][1])
        best_cond_dist_sample.append(new_metrics["best_cond_dist"][1])
        std.append(new_metrics["std"][1])
    metrics = {k: v / x.shape[0] for k, v in metrics.items()}
    return (
        metrics,
        torch.stack(best_L2_sample),
        torch.stack(best_cond_dist_sample),
        torch.stack(std),
        best_L2_measures,
    )


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
