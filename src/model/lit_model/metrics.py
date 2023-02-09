from collections import defaultdict

import torch
import time
from src.utils import get_idx_val


def compute_L2(preds, targets, p=2, y_1_idxs=None):
    # return (preds-targets).square().sum(dim=(1,2,3)).sqrt()
    if preds.dim() == 3:
        return (preds - targets).abs().pow(p).mean(dim=-1).cpu().float()
        return torch.cdist(preds, targets, p=p).cpu().float() / preds.shape[2]
    elif preds.dim() == 4:
        return torch.cdist(
            preds.view(preds.shape[0], preds.shape[1], preds.shape[2] * preds.shape[3]),
            targets.view(
                preds.shape[0], preds.shape[1], preds.shape[2] * preds.shape[3]
            ),
            p=p,
        ).cpu().float() / (preds.shape[2] * preds.shape[3])


def compute_cond_dist(preds, targets, y_1_idxs):
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
            mean_metric = measures.mean(dim=0)
            metrics[name].append(
                (
                    min_metric.values.squeeze(),
                    mean_metric.squeeze(),
                    samples[min_metric.indices.squeeze()].view(
                        initial_samples_shape[1],
                        initial_samples_shape[2],
                        initial_samples_shape[3],
                    )
                    if len(initial_samples_shape) == 4
                    else samples[min_metric.indices.squeeze()],
                )
            )

    return metrics


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
        time_inference = 0
        for real_x, label in zip(x, y):
            start_inference = time.time()
            samples = inference_model(
                labels=torch.cat(
                    [label.unsqueeze(0)] * n_sample_for_metric,
                    dim=0,
                ).cuda()
            )
            end_inference = time.time()
            time_inference += end_inference - start_inference
            new_metrics = compute_metrics(
                samples,
                real_x=torch.stack([real_x] * n_sample_for_metric),
                y=torch.stack([label] * n_sample_for_metric),
                metrics_names=metrics_names,
                n_sample_for_metric=n_sample_for_metric,
            )
            for k, v in new_metrics.items():
                metrics_measures[f"{k}_min"] += [item[0].cpu() for item in v]
                metrics_measures[f"{k}_mean"] += [item[1].cpu() for item in v]
                metrics_samples[k] += [item[2].cpu() for item in v]
        metrics_measures["time_inference"] = [time_inference / len(x)]
        return metrics_measures, metrics_samples
    else:
        metrics_measures = defaultdict(list)
        metrics_samples = defaultdict(list)
        real_x = torch.cat([x] * n_sample_for_metric, dim=0)
        labels = torch.cat([y] * n_sample_for_metric, dim=0)
        start_inference = time.time()
        samples = inference_model(labels=labels)
        end_inference = time.time()
        new_metrics = compute_metrics(
            samples,
            real_x=real_x,
            y=labels,
            metrics_names=metrics_names,
            n_sample_for_metric=n_sample_for_metric,
        )
        for k, v in new_metrics.items():
            metrics_measures[f"{k}_min"] += [item[0].cpu() for item in v]
            metrics_measures[f"{k}_mean"] += [item[1].cpu() for item in v]
            metrics_samples[k] += [item[2].cpu() for item in v]
        metrics_measures["time_inference"] += [end_inference - start_inference]
        return metrics_measures, metrics_samples
