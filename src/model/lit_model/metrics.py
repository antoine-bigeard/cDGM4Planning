from torchmetrics import Metric
import torch
import torch.nn as nn
from torchmetrics.functional import pearson_corrcoef


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
