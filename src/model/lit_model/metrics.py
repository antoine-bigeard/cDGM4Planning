from torchmetrics import Metric
import torch
import torch.nn as nn
from torchmetrics.functional import pearson_corrcoef


class L2Metric(Metric):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
        self.add_state("distance", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        self.distance = (
            self.distance
            + (
                torch.cdist(preds, target[0], p=self.p)
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
                preds[idx1[:, 0], :, idx1[:, 1]]
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
