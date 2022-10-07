import torch.nn.functional as F


def LSLoss(pred, target):
    return F.mse_loss(F.sigmoid(pred), target)
