from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Tuple
import os
import numpy as np
import time
from src.utils import get_idx_val

# from utils import transform_window


class MyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sequential_cond=False,
    ) -> None:
        self.df = df
        self.sequential_cond = sequential_cond

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        row = self.df.iloc[idx, :]
        # x, y = torch.tensor(row["surfaces"]), torch.tensor(row["observations"])
        x, y = row["surfaces"], row["observations"]
        if not self.sequential_cond:
            return x, y
        else:
            # if y.dim() == 2:
            y = torch.Tensor(y.reshape(y.shape[0] // 2, 2, y.shape[-1]))
            for i in range(len(y)):
                if 1 not in y[i]:
                    y[i] = torch.full_like(y[i], -1)
            return x, y
            # if y.dim() == 3:
            #     y = torch.Tensor(y.reshape(y.shape[0], y.shape[1] // 2, 2, y.shape[-1]))
            #     for i in range(len(y)):
            #         for j in range(len(y[i])):
            #             if 1 not in y[i, j]:
            #                 y[i, j] = torch.full_like(y[i, j], -1)
            #     return x, y


class MyDataset2d(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sequential_cond=False,
    ) -> None:
        self.df = df
        self.sequential_cond = sequential_cond

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        row = self.df.iloc[idx, :]
        x, y = row["surfaces"], row["observations"]
        return x, y
