from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Tuple
import numpy as np

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
        x, y = row["surfaces"], row["observations"]
        if not self.sequential_cond:
            return x, y
        else:
            y = torch.Tensor(y.reshape(y.shape[0] // 2, 2, y.shape[-1]))
            for i in range(len(y)):
                if 1 not in y[i]:
                    y[i] = torch.full_like(y[i], -1)
            return x, y


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
        # return x, torch.stack(list(y))
        # return x, torch.zeros_like()
