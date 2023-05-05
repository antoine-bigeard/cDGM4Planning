from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Tuple
import numpy as np
import torch.nn.functional as F

# from utils import transform_window


class MyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sequential_cond=False,
        random_subsequence=False,
        pad_all=False,
        dict_output=False,
    ) -> None:
        self.df = df
        self.sequential_cond = sequential_cond
        self.random_subsequence = random_subsequence
        self.pad_all = pad_all
        self.dict_output = dict_output

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        row = self.df.iloc[idx, :]
        # surfaces, surfaces_padding_masks = torch.Tensor(row["surfaces"]), torch.Tensor(
        #     row["surfaces_padding_masks"]
        # )
        # observations, observations_padding_masks = torch.Tensor(
        #     row["observations"]
        # ), torch.Tensor(row["observations_padding_masks"])

        if self.pad_all:
            surfaces, surfaces_padding_masks = torch.Tensor(
                row["surfaces"][0]
            ), torch.Tensor(row["surfaces"][1])
            observations, observations_padding_masks = torch.Tensor(
                row["observations"][0]
            ), torch.Tensor(row["observations"][1])
            return (surfaces, surfaces_padding_masks), (
                observations,
                observations_padding_masks,
            )

        surfaces, observations = torch.Tensor(row["surfaces"]), torch.Tensor(
            row["observations"]
        )
        if self.dict_output:
            return {
                "surfaces": surfaces,
                "observations": observations,
            }

        if not self.random_subsequence:
            return surfaces[:, [0], :], observations

        else:
            idx = np.random.randint(1, len(surfaces) - 1)
            return surfaces, observations[:idx]

        # if not self.sequential_cond:
        # check if surfaces is a tensor
        if not isinstance(surfaces, torch.Tensor):
            raise TypeError("surfaces must be a tensor")
        # check if surfaces_padding_masks is a tensor
        if not isinstance(surfaces_padding_masks, torch.Tensor):
            raise TypeError("surfaces_padding_masks must be a tensor")
        # check if observations is a tensor
        if not isinstance(observations, torch.Tensor):
            raise TypeError("observations must be a tensor")
        # check if observations_padding_masks is a tensor
        if not isinstance(observations_padding_masks, torch.Tensor):
            raise TypeError("observations_padding_masks must be a tensor")
        # return (surfaces, surfaces_padding_masks), (
        #     observations,
        #     observations_padding_masks,
        # )
        # else:
        #     y = torch.Tensor(y.reshape(y.shape[0] // 2, 2, y.shape[-1]))
        #     for i in range(len(y)):
        #         if 1 not in y[i]:
        #             y[i] = torch.full_like(y[i], -1)
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
