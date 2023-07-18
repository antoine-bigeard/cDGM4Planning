from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Tuple
import numpy as np
import torch.nn.functional as F
import h5py

# from utils import transform_window


class MyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sequential_cond=False,
        random_subsequence=False,
        pad_all=False,
        dict_output=False,
        large_dataset_hdf5=False,
        path_large_hdf5=None,
    ) -> None:
        self.df = df
        self.sequential_cond = sequential_cond
        self.random_subsequence = random_subsequence
        self.pad_all = pad_all
        self.dict_output = dict_output
        self.large_dataset_hdf5 = large_dataset_hdf5
        self.path_large_hdf5 = path_large_hdf5

        if self.large_dataset_hdf5:
            self.data = h5py.File(self.path_large_hdf5, "r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        row = self.df.iloc[idx, :]

        if self.large_dataset_hdf5:
            with h5py.File(self.path_large_hdf5, "r") as f:
                surfaces = torch.Tensor(f[f"states/{row['surfaces']}"][:])
                observations = torch.cat(
                    [
                        torch.Tensor(f[f"actions/{row['observations']}"][:]),
                        torch.Tensor(f[f"observations/{row['observations']}"][:]),
                    ],
                    dim=1,
                )

        elif self.pad_all:
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

        else:
            surfaces, observations = torch.Tensor(row["surfaces"]), torch.Tensor(
                row["observations"]
            )

        if self.dict_output:
            return {
                "surfaces": surfaces,
                "observations": observations,
            }

        if not self.random_subsequence:
            return surfaces, observations

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
        dict_output=False,
        *args,
        **kwargs,
    ) -> None:
        self.df = df
        self.sequential_cond = sequential_cond
        self.dict_output = dict_output

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        row = self.df.iloc[idx, :]
        x, y = row["surfaces"], row["observations"]
        if self.dict_output:
            return {
                "surfaces": x,
                "observations": y,
            }
        return x, y
