from torch.utils.data import DataLoader
import pytorch_lightning as pl
from PIL import Image
import h5py
import json
import torch
from src.data.dataset.dataset import MyDataset, MyDataset2d
from src.utils import padding_data, merge_actions_observations, collate_fn_for_trans
import pandas as pd
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from PIL import Image
import h5py
import json
import torch
from src.data.dataset.dataset import MyDataset, MyDataset2d
from src.utils import padding_data
import pandas as pd


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path_surfaces_h5py: str = None,
        path_observations_h5py: str = None,
        path_data_json: str = None,
        batch_size: int = 256,
        num_workers: int = 4,
        pct_train: float = 0.9,
        pct_val: float = 0.08,
        pct_test: float = 0.02,
        sequential_cond=False,
        two_dimensional=False,
        shuffle_data=True,
        sequential_surfaces=False,
        random_subsequence=False,
        pad_all=False,
        use_collate_fn=False,
        dict_output=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.path_surfaces_h5py = path_surfaces_h5py
        self.path_observations_h5py = path_observations_h5py
        self.path_data_json = path_data_json
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pct_train = pct_train
        self.pct_val = pct_val
        self.pct_test = pct_test
        self.sequential_cond = sequential_cond
        self.two_dimensional = two_dimensional
        self.shuffle_data = shuffle_data
        self.sequential_surfaces = sequential_surfaces
        self.random_subsequence = random_subsequence
        self.pad_all = pad_all
        self.use_collate_fn = use_collate_fn
        self.dict_output = dict_output

    def prepare_data(self):
        if self.path_observations_h5py is not None:
            with h5py.File(self.path_surfaces_h5py, "r") as f:
                a_group_key = list(f.keys())[0]
                surfaces = list(f[a_group_key])
            with h5py.File(self.path_observations_h5py, "r") as f:
                a_group_key = list(f.keys())[0]
                observations = list(f[a_group_key])
        elif self.path_data_json is not None:
            with open(self.path_data_json, "r") as f:
                data = json.load(f)
                surfaces = data["states"]
                observations = data["observations"]
                actions = data["actions"]
                if self.sequential_cond:
                    if not self.sequential_surfaces:
                        for i in range(len(surfaces)):
                            surfaces[i] = surfaces[i][-1]
                    observations = merge_actions_observations(actions, observations)
                    if self.pad_all:
                        surfaces = padding_data(surfaces)
                        observations = padding_data(observations)
                        tmp_surfaces = torch.transpose(surfaces[0], 1, 2)
                        tmp_observations = torch.transpose(observations[0], 1, 2)
                        tmp_surfaces = tmp_surfaces.reshape(
                            tmp_surfaces.shape[0],
                            tmp_surfaces.shape[1],
                            tmp_surfaces.shape[2] * tmp_surfaces.shape[3],
                        )
                        tmp_observations = tmp_observations.reshape(
                            tmp_observations.shape[0],
                            tmp_observations.shape[1],
                            tmp_observations.shape[2] * tmp_observations.shape[3],
                        )
                        # tmp_surfaces = surfaces[0]
                        # tmp_observations = observations[0]
                        surfaces = zip(tmp_surfaces, surfaces[1])
                        observations = zip(tmp_observations, observations[1])
        df = pd.DataFrame(
            {"surfaces": list(surfaces), "observations": list(observations)}
        )
        if self.shuffle_data:
            df = df.sample(frac=1, random_state=1).reset_index(drop=True, inplace=False)
        total = len(df)

        self.train_df = df.iloc[: int(total * self.pct_train)].reset_index(
            drop=True, inplace=False
        )
        self.val_df = df.iloc[
            int(total * self.pct_train) : int(total * (self.pct_train + self.pct_val))
        ].reset_index(drop=True, inplace=False)
        if self.pct_test > 0:
            self.test_df = df.iloc[
                int(total * (self.pct_train + self.pct_val)) : int(
                    total * (self.pct_train + self.pct_val + self.pct_test)
                )
            ].reset_index(drop=True, inplace=False)

    def setup(self, stage="fit"):  # stage = fit or test or predict
        self.prepare_data()

        if stage == "fit":
            self.train_dataset = (
                MyDataset2d(self.train_df, self.sequential_cond)
                if self.two_dimensional
                else MyDataset(
                    self.train_df,
                    self.sequential_cond,
                    self.random_subsequence,
                    self.pad_all,
                    self.dict_output,
                )
            )
            self.val_dataset = (
                MyDataset2d(self.val_df, self.sequential_cond)
                if self.two_dimensional
                else MyDataset(
                    self.val_df,
                    self.sequential_cond,
                    self.random_subsequence,
                    self.pad_all,
                    self.dict_output,
                )
            )

        elif stage == "test":
            self.test_dataset = (
                MyDataset2d(self.test_df, self.sequential_cond)
                if self.two_dimensional
                else MyDataset(
                    self.test_df,
                    self.sequential_cond,
                    self.random_subsequence,
                    self.pad_all,
                    self.dict_output,
                )
            )

        elif stage == "predict":
            self.test_dataset = (
                MyDataset2d(self.test_df, self.sequential_cond)
                if self.two_dimensional
                else MyDataset(
                    self.test_df,
                    self.sequential_cond,
                    self.random_subsequence,
                    self.pad_all,
                    self.dict_output,
                )
            )

        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_for_trans if self.use_collate_fn else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_for_trans if self.use_collate_fn else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_for_trans if self.use_collate_fn else None,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_for_trans if self.use_collate_fn else None,
        )


"""class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path_surfaces_h5py: str = None,
        path_observations_h5py: str = None,
        path_data_json: str = None,
        batch_size: int = 256,
        num_workers: int = 4,
        pct_train: float = 0.9,
        pct_val: float = 0.08,
        pct_test: float = 0.02,
        sequential_cond=False,
        two_dimensional=False,
        shuffle_data=True,
        sequential_surfaces=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.path_surfaces_h5py = path_surfaces_h5py
        self.path_observations_h5py = path_observations_h5py
        self.path_data_json = path_data_json
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pct_train = pct_train
        self.pct_val = pct_val
        self.pct_test = pct_test
        self.sequential_cond = sequential_cond
        self.two_dimensional = two_dimensional
        self.shuffle_data = shuffle_data
        self.sequential_surfaces = sequential_surfaces

    def prepare_data(self):
        if self.path_observations_h5py is not None:
            with h5py.File(self.path_surfaces_h5py, "r") as f:
                a_group_key = list(f.keys())[0]
                surfaces = list(f[a_group_key])
            with h5py.File(self.path_observations_h5py, "r") as f:
                a_group_key = list(f.keys())[0]
                observations = list(f[a_group_key])
            for i in range(len(observations)):
                y = observations[i]
                x = surfaces[i]
                if np.shape(x)[2] == 64:
                    x = F.interpolate(
                        torch.Tensor(x).unsqueeze(0),
                        scale_factor=(32 / 64),
                        mode="linear",
                    ).squeeze(0)
                    y = F.interpolate(
                        torch.Tensor(y).unsqueeze(0),
                        scale_factor=(32 / 64),
                        mode="linear",
                    ).squeeze(0)
                surfaces[i] = x
                observations[i] = y

        elif self.path_data_json is not None:
            with open(self.path_data_json, "r") as f:
                data = json.load(f)
                surfaces = data["states"]
                observations = data["observations"]
                actions = data["actions"]
                if self.sequential_cond:
                    if not self.sequential_surfaces:
                        for i in range(len(surfaces)):
                            surfaces[i] = torch.Tensor(surfaces[i][-1])
                    # surfaces = surfaces[:, 0, :, :]
                    # surfaces = surfaces[:, [1], :]
                    observations = merge_actions_observations(actions, observations)
                    # surfaces = padding_data(surfaces)
                    # observations = padding_data(observations)
                    # actions = padding_data(actions)
                    # observations = torch.cat([observations, actions], dim=-2)
                    # observations = merge_actions_observations(actions, observations)

        # df = pd.DataFrame(
        #     {
        #         "surfaces": list(surfaces[0]),
        #         "surfaces_padding_masks": list(surfaces[1]),
        #         "observations": list(observations[0]),
        #         "observations_padding_masks": list(observations[1]),
        #     }
        # )
        df = pd.DataFrame(
            {
                "surfaces": list(surfaces),
                "observations": list(observations),
            }
        )
        if self.shuffle_data:
            df = df.sample(frac=1, random_state=1).reset_index(drop=True, inplace=False)
        total = len(df)

        self.train_df = df.iloc[: int(total * self.pct_train)].reset_index(
            drop=True, inplace=False
        )
        self.val_df = df.iloc[
            int(total * self.pct_train) : int(total * (self.pct_train + self.pct_val))
        ].reset_index(drop=True, inplace=False)
        if self.pct_test > 0:
            self.test_df = df.iloc[
                int(total * (self.pct_train + self.pct_val)) : int(
                    total * (self.pct_train + self.pct_val + self.pct_test)
                )
            ].reset_index(drop=True, inplace=False)

    def setup(self, stage="fit"):  # stage = fit or test or predict
        self.prepare_data()

        if stage == "fit":
            self.train_dataset = (
                MyDataset2d(self.train_df, self.sequential_cond)
                if self.two_dimensional
                else MyDataset(self.train_df, self.sequential_cond),
            )
            self.val_dataset = (
                MyDataset2d(self.val_df, self.sequential_cond)
                if self.two_dimensional
                else MyDataset(self.val_df, self.sequential_cond)
            )

        elif stage == "test":
            self.test_dataset = (
                MyDataset2d(self.test_df, self.sequential_cond)
                if self.two_dimensional
                else MyDataset(self.test_df, self.sequential_cond)
            )

        elif stage == "predict":
            self.test_dataset = (
                MyDataset2d(self.test_df, self.sequential_cond)
                if self.two_dimensional
                else MyDataset(self.test_df, self.sequential_cond)
            )

        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=collate_fn_for_trans
            # if self.path_data_json is not None
            # else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=collate_fn_for_trans
            # if self.path_data_json is not None
            # else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=collate_fn_for_trans
            # if self.path_data_json is not None
            # else None,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=collate_fn_for_trans
            # if self.path_data_json is not None
            # else None,
        )
"""
