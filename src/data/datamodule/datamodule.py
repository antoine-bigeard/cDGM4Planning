from torch.utils.data import DataLoader
import pytorch_lightning as pl
from PIL import Image
import h5py
import torch
from src.data.dataset.dataset import MyDataset, MyDataset2d
import pandas as pd


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path_surfaces_h5py: str,
        path_observations_h5py: str,
        batch_size: int = 256,
        shuffle: bool = True,
        num_workers: int = 4,
        pct_train: float = 0.8,
        pct_val: float = 0.2,
        pct_test: float = 0,
        sequential_cond=False,
        two_dimensional=False,
    ):
        super().__init__()
        self.path_surfaces_h5py = path_surfaces_h5py
        self.path_observations_h5py = path_observations_h5py
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pct_train = pct_train
        self.pct_val = pct_val
        self.pct_test = pct_test
        self.sequential_cond = sequential_cond
        self.two_dimensional = two_dimensional

        # assert self.pct_train + self.pct_val + self.pct_test == 1

    def prepare_data(self):
        with h5py.File(self.path_surfaces_h5py, "r") as f:
            a_group_key = list(f.keys())[0]
            surfaces = list(f[a_group_key])
        with h5py.File(self.path_observations_h5py, "r") as f:
            a_group_key = list(f.keys())[0]
            observations = list(f[a_group_key])
        df = pd.DataFrame({"surfaces": surfaces, "observations": observations})
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
                else MyDataset(self.train_df, self.sequential_cond)
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
            pass

        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        pass