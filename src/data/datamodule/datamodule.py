from torch.utils.data import DataLoader
import pytorch_lightning as pl
from PIL import Image
import h5py
import torch
from src.data.dataset.dataset import MyDataset, MyDataset2d
import pandas as pd
import torchvision

from src.utils import normalize_df, generate_G


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path_surfaces_h5py: str,
        path_observations_h5py: str,
        batch_size: int = 256,
        num_workers: int = 4,
        pct_train: float = 0.9,
        pct_val: float = 0.08,
        pct_test: float = 0.02,
        sequential_cond=False,
        two_dimensional=False,
        shuffle_data=True,
        gravity_data=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.path_surfaces_h5py = path_surfaces_h5py
        self.path_observations_h5py = path_observations_h5py
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pct_train = pct_train
        self.pct_val = pct_val
        self.pct_test = pct_test
        self.sequential_cond = sequential_cond
        self.two_dimensional = two_dimensional
        self.shuffle_data = shuffle_data
        self.gravity_data = gravity_data

    def prepare_data(self):
        if self.gravity_data:
            """
            G = generate_G()
            train_path = self.gravity_data["train_path"]
            val_path = self.gravity_data["val_path"]
            test_path = self.gravity_data["test_path"]
            with h5py.File(train_path, "r") as f:
                a_group_key = list(f.keys())[0]
                surfaces = list(f["x"])
                # observations = list(f["y"])
                x = torch.tensor(surfaces).flatten(start_dim=1).swapaxes(0, 1).float()
                observations = list(torch.matmul(G, x).swapaxes(0, 1))
                lines = list(f["lines"])
                df = pd.DataFrame(
                    {"surfaces": surfaces, "observations": observations, "lines": lines}
                )
                if self.shuffle_data:
                    df = df.sample(frac=1, random_state=1).reset_index(
                        drop=True, inplace=False
                    )
                self.train_df = normalize_df(df)
                # self.train_df = df
            with h5py.File(val_path, "r") as f:
                a_group_key = list(f.keys())[0]
                surfaces = list(f["x"])
                # observations = list(f["y"])
                x = torch.tensor(surfaces).flatten(start_dim=1).swapaxes(0, 1).float()
                observations = list(torch.matmul(G, x).swapaxes(0, 1))
                lines = list(f["lines"])
                df = pd.DataFrame(
                    {"surfaces": surfaces, "observations": observations, "lines": lines}
                )
                if self.shuffle_data:
                    df = df.sample(frac=1, random_state=1).reset_index(
                        drop=True, inplace=False
                    )
                self.val_df = normalize_df(df)
                # self.val_df = df
            with h5py.File(test_path, "r") as f:
                a_group_key = list(f.keys())[0]
                surfaces = list(f["x"])
                # observations = list(f["y"])
                x = torch.tensor(surfaces).flatten(start_dim=1).swapaxes(0, 1).float()
                observations = list(torch.matmul(G, x).swapaxes(0, 1))
                lines = list(f["lines"])
                df = pd.DataFrame(
                    {"surfaces": surfaces, "observations": observations, "lines": lines}
                )
                if self.shuffle_data:
                    df = df.sample(frac=1, random_state=1).reset_index(
                        drop=True, inplace=False
                    )
                self.test_df = normalize_df(df)
                # self.test_df = df
            return
            """
        with h5py.File(self.path_surfaces_h5py, "r") as f:
            a_group_key = list(f.keys())[0]
            surfaces = list(f[a_group_key])
        with h5py.File(self.path_observations_h5py, "r") as f:
            a_group_key = list(f.keys())[0]
            observations = list(f[a_group_key])
        df = pd.DataFrame({"surfaces": surfaces, "observations": observations})
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
            self.train_dataset = torchvision.datasets.MNIST(
                "files/",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            )

            self.val_dataset = torchvision.datasets.MNIST(
                "files/",
                train=False,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            )
            # self.train_dataset = (
            #     MyDataset2d(self.train_df, self.sequential_cond)
            #     if self.two_dimensional
            #     else MyDataset(self.train_df, self.sequential_cond)
            # )
            # self.val_dataset = (
            #     MyDataset2d(self.val_df, self.sequential_cond)
            #     if self.two_dimensional
            #     else MyDataset(self.val_df, self.sequential_cond)
            # )

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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
        )
