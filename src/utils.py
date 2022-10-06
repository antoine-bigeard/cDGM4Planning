import yaml
import numpy as np
import pandas as pd
import torch
from PIL import Image
from random import randrange


def read_yaml_config_file(path_config: str):
    with open(path_config) as conf:
        return yaml.load(conf, yaml.FullLoader)


def random_observation(shape, return_1_idx=False):
    y = torch.ones(shape[0], 2, shape[2])
    y_ones_idx = [randrange(start=0, stop=shape[2]) for i in range(shape[0])]
    for i in range(shape[0]):
        single_one = torch.tensor(
            [1 if j == y_ones_idx[i] else 0 for j in range(shape[2])]
        )
        y[i, 0, :] = single_one
        y[i, 1, :] = torch.rand(shape[2]) * single_one
    if return_1_idx:
        return y, y_ones_idx
    return y
