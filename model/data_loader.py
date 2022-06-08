import random
import os

from PIL import Image
import torch
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms


def fetch_dataloader(types, data_dir, params):
    data_all = torch.load(os.path.join(data_dir, "dataset.pt"))

    dataloaders = {"train": DataLoader(TensorDataset(data_all["x"][0:600, :], data_all["y"][0:600, None]),
                                       batch_size=params.batch_size, shuffle=True,
                                       num_workers=params.num_workers,
                                       pin_memory=params.cuda),
                   "val": DataLoader(TensorDataset(data_all["x"][600:800, :], data_all["y"][600:800, None]),
                                     batch_size=params.batch_size, shuffle=True,
                                     num_workers=params.num_workers,
                                     pin_memory=params.cuda),
                   "test": DataLoader(TensorDataset(data_all["x"][800:, :], data_all["y"][800:, None]),
                                      batch_size=params.batch_size, shuffle=True,
                                      num_workers=params.num_workers,
                                      pin_memory=params.cuda)}

    return dataloaders


