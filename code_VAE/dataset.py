#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2023/9/3 21:42
# @Author: hanluyt

import os
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable, Tuple
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random

class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True

def build_dataset_celeba(data_path: str, img_size: int, is_train: bool=True):
    if is_train:
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.CenterCrop(148),
                                               transforms.Resize(img_size),
                                               transforms.ToTensor(), ])

        dataset = MyCelebA(
            root=data_path,
            split='train',
            transform=train_transforms,
            download=False)
    else:
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.CenterCrop(148),
                                             transforms.Resize(img_size),
                                             transforms.ToTensor(), ])

        dataset = MyCelebA(
            root=data_path,
            split='test',
            transform=val_transforms,
            download=False)

    return dataset


def split_dataset(directory: str, random_state:int, num_part=5):
    """ split into train and validation dataset： 5-fold cv"""

    random.seed(random_state)
    lst = os.listdir(directory)
    random.shuffle(lst)

    return [lst[i::num_part] for i in range(num_part)]


class IMAGEN_contrast(Dataset):
    def __init__(self, directory, cv_name, show_path=False):
        self.transform = transforms.ToTensor()
        self.show_path = show_path
        img_list = []

        for filename in cv_name:
            fullname = os.path.join(directory, filename)
            img_list.append(fullname)

        self.image_list = img_list

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        while True:
            try:
                img_path = self.image_list[index]
                img = np.load(img_path)
                img = img.astype(np.float32)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.image_list) - 1)

            img = self.transform(img)
        if self.show_path:
            return img, img_path
        else:
            return img

    def __len__(self) -> int:
        return len(self.image_list)





