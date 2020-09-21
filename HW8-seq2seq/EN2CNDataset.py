# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-01 10:59
@Auth ： songxinxin
@File ：EN2CNDataset.py
"""

from torch.utils.data import Dataset
import torch


class EN2CNDataset(Dataset):
    def __init__(self, en, cn):
        self.cn = cn
        self.en = en

    def __len__(self):
        return len(self.en)

    def __getitem__(self, item):
        return torch.tensor(self.en[item]), torch.tensor(self.cn[item])
