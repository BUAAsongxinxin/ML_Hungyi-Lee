# -*- coding: utf-8 -*-
"""
@Time ： 2020-06-20 17:17
@Auth ： songxinxin
@File ：data.py
"""
from torch.utils.data import Dataset


class TwitterDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.label is not None:
            return self.data[index], self.label[index]
        else:
            return self.data[index]
