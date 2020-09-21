# -*- coding: utf-8 -*-
"""
@Time ： 2020-06-20 17:18
@Auth ： songxinxin
@File ：utils.py.py
"""
import torch


def load_train_data(path, label):
    x = []
    y = []
    if label:
        with open(path, 'r') as f:
            for raw in f.readlines():
                raws = raw.strip('\n').split(' ')
                y.append(int(raws[0]))
                x.append(raws[2:])
        f.close()
        return x, y
    else:
        with open(path, 'r') as f:
            for raw in f.readlines():
                sent = raw.strip('\n')
                x.append(sent.split(' '))
        f.close()
        return x


def load_test_data(path):
    x = []
    with open(path, 'r') as f:
        for raw in f.readlines()[1:]:
            raws = raw.strip('\n').split(',')
            text = ','.join(raws[1:])
            x.append(text.split(' '))
    f.close()
    return x


def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0

    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct