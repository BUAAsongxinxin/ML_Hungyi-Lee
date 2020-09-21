# -*- coding: utf-8 -*-
"""
@Time ： 2020-06-20 17:17
@Auth ： songxinxin
@File ：main.py
"""

import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import random
from train import train, test
from utils import load_train_data, load_test_data
from preprocess import *
import data
from model import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_with_label_dir = 'data/training_label_part.txt'
    train_no_label_dir = 'data/training_nolabel.txt'
    testing_data_dir = 'data/testing_data.txt'
    w2v_path = 'data/w2v.model'
    glove_path = 'glove/glove.6B.300d.txt'

    batch_size = 32
    epoch = 20
    lr = 0.0001
    seq_len = 30
    model_dir = 'ckpt.model'
    model_semi_dir = 'semi-ckpt.model'
    semi_training = True
    bi_direc = True

    # 读取数据
    train_x, y = load_train_data(train_with_label_dir, True)
    random.seed(5)
    random.shuffle(train_x)
    random.seed(5)
    random.shuffle(y)
    train_no_label = load_train_data(train_no_label_dir, False)
    test_x = load_test_data(testing_data_dir)

    # 数据预处理
    preprocess = Preprocess(seq_len, w2v_path, glove_path, embed='glove')
    embedding = preprocess.make_embedding()
    print(embedding.shape)
    train_x = preprocess.sentence_word2idx(train_x)
    train_y = torch.tensor(y)
    test_x = preprocess.sentence_word2idx(test_x)
    train_no_label_x = preprocess.sentence_word2idx(train_no_label)

    split_idx = int(len(train_x)*0.8)
    X_train, Y_train, X_val, Y_val = train_x[:split_idx], train_y[:split_idx], train_x[split_idx:], train_y[split_idx:]
    train_dataset = data.TwitterDataset(X_train, Y_train)
    val_dataset = data.TwitterDataset(X_val, Y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # 定义model
    model = LSTM_Net(embedding, embedding_dim=300, hidden_dim=300, bi_direc=bi_direc,num_layers=1, dropout=0.5, fix_embedding=True)
    model = model.to(device)

    # train
    train(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)

    # 半监督
    if semi_training:
        # 读取数据
        train_no_label_dataset = data.TwitterDataset(train_no_label_x, None)
        train_no_label_loader = DataLoader(dataset=train_no_label_dataset, batch_size=batch_size, shuffle=False)

        # 预测结果,制作新的数据集
        train_add_x = []
        train_add_y = []
        model = torch.load(model_dir)
        outputs = test(train_no_label_loader, model, device)
        num = 0
        for i in range(len(outputs)):
            if outputs[i] > 0.7:
                train_add_x.append(train_no_label_x[i].tolist())
                train_add_y.append(1)
                num += 1
            elif outputs[i] < 0.3:
                train_add_x.append(train_no_label_x[i].tolist())
                train_add_y.append(0)
                num += 1
            else:
                pass
        print('add {}/{} no_label_data'.format(num, len(train_no_label_x)))

        # 原来train-set取出一部分做验证集
        split_idx = int(len(train_x) * 0.3)
        X_train, Y_train, X_val, Y_val = train_x[:split_idx], train_y[:split_idx], train_x[split_idx:], train_y[split_idx:]

        # X_train, X_val = random_split(train_x, [train_size, val_size])
        # Y_train, Y_val = random_split(train_y, [train_size, val_size])

        X_train = torch.cat((X_train, torch.tensor(train_add_x)), dim=0)
        Y_train = torch.cat((Y_train, torch.tensor(train_add_y)), dim=0)
        print("After add no_label_data - X.shape: {}, Y.shape: {}".format(X_train.shape, Y_train.shape))

        # split_idx = int(len(train_new_x) * 0.8)
        train_dataset = data.TwitterDataset(X_train, Y_train)
        val_dataset = data.TwitterDataset(X_val, Y_val)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        # 定义model
        model = LSTM_Net(embedding, embedding_dim=300, hidden_dim=300, bi_direc=bi_direc, num_layers=1, dropout=0.5, fix_embedding=True)
        model = model.to(device)

        # train
        train(batch_size, epoch, lr, model_semi_dir, train_loader, val_loader, model, device)

    # test
    test_dataset = data.TwitterDataset(test_x, None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = torch.load(model_semi_dir)
    outputs = test(test_loader, model, device)
    for idx in range(len(outputs)):
        if outputs[idx] >= 0.5:
            outputs[idx] = 1
        else:
            outputs[idx] = 0

    pred = pd.DataFrame({'id':[str(i) for i in range(len(test_x))], "label": outputs})
    print('save csv……')
    pred.to_csv('predict.csv', index=False)

