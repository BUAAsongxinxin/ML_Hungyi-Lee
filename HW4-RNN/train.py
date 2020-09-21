# -*- coding: utf-8 -*-
"""
@Time ： 2020-06-20 17:54
@Auth ： songxinxin
@File ：train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from utils import evaluation
import matplotlib.pyplot as plt


def train(batch_size, n_epoch, lr, model_dir, train_loader, val_loader, model, device):
    total = sum(p.numel() for p in model.parameters()) # numel 返回数组中的元素个数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("start training ----\ntotal paras: {}, trainable paras: {}".format(total, trainable))

    model.train()
    loss_func = nn.BCELoss() # 二分类 loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    for epoch in range(n_epoch):
        loss_tra = 0
        loss_val = 0
        correct_num = 0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype = torch.float) # 因为要给loss，所以是float
            out = model(inputs)
            out = out.squeeze() # out.shape[batch, 1]去掉最后一维

            optimizer.zero_grad()
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            correct = evaluation(out, labels)
            loss_tra += loss
            correct_num += correct
        train_loss.append(loss_tra / (len(train_loader) * batch_size))
        train_acc.append(correct_num / (len(train_loader) * batch_size))

        model.eval()
        with torch.no_grad():
            correct_num = 0
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype = torch.float) # 因为要给loss，所以是float
                out = model(inputs)
                out = out.squeeze()

                loss = loss_func(out, labels)

                correct = evaluation(out, labels)
                loss_val += loss
                correct_num += correct
            val_loss.append(loss_val / (len(val_loader) * batch_size))
            val_acc.append(correct_num / (len(val_loader) * batch_size))
            if val_acc[-1] > best_acc:
                best_acc = val_acc[-1]
                torch.save(model, model_dir)

            print("epoch: {} | train_loss: {:.5f}, train_acc: {:.5f} | val_loss: {:.5f}, val_acc: {:.5f}".format(
                epoch, train_loss[-1], train_acc[-1], val_loss[-1], val_acc[-1]))

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train_loss', 'val_loss'])
    plt.title('loss')
    plt.show()


def test(test_loader, model, device):
    print("testing")
    model.eval()
    with torch.no_grad():
        result = []
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()

            result.extend(outputs.tolist())

    return result
