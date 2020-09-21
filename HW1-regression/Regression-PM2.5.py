import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import csv

train_dir = 'data/train.csv'
test_dir = 'data/test.csv'
result_dir = 'data/submission.csv'


def read_data(file_name):
    data = pd.read_csv(file_name, encoding='big5')  # 繁体字使用big5编码
    data = data.iloc[:, 3:]  # 去掉前三列，因为前三列数据没用
    data = data.replace('NR', 0)  # NR 表示不下雨，改为0
    data = np.array(data)  # 转化为numpy格式
    return data


def pro_train(train_data):
    # 将每月的数据(18*20)*24 转化为 (18*480) 使得连续天数首尾相连
    months_data = train_data.reshape(12, -1, 24)  # 按月分割后的数据 (12*360*24)

    month_data_res = {}
    month = 0
    for month_data in months_data:
        sample = np.zeros([18, 480])
        for day in range(20):
            sample[:, day * 24:(day + 1) * 24] = month_data[18 * day:18 * (day + 1), :]  # 一天的开始取到一天的结尾
        month_data_res[month] = sample
        month += 1
    # 以步长为1取值,每月20天一共480h，一共有471组数据
    train_x = np.zeros([12 * 471, 18 * 9])
    train_y = np.zeros([12 * 471, 1])
    for month in range(12):
        for index in range(471):
            train_x[index + 471 * month, :] = month_data_res[month][:, index:index + 9].reshape(1, -1)
            train_y[index + 471 * month, :] = month_data_res[month][9, index + 9]

    # 标准化
    mean_x = np.mean(train_x, axis=0)  # 取列均值
    std_x = np.std(train_x, axis=0)
    print(mean_x.shape, std_x.shape)

    for i in range(len(train_x)):
        for j in range(len(std_x)):
            if std_x[j] != 0:
                train_x[i][j] = (train_x[i][j] - mean_x[j]) / std_x[j]
            else:
                train_x[i][j] = (train_x[i][j] - mean_x[j])

    # 转化为tensor
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    # 数据集切分
    split_index = int(train_x.shape[0] * 0.8)
    val_x = train_x[split_index:, :]
    val_y = train_y[split_index:, :]

    train_x = train_x[:split_index, :]
    train_y = train_y[:split_index, :]

    # x上添加一个维度，bias
    train_x = torch.cat([train_x, torch.ones([train_x.shape[0], 1]).double()], dim=1).float()
    val_x = torch.cat([val_x, torch.ones(val_x.shape[0], 1).double()], dim=1).float()

    return train_x, train_y, val_x, val_y


def train(train_x, train_y, val_x, val_y):
    w = torch.zeros(train_x.shape[1], 1)
    lr = 1e-5
    iter_time = 1000
    num_train = train_x.shape[0]
    num_val = val_x.shape[0]
    adagrad = torch.zeros([train_x.shape[1], 1])
    eps = 0.000001  # 防止为0

    train_loss = torch.empty(iter_time, 1)
    val_loss = torch.empty(int(iter_time / 100), 1)
    for i in range(iter_time):
        pred = torch.mm(train_x, w)
        loss = torch.sqrt(torch.sum(torch.pow(pred - train_y, 2)) / num_train) # 记得加根号，否则会导致loss太大
        train_loss[i] = loss
        if i % 100 == 0 and i != 0:
            val_pred = torch.mm(val_x, w)
            val_loss[int(i/100)] = torch.sqrt(torch.sum(torch.pow(val_pred - val_y, 2))/num_val)

        gradient = torch.mm(train_x.t(), (pred - train_y).float())
        adagrad += gradient ** 2
        #     w = w - lr * gradient / torch.sqrt(adagrad + eps)
        w = w - lr * gradient
    return w, train_loss, val_loss


def draw(train_loss, val_loss):
    val_j = list(range(0, 1000, 100)) # 1000为迭代次数
    train_j = list(range(0, 1000, 1))
    plt.figure()
    plt.plot(train_j, train_loss.data.numpy())
    plt.plot(val_j, val_loss.data.numpy(), 'r-')
    plt.show()

def pro_test(test_dir):
    test_data = pd.read_csv(test_dir, header=None)
    test_data = test_data.iloc[:, 2:]  # 去除前两行数据
    test_data = test_data.replace('NR', 0)
    test_data = np.array(test_data)

    # 拼接数据
    assert test_data.shape[0] % 18 == 0
    l = int(test_data.shape[0] / 18)
    print(l)
    test_x = np.zeros((l, 18 * 9))
    for i in range(l):  # test_x的索引
        for j in range(18):  # test_data的索引
            test_x[i, 9 * j: 9 * j + 9] = test_data[i * 18 + j]

    # 正则化
    mean_x = np.mean(test_x, axis=0)
    std_x = np.std(test_x, axis=0)

    for i in range(len(test_x)):
        for j in range(len(std_x)):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
            else:
                test_x[i][j] = (test_x[i][j] - mean_x[j])

    test_x = torch.from_numpy(test_x)
    test_X = torch.cat([test_x, torch.ones([test_x.shape[0], 1]).double()], dim=1)

    return test_X


if __name__ == '__main__':
    train_data = read_data(train_dir)

    train_x, train_y, val_x, val_y = pro_train(train_data)
    w, train_loss, val_loss = train(train_x, train_y, val_x, val_y)
    draw(train_loss, val_loss)

    test_x = pro_test(test_dir)
    pred = torch.mm(test_x, w.double())

    with open(result_dir, 'w', encoding='utf-8') as f:
        header = ['id', 'value']
        rows = []
        for index in range(len(pred)):
            if pred[index].item() < 0: # 只是为了pm2.5的实际意义
                result = 0
            else:
                result = pred[index].item()
            rows.append(['id_' + str(index), round(result)])
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

