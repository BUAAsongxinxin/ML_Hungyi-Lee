import numpy as np
import matplotlib.pyplot as plt
import csv

X_train_dir = 'data/X_train'
Y_train_dir = 'data/Y_train'
X_test_dir = 'data/X_test'


def data_prop(X_train_dir, Y_train_dir, X_test_dir):
    with open(X_train_dir, 'r') as f:
        lines = f.readlines()[1:]  # 第一行为目录
        X_train = np.array([line.strip().split(',')[1:] for line in lines], dtype=float)  # 第一列为id
        f.close()

    with open(Y_train_dir, 'r') as f:
        lines = f.readlines()[1:]
        Y_train = np.array([line.strip().split(',')[1:] for line in lines], dtype=float)
        f.close()

    with open(X_test_dir, 'r') as f:
        lines = f.readlines()[1:]
        X_test = np.array([line.strip().split(',')[1:] for line in lines], dtype=float)
        f.close()

    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0)
    for j in range(X_train.shape[1]):
        X_train[:, j] = (X_train[:, j] - x_mean[j]) / (x_std[j] + 1e-8)  # 防止分母为0

    x_mean_test = np.mean(X_test, axis=0)
    x_std_test = np.std(X_test, axis=0)
    for j in range(X_test.shape[1]):
        X_test[:, j] = (X_test[:, j] - x_mean_test[j]) / (x_std_test[j] + 1e-8)

    X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)

    split_index = int(X_train.shape[0] * 0.9)
    return X_train[:split_index, :], Y_train[:split_index], X_train[split_index:, :], Y_train[split_index:], X_test


def func(X, w):
    z = np.dot(X, w)
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - 1e-8)  # 限制在某个区间内,防止出现无穷大


def cross_entropy_loss(y, y_pred):
    loss = - np.dot(y.T, np.log(y_pred)) - np.dot((1 - y).T, np.log(1 - y_pred))
    loss = int(loss[0])
    return loss / y.shape[0]


def grad(x, label, y_pred):
    error = y_pred - label
    grad = np.dot(x.T, error)
    return grad


def acc(y, y_pred):
    right_num = np.sum(y == y_pred)
    return right_num / y.shape[0]


def train(X_train, Y_train, X_dev, Y_dev):
    max_iter = 10
    batch_size = 8
    lr = 0.005
    dev_loss = []
    train_loss = []
    dev_acc = []
    train_acc = []
    batch_num = np.floor(X_train.shape[0] / batch_size)
    w = np.random.randn(X_train.shape[1], 1)

    step = 0
    for epoch in range(max_iter):
        for idx in range(int(batch_num)):
            X = X_train[idx * batch_size: (idx + 1) * batch_size, :]
            Y = Y_train[idx * batch_size: (idx + 1) * batch_size]

            y_pred = func(X, w)
            grad_w = grad(X, Y, y_pred)

            w = w - lr * grad_w

            step += 1
            # if step % 100 == 0:
        y_dev = func(X_dev, w)
        y_train = func(X_train, w)
        label_dev = np.round(y_dev)  # 预测的label
        label_train = np.round(y_train)
        dev_loss.append(cross_entropy_loss(Y_dev, y_dev))
        train_loss.append(cross_entropy_loss(Y_train, y_train))

        dev_acc.append(acc(Y_dev, label_dev))
        train_acc.append(acc(Y_train, label_train))

        print('Train_loss: {} | dev_loss: {} | train_acc: {} | dev_acc: {}'.
              format(train_loss[-1], dev_loss[-1], train_acc[-1], dev_acc[-1]))

    return w, train_loss, dev_loss, train_acc, dev_acc


def plot(train_loss, dev_loss, train_acc, dev_acc):
    plt.title('loss')
    plt.plot(train_loss)
    plt.plot(dev_loss)
    plt.legend(['train', 'dev'])
    plt.show()

    plt.title('acc')
    plt.plot(train_acc)
    plt.plot(dev_acc)
    plt.legend(['train', 'dev'])
    plt.show()


if __name__ == '__main__':
    X_train, Y_train, X_dev, Y_dev, X_test = data_prop(X_train_dir, Y_train_dir, X_test_dir)

    print('X_train.shape: {}'.format(X_train.shape))
    print('Y_train.shape: {}'.format(Y_train.shape))
    print('X_dev.shape: {}'.format(X_dev.shape))
    print('Y_dev.shape: {}'.format(Y_dev.shape))
    print('X_test.shape: {}'.format(X_test.shape))

    w, train_loss, dev_loss, train_acc, dev_acc = train(X_train, Y_train, X_dev, Y_dev)
    plot(train_loss, dev_loss, train_acc, dev_acc)

    pred_prob = func(X_test, w)
    pred_label = np.round(pred_prob)

    with open('data/submission.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'label'])
        for i, label in enumerate(pred_label):
            csv_writer.writerow([i, int(label[0])])


