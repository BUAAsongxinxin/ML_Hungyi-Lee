# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-01 19:36
@Auth ： songxinxin
@File ：main.py
"""
from utils import *
from EN2CNDataset import *
from data_process import *
from train import *
from torch.utils.data import DataLoader


class Configurations(object):
    def __init__(self):
        self.batch_size = 32
        self.embed_dim = 300
        self.hidden_dim = 512
        self.n_layers = 2
        self.dropout = 0.5
        self.lr = 1e-4
        self.max_seq_len = 50
        self.bidirectional = True
        self.is_attn = True
        self.epochs = 30
        self.model_dir = './model'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_dir = './data'
        self.train_dir = 'data/training.txt'
        self.val_dir = 'data/validation.txt'
        self.test_dir = 'data/testing.txt'


if __name__ == '__main__':
    config = Configurations()

    # 读取数据
    en_train, cn_train = load_data(config.train_dir)
    en_val, cn_val = load_data(config.val_dir)
    en_test, cn_test = load_data(config.test_dir)
    print("length of train:{}".format(len(en_train)))
    print("length of val:{}".format(len(en_val)))
    print("length of test:{}".format(len(en_test)))

    data_process = DataProcess(config.data_dir, config.embed_dim, config.max_seq_len)
    en_vocab_size = len(data_process.int2word_en)
    cn_vocab_size = len(data_process.word2int_cn)
    train_x, train_y = data_process.seq2idx(en_train, cn_train)
    val_x, val_y = data_process.seq2idx(en_val, cn_val)
    test_x, test_y = data_process.seq2idx(en_test, cn_test)

    # 使用DataLoader装载
    train_dataset = EN2CNDataset(train_x, train_y)
    val_dataset = EN2CNDataset(val_x, val_y)
    test_dataset = EN2CNDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # model
    model, optimizer, loss_func = build_model(config, en_vocab_size, cn_vocab_size)
    train(model, optimizer, train_loader, val_loader, test_loader, loss_func, config.epochs, config.device, config.model_dir, data_process.int2word_cn, data_process.int2word_en)

    best_model = load_model(model, path=config.model_dir)
    loss_test, bleu_test, result = test(model, test_loader, loss_func, config.device, data_process.int2word_cn, data_process.int2word_en)
    print('loss_test:{:.3f} | bleu_test:{:.3f}'.format(loss_test, bleu_test))
    print(result[0])
    print(result[100])
    print(result[2333])
