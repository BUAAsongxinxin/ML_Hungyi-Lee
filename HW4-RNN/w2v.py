# -*- coding: utf-8 -*-
"""
@Time ： 2020-06-20 17:18
@Auth ： songxinxin
@File ：w2v.py
"""
from gensim.models import word2vec
from utils import load_test_data, load_train_data


def train_word2vec(x):
    model = word2vec.Word2Vec(x, size=200, window=5, min_count=5, iter=10, sg=1)
    return model


if __name__ == '__main__':
    print('loading training data ……')
    train_x, train_y = load_train_data('data/training_label.txt')
    print('train_x.size: {}'.format(len(train_x)))

    print('loading testing data……')
    test_x = load_test_data('data/testing_data.txt')
    print('test_x.size: {}'.format(len(test_x)))

    word2vec_model = train_word2vec(train_x + test_x)
    print('saving model ……')
    word2vec_model.save('data/w2v.model')
