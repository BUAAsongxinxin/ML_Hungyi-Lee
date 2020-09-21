# -*- coding: utf-8 -*-
"""
@Time ： 2020-06-20 17:33
@Auth ： songxinxin
@File ：model.py
"""
import torch.nn as nn


class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, bi_direc, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # embedding
        super().__init__()
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bi = 2 if bi_direc else 1

        # lstm
        self.lstm = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bi_direc)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * self.bi, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None) # x.shape:(batch, seq_len, hidden_size)
        # 取最后一维的hidden state
        x = x[:, -1, :]
        out = self.classifier(x)
        return out
