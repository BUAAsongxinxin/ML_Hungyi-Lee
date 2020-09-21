# -*- coding: utf-8 -*-
"""
@Time ： 2020-06-30 21:36
@Auth ： songxinxin
@File ：data_process.py
"""

import os
import json
import torch


class DataProcess:
    def __init__(self, root, embed_dim, max_seq_len):
        self.root = root
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en, self.int2word_en = self.get_dictionary('en')
        # self.embedding_cn, self.embedding_en = self.init_embedding()

    def get_dictionary(self, language):
        word2int_dir = os.path.join(self.root, f'word2int_{language}.json')
        int2word_dir = os.path.join(self.root, f'int2word_{language}.json')
        with open(word2int_dir, 'r') as f:
            word2int = json.load(f)
        with open(int2word_dir, 'r') as f:
            int2word = json.load(f)

        return word2int, int2word

    # def init_embedding(self):
    #     length_cn = len(self.word2int_cn)
    #     length_en = len(self.word2int_en)
    #     embedding_cn = torch.empty((length_cn, self.embed_dim))
    #     embedding_en = torch.empty((length_en, self.embed_dim))
    #
    #     self.embedding_cn = torch.nn.init.xavier_normal_(embedding_cn, gain=1)
    #     self.embedding_en = torch.nn.init.xavier_normal_(embedding_en, gain=1)
    #
    #     return self.embedding_cn, self.embedding_en

    def pad_sequence(self, seq):
        if len(seq) < self.max_seq_len:
            num_pad = self.max_seq_len - len(seq)
            for _ in range(num_pad):
                seq.append('<PAD>')
        else:
            seq = seq[:self.max_seq_len]

        assert len(seq) == self.max_seq_len
        return seq

    def seq2idx(self, sentences_en, sentences_cn):
        sentences_idx_cn = []
        sentences_idx_en = []
        for seq in sentences_cn:
            seq = self.pad_sequence(seq)
            seq_idx = [self.word2int_cn[word] if word in self.word2int_cn else self.word2int_cn['<UNK>'] for word in seq]
            sentences_idx_cn.append(seq_idx)
        for seq in sentences_en:
            seq = self.pad_sequence(seq)
            seq_idx = [self.word2int_en[word] if word in self.word2int_en else self.word2int_en['<UNK>'] for word in seq]
            sentences_idx_en.append(seq_idx)

        return torch.tensor(sentences_idx_en), torch.tensor(sentences_idx_cn)


if __name__ == '__main__':
    process = DataProcess('data', 200, 20)
    embedding_cn, embedding_en = process.init_embedding()
    print(embedding_en.shape)
    print(embedding_cn[0])
    print(embedding_en[0])
