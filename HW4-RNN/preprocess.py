# -*- coding: utf-8 -*-
"""
@Time ： 2020-06-20 17:18
@Auth ： songxinxin
@File ：preprocess.py.py
"""
import torch
from gensim.models import Word2Vec


class Preprocess:
    def __init__(self, seq_len, w2v_path='data/w2v.model', glove_path = 'glove/glove.6B.300d.txt' ,embed = 'glove'):
        self.seq_len = seq_len
        self.idx2word = []  # idx2word[0] = 'he'
        self.word2idx = {}  # word2idx['he'] = 0
        self.embedding_matrix = []  # sentences的embedding矩阵
        self.embedding = Word2Vec.load(w2v_path)  # 加载预训练好的词向量 embedding['word'] = vector
        self.embed = embed
        self.glove_path = glove_path

    def add_embedding(self, word):
        # 将<unk>,<pad>加入,随机初始化值即可
        vector = torch.randn(1, self.embedding.vector_size)
        self.word2idx[word] = len(self.idx2word)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self): # embedding matrix
        print('get embedding')
        if self.embed == 'w2v':
            for i, word in enumerate(self.embedding.wv.vocab):
                self.word2idx[word] = len(self.word2idx)
                self.idx2word.append(word)
                self.embedding_matrix.append(self.embedding[word])
        elif self.embed == 'glove':
            with open(self.glove_path, 'r') as f:
                lines = f.readlines()
                self.embedding.vector_size = len(lines[0].strip().split(' ')) - 1
                for line in lines:
                    word = line.strip().split(' ')[0]
                    vec = line.strip().split(' ')[1:]
                    vec_num = [float(i) for i in vec]
                    assert len(vec_num) == self.embedding.vector_size
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)
                    self.embedding_matrix.append(vec_num)
            f.close()
        else:
            pass

        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding('<PAD>')
        self.add_embedding('<UNK>')
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        if len(sentence) > self.seq_len:
            sentence = sentence[:self.seq_len]
        else:
            pad_len = self.seq_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx['<PAD>'])
        assert len(sentence) == self.seq_len
        return sentence

    def sentence_word2idx(self, sentences): # 将整个sentence的句子对应到index
        # 把句子里的词对应到index
        sentence_list = []
        for i, sen in enumerate(sentences):
            sentence_idx = []
            for i, word in enumerate(sen):
                if word in self.word2idx.keys():
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx['<UNK>'])
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.tensor(sentence_list)
