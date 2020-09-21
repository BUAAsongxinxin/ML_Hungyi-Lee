# -*- coding: utf-8 -*-
"""
@Time ： 2020-06-30 21:23
@Auth ： songxinxin
@File ：utils.py
"""
import torch.nn
from model import *
from data_process import *
from nltk.translate.bleu_score import sentence_bleu


def load_data(path):
    en = []
    ch = []
    total_len = 0
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            seq = line.strip().split('\t')
            total_len += len(seq[0].strip().split(' '))
            total_len += len(seq[1].strip().split(' '))
            en.append(['<BOS>'] + seq[0].strip().split(' ') + ['<EOS>'])
            ch.append(['<BOS>'] + seq[1].strip().split(' ') + ['<EOS>'])
        # print("average len:{}".format(total_len / len(lines)))

    return en, ch


def save_model(model, path, step):
    print("saving the best model:epoch_{}".format(step))
    torch.save(model.state_dict(), f'{path}/best_model.ckpt')
    return


def load_model(model, path):
    print(f'loading model from {path}')
    model.load_state_dict(torch.load(f'{path}/best_model.ckpt'))
    return model


def build_model(config, en_vocab_size, cn_vocab_size):
    encoder = Encoder(en_vocab_size, config.embed_dim, config.hidden_dim, config.n_layers, config.dropout,
                      config.bidirectional)
    decoder = Decoder(cn_vocab_size, config.embed_dim, config.hidden_dim, config.n_layers, config.dropout,
                      config.is_attn)
    model = Seq2Seq(encoder, decoder, config.device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    print(optimizer)

    model = model.to(config.device)

    loss_func = torch.nn.CrossEntropyLoss()

    return model, optimizer, loss_func


def idx2sentences(outputs, int2word):
    """
    index转句子
    :param outputs: idx
    :param int2word: int2word_dict
    :return: sentences
    """
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word != '<EOS>':
                sentence.append(word)
            else:
                break
        sentences.append(sentence)

    return sentences


def computebleu(sentences, targets):
    """
    计算BLEU
    :param sentences: pred result
    :param targets: true result
    :return: score : bleu score of sentences
    """
    score = 0
    assert len(sentences) == len(targets)

    def cut_token(sent):
        tmp = []
        for token in sent:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:  # 数字，首字不是汉字
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))

    return score


if __name__ == '__main__':
    train_dir = 'data/training.txt'
    val_dir = 'data/validation.txt'
    test_dir = 'data/testing.txt'

    en_train, ch_train = load_data(train_dir)
    en_val, ch_val = load_data(val_dir)
    en_test, ch_test = load_data(test_dir)

    print("length of train:{}".format(len(en_train)))
    print("length of val:{}".format(len(en_val)))
    print("length of test:{}".format(len(en_test)))
    print(en_train[10], ch_train[10], en_val[0], ch_val[0], en_test[66], ch_test[66], sep='\n')
