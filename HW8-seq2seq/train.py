# -*- coding: utf-8 -*-
"""
@Time ： 2020-07-01 17:21
@Auth ： songxinxin
@File ：train.py.py
"""
import copy
import torch.nn
from utils import *


def train(model, optimizer, train_loader, val_loader, test_loader, loss_function, epochs, device, model_path, int2word_dict_cn, int2word_dict_en):
    model.train()
    # model.zero_grad()
    train_losses = []
    val_losses = []
    bleu_scores = []
    best_bleu_score = 0.0
    # best_loss, bleu_score , _ = test(model, val_loader, loss_function, device, int2word_dict_cn, int2word_dict_en)
    # print("best_loss/bleu score from saved model is {:.3f}/{:.3f}".format(best_loss, bleu_score))

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batch_size = 32
        for i, (sources, targets) in enumerate(train_loader):
            batch_size = sources.shape[0]
            sources = sources.to(device)
            targets = targets.to(device)

            outputs, preds = model(sources, targets, schedual_sampling(epoch, epochs, 'train'), 'train')
            outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
            targets = targets[:, 1:].reshape(-1)
            loss = loss_function(outputs, targets)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # clip 防止梯度爆炸
            optimizer.step()

        train_losses.append(train_loss / len(train_loader) * batch_size)
        model.eval()
        with torch.no_grad():
            val_loss, bleu_score, result = test(model, val_loader, loss_function, device, int2word_dict_cn, int2word_dict_en)
            if bleu_score > best_bleu_score:
                best_bleu_score = bleu_score
                save_model(model, path=model_path, step=epoch)
                best_model = copy.deepcopy(model)
            val_losses.append(val_loss)
            bleu_scores.append(bleu_score)
            test_loss, test_bleu, _ = test(model, test_loader, loss_function, device, int2word_dict_cn, int2word_dict_en)
        print("epoch: {}/{} | train_loss: {:.3f} | val_loss: {:.3f} val_bleu: {:.3f}  \
        | test_loss: {:.3f} test_bleu: {:.3f}".format(epoch, epochs, train_losses[-1], val_loss, bleu_score, test_loss, test_bleu))
    return best_model


def test(model, dataloader, loss_func, device, int2word_dict_cn, int2word_dict_en):
    model.eval()
    loss_sum = 0
    bleu_score = 0.0
    num = 0
    result = []
    for i, (sources, targets) in enumerate(dataloader):
        num += sources.size(0)  # batch_size
        sources = sources.to(device)
        targets = targets.to(device)

        outputs, preds = model(sources, targets, schedual_sampling(1, 1, 'test'), 'test')
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        loss = loss_func(outputs, targets)
        loss_sum += loss.item()

        # 将预测结果转化为文字
        targets = targets.view(sources.size(0), -1)
        preds = idx2sentences(preds, int2word_dict_cn)
        sources = idx2sentences(sources, int2word_dict_en)
        targets = idx2sentences(targets, int2word_dict_cn)

        for idx in range(len(targets)):
            result.append((sources[idx], preds[idx], targets[idx]))

        bleu_score += computebleu(preds, targets)

    return loss_sum / len(dataloader), bleu_score / num, result


def schedual_sampling(epoch, epochs, mode):
    if mode == 'train':
        return epoch / epochs
    else:
        return 1
