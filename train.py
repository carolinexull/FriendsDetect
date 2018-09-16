# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 18:45
# @Author  : uhauha2929
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import const as c
from torch.utils.data import DataLoader

from core.model import *
from data import MyDataset


def train(train_loader, model, optimizer, criterion):
    model.train()
    epoch_loss = 0
    bar = tqdm(total=len(train_loader))
    b_ix = 1
    for X, Y in train_loader:
        X = X.to(c.device)
        Y = Y.to(c.device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if b_ix % 10 == 0:
            bar.update(10)
            bar.set_description('current loss:{:.4f}'.format(epoch_loss / b_ix))
        b_ix += 1
    bar.update((b_ix - 1) % 10)
    bar.close()
    return epoch_loss / len(train_loader)


def evaluate(model, val_loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for X, Y in val_loader:
            X = X.to(c.device)
            Y = Y.to(c.device)
            outputs = model(X)
            epoch_acc += torch.sum((torch.round(outputs) == Y)).float() / Y.size(0)
            loss = criterion(outputs, Y)
            epoch_loss += loss
    model.train()
    return epoch_loss / len(val_loader), epoch_acc / len(val_loader)


def main():
    word2ix = np.load('word2ix.pkl')
    pos = np.load('friends.pkl')
    neg = np.load('strangers.pkl')
    train_data = MyDataset(pos[:9000] + neg[:9000], word2ix, max_len=c.max_len)
    val_data = MyDataset(pos[9000:] + neg[9000:], word2ix, max_len=c.max_len)
    train_loader = DataLoader(dataset=train_data, batch_size=c.batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(dataset=val_data, batch_size=c.batch_size, shuffle=True, num_workers=3)

    # weight = torch.Tensor(np.load('embedding_matrix.npy'))
    model = LSTM(len(word2ix) + 1, c.embedding_dim, c.hidden_dim, c.dropout)
    # model = LSTMATT(len(word2ix) + 1, c.embedding_dim, c.hidden_dim, c.dropout)
    # model = EmbedATT(len(word2ix) + 1, c.embedding_dim, c.hidden_dim, c.dropout)
    model.to(c.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
    criterion = nn.BCELoss().to(c.device)

    for epoch in range(c.epochs):
        train_loss = train(train_loader, model, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print('| epoch: {:02} | train Loss: {:.3f} | val Loss: {:.3f} | val acc: {:.3f}'
              .format(epoch + 1, train_loss, val_loss, val_acc))


if __name__ == '__main__':
    main()
