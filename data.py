# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 16:29
# @Author  : uhauha2929

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, samples, word2ix, max_len=100):
        self.samples = samples
        self.word2ix = word2ix
        self.max_len = max_len

    def __getitem__(self, index):
        x = torch.zeros(self.max_len * 2, dtype=torch.int64)
        for i, word in enumerate(self.samples[index]['p1']):
            if i >= self.max_len:
                break
            if word in self.word2ix:
                x[i] = self.word2ix[word]
        for i, word in enumerate(self.samples[index]['p2']):
            if i >= self.max_len:
                break
            if word in self.word2ix:
                x[i + self.max_len] = self.word2ix[word]
        y = torch.Tensor([self.samples[index]['tag']])
        return x, y

    def __len__(self):
        return len(self.samples)
