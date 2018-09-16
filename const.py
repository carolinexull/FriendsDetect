# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 16:22
# @Author  : uhauha2929
import torch

glove_path = '/home/yzhao/data/glove/'

max_len = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 20
batch_size = 64
learning_rate = 0.0005
embedding_dim = 200
hidden_dim = 256
dropout = 0.2

