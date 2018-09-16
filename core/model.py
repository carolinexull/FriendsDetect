# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 16:26
# @Author  : uhauha2929
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attn import SelfAttention


class LSTM(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout=0.2, weight=None):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim, _weight=weight)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim * 2, 1)

    def forward(self, inputs):
        # [B, L]
        x = self.embedding(inputs)
        # [B, L, E]
        x = self.dropout(x)
        _, (h_n, _) = self.lstm(x)
        # [2, B, H]
        x = torch.cat((h_n[0], h_n[1]), 1)
        # [B, 2H]
        x = torch.sigmoid(self.dropout(self.fc(x)))
        return x


class LSTMATT(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout=0.2, weight=None):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim, _weight=weight)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.attention = SelfAttention(self.hidden_dim*2)
        self.fc = nn.Linear(self.hidden_dim * 2, 1)

    def forward(self, inputs):
        # [B, L]
        x = self.embedding(inputs)
        # [B, L, E]
        x = self.dropout(x)
        x, _ = self.lstm(x)
        # [B, L, 2H]
        x, _ = self.attention(x)
        # [B, 2H]
        x = torch.sigmoid(self.dropout(self.fc(x)))
        return x


class EmbedATT(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, dropout=0.2, weight=None):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim, _weight=weight)

        self.attention = SelfAttention(self.embedding_dim)
        self.fc = nn.Linear(self.embedding_dim, 1)

    def forward(self, inputs):
        # [B, L]
        x = self.embedding(inputs)
        # [B, L, E]
        x = self.dropout(x)
        x, _ = self.attention(x)
        x = torch.sigmoid(self.dropout(self.fc(x)))
        return x
