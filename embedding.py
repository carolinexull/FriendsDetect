# -*- coding: utf-8 -*-
# @Time    : 2018/8/18 21:59
# @Author  : uhauha2929
import const as c
import json
import numpy as np

GLOVE_PATH = '/home/yzhao/data/glove/glove.6B.200d.txt'


def load_embedding_matrix(word2ix, embedding_dim):
    embedding_index = {}
    with open(GLOVE_PATH, 'rt', encoding='utf-8') as g:
        for line in g:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:])
            embedding_index[word] = vector

    print('Found {} word vectors.'.format(len(embedding_index)))
    embedding_matrix = np.zeros((len(word2ix) + 1, embedding_dim))
    for word, i in word2ix.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


if __name__ == '__main__':
    word2ix = np.load('word2ix.pkl')
    embedding_matrix = load_embedding_matrix(word2ix, c.embedding_dim)
    np.save('embedding_matrix', embedding_matrix)

