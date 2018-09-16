# -*- coding: utf-8 -*-
# @Time    : 2018/8/15 2:06
# @Author  : uhauha2929
import spacy

spacy_en = spacy.load('en')


def split_sents(text):
    """
    split a paragraph into a list of sentences
    """
    return [s.text for s in spacy_en(text).sents]


def tokenize_en(text):
    """
    tokenize English text from a string into a list of strings
    """
    return (tok.text for tok in spacy_en.tokenizer(text))
