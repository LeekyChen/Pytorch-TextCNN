# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:29:01 2019

@author: HSU, CHIH-CHAO

"""
import re

import pandas as pd
from numpy.random import RandomState

import torchtext
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe
from torchtext.data import Iterator, BucketIterator
import torchtext.datasets

# import spacy

def split_train_valid_test(path_data, path_train, path_valid, path_test, train_frac=0.7):
    df = pd.read_csv(path_data)
    rng = RandomState()
    tr = df.sample(frac=train_frac, random_state=rng)
    rest = df.loc[~df.index.isin(tr.index)]
    val = rest.sample(frac=0.5, random_state=rng)
    tst = rest.loc[~rest.index.isin(val.index)]

    print("Spliting original file to train/valid/test set (70:15:15)...")
    tr.to_csv(path_train, index=False)
    val.to_csv(path_valid, index=False)
    tst.to_csv(path_test, index=False)

"""
Code taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
"""
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def create_tabular_dataset(path_train, path_valid, path_test,
                          lang='en', pretrained_emb='glove.6B.300d'):
    tokenizer = torchtext.data.get_tokenizer('basic_english')

    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = Field(sequential=False, unk_token=None, pad_token=None)
    
    print('Preprocessing the text...')
    #clean the text
    TEXT.preprocessing = torchtext.data.Pipeline(clean_str)

    print('Creating tabular datasets...It might take a while to finish!')
    my_datafields = [('comment_id', None), ('text', TEXT), ('label', LABEL)]
    train_datafield = [('text', TEXT),  ('label', LABEL)]
    tabular_train = TabularDataset(path = path_train,  
                                 format= 'csv',
                                 skip_header=True,
                                 fields=my_datafields)
    
    valid_datafield = [('text', TEXT),  ('label',LABEL)]
    tabular_valid = TabularDataset(path = path_valid, 
                           format='csv',
                           skip_header=True,
                           fields=my_datafields)

    test_datafield = [('text', TEXT), ('label', LABEL)]
    tabular_test = TabularDataset(path=path_test,
                                  format='csv',
                                  skip_header=True,
                                  fields=my_datafields)
    
    print('Building vocaulary...')
    TEXT.build_vocab(tabular_train, vectors= pretrained_emb)
    LABEL.build_vocab(tabular_train)
    print("标签映射：", LABEL.vocab.stoi)

    
    return tabular_train, tabular_valid, tabular_test, TEXT.vocab

def create_data_iterator(tr_batch_size, val_batch_size,tabular_train, 
                         tabular_valid, tabular_test, d):
    #Create the Iterator for datasets (Iterator works like dataloader)
    
    train_iter = Iterator(
            tabular_train, 
            batch_size=tr_batch_size,
            device = d, 
            sort_within_batch=False,
            repeat=False)
    
    valid_iter = Iterator(
            tabular_valid, 
            batch_size=val_batch_size,
            device=d,
            sort_within_batch=False, 
            repeat=False)

    test_iter = Iterator(tabular_test,
                         batch_size=val_batch_size,
                         device=d,
                         sort_within_batch=False,
                         repeat=False)
    
    return train_iter, valid_iter, test_iter