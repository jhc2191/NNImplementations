from aiayn.Model import Seq2Seq
import spacy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

from aiayn.Trainer import train
from aiayn.Model import Transformer

import random
import math
import time

###taken from https://github.com/bentrevett/pytorch-seq2seq/blob/rewrite/6%20-%20Attention%20is%20All%20You%20Need.ipynb
def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def main():
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
    
    SRC = Field(tokenize = tokenize_de, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

    TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),fields = (SRC, TRG))

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    BATCH_SIZE = 128

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
         batch_size = BATCH_SIZE)
    
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = Transformer(input_dim=len(SRC.vocab), output_dim=len(TRG.vocab), src_pad_idx=SRC_PAD_IDX, trg_pad_idx=TRG_PAD_IDX,
        encoding_dim=512, model_dim=512, intermediate_dim=2048, num_layers=6, num_heads=8, k_dim=64, v_dim=64, dropout=0.1, n_position=200)

    model.apply(initialize_weights)

    LEARNING_RATE = 0.0005

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    train(model, train_iterator, valid_iterator, optimizer, criterion)


    




