# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:23:33 2021

@author: RKorkin
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext

from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator, TabularDataset

import torch.nn.functional as F

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional = True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 3), hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #hidden, encoder_outputs torch.Size([128, 58, 512]) torch.Size([128, 58, 1024]) torch.Size([128, 58, 1536])
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(0)
        embedded = self.dropout(self.embedding(x))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)    
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio = 0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        #print('CCCCC', outputs.shape)
        encoder_outputs, hidden = self.encoder(src)
        x = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(x, hidden, encoder_outputs)
            #print('DDDDD', output.shape, hidden.shape)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            x = (trg[t] if teacher_force else top1)
        return outputs


if __name__ == '__main__':
    from nltk.tokenize import WordPunctTokenizer
    from subword_nmt.learn_bpe import learn_bpe
    from subword_nmt.apply_bpe import BPE
    
    tokenizer_W = WordPunctTokenizer()
    def tokenize(x, tokenizer=tokenizer_W):
        return tokenizer.tokenize(x.lower())
    
    SRC = Field(tokenize=tokenize,
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)
    
    TRG = Field(tokenize=tokenize,
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)
    
    dataset = TabularDataset(
        path='data.txt',
        format='tsv',
        fields=[('trg', TRG), ('src', SRC)]
    )
    
    train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])
    SRC.build_vocab(train_data, min_freq = 3)
    TRG.build_vocab(train_data, min_freq = 3)
    
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    trg_vocab_size = OUTPUT_DIM
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    attn = Attention(HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT, attn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _len_sort_key(x):
        return len(x.src)
    
    BATCH_SIZE = 128
    
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE, 
        device = device,
        sort_key=_len_sort_key
    )
    
    for x in train_iterator:
        src = x.src
        trg = x.trg
        break
    
    
    # dont forget to put the model to the right device
    model = Seq2Seq(enc, dec, device).to(device)
    
    model.train()
        
    epoch_loss = 0
    history = []
    
    batch_size=128
    
    outputs = model(src, trg)