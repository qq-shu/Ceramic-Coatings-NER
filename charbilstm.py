# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np

class CharBiLSTM(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, hidden_dim, dropout):
        super(CharBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(CharBiLSTM.random_embedding(alphabet_size, embedding_dim)))
        self.char_bilstm = nn.LSTM(embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)



    def random_embedding(vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        pretrain_emb[0, :] = np.zeros((1, embedding_dim))
        return pretrain_emb


    def forward(self, input):
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden,_=self.char_bilstm(char_embeds)
        forward=char_hidden[:,-1,:self.hidden_dim]
        backward=char_hidden[:,0,self.hidden_dim:]
        char_bilstm_out=torch.cat((forward,backward),dim=-1)
        return char_bilstm_out






