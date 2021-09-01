"""
BiMogrifierLSTM
"""


import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import *
from enum import IntEnum


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class BiMogLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, mog_iteration):
        super(BiMogLSTM, self).__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.mog_iterations = mog_iteration


        self.Wih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.Whh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bih = Parameter(torch.Tensor(hidden_sz * 4))
        self.bhh = Parameter(torch.Tensor(hidden_sz * 4))


        self.Wih_r = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.Whh_r = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bih_r = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.bhh_r = nn.Parameter(torch.Tensor(hidden_sz * 4))



        self.Q = Parameter(torch.Tensor(hidden_sz, input_sz))
        self.R = Parameter(torch.Tensor(input_sz, hidden_sz))

        self.init_weights()



    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)



    def mogrify(self, xt, ht):
        for i in range(1, self.mog_iterations + 1):
            if (i % 2 == 0):
                ht = (2 * torch.sigmoid(xt @ self.R) * ht)
            else:
                xt = (2 * torch.sigmoid(ht @ self.Q) * xt)
        return xt, ht


    def forward(self, x: torch.Tensor, init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            ht = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
        else:
            ht, Ct = init_states


        for t in range(seq_sz):
            xt = x[:, t, :]
            xt, ht = self.mogrify(xt, ht)
            gates = (xt @ self.Wih + self.bih) + (ht @ self.Whh + self.bhh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ft = torch.sigmoid(forgetgate)
            it = torch.sigmoid(ingate)
            Ct_candidate = torch.tanh(cellgate)
            ot = torch.sigmoid(outgate)

            Ct = (ft * Ct) + (it * Ct_candidate)
            ht = ot * torch.tanh(Ct)
            hidden_seq.append(ht.unsqueeze(Dim.batch))



        if init_states is None:
            ht_r = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
            Ct_r = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
        else:
            ht_r, Ct_r = init_states
        for t in range(seq_sz - 1, -1, -1):
            xt_r = x[:, t, :]
            xt_r, ht_r = self.mogrify(xt_r, ht_r)
            gates_r = (xt_r @ self.Wih_r + self.bih_r) + (ht_r @ self.Whh_r + self.bhh_r)
            ingate_r, forgetgate_r, cellgate_r, outgate_r = gates_r.chunk(4, 1)


            ft_r = torch.sigmoid(forgetgate_r)
            it_r = torch.sigmoid(ingate_r)
            Ct_candidate_r = torch.tanh(cellgate_r)
            ot_r = torch.sigmoid(outgate_r)

            Ct_r = (ft_r * Ct_r) + (it_r * Ct_candidate_r)
            ht_r = ot_r * torch.tanh(Ct_r)

            hidden_seq[t] = torch.cat((hidden_seq[t], ht_r.unsqueeze(Dim.batch)), axis=2)


        h = torch.stack((ht, ht_r))
        C = torch.stack((Ct, Ct_r))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)


        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h, C)



