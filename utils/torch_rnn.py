import numpy as np
import torch
from torch.nn import Module, LSTM, BatchNorm1d, Linear, MaxPool1d, MaxPool2d, Identity


class HurricaneRNN(Module):

    def __init__(self, name, series_length, dropout=0.0):
        super().__init__()
        self.name = name
        self.dropout = dropout
        self.hidden_size = 256
        self.bn = BatchNorm1d(series_length)
        self.lstm1 = LSTM(input_size=2, dropout=self.dropout, hidden_size=self.hidden_size)
        self.lstm2 = LSTM(input_size=self.hidden_size, dropout=self.dropout, hidden_size=self.hidden_size)
        self.lstm3 = LSTM(input_size=self.hidden_size, dropout=self.dropout, hidden_size=self.hidden_size)
        self.dense = Linear(in_features=self.hidden_size*series_length+2*series_length, out_features=256)
        self.out = Linear(in_features=256, out_features=2)
        self.resblock = Identity()

    def forward(self, x):
        #x = self.bn(x)
        x1 = self.resblock(x)
        x, (h1, c1) = self.lstm1(x)
        #x, (h2, c2) = self.lstm2(x, (h1, c1))
        #x, (h3, c3) = self.lstm3(x, (h2, c2))
        #x=self.maxpool(x)
        x = torch.cat((x, x1), dim=2)
        x = self.dense(x.view(len(x), -1))
        x = self.out(x)
        return x
