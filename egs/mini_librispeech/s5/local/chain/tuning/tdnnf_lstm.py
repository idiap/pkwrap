#!/usr/bin/env python3
# Copyright 2020    Srikant Madikeri (Idiap Research Institute)

"""
    simple TDNNF-LSTM implementation
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import pkwrap
from pkwrap.nn import TDNNFBatchNorm, NaturalAffineTransform
from pkwrap.chain import ChainModel


class Net(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super(Net, self).__init__()
        self.input_dim = feat_dim
        self.output_dim = output_dim
        self.tdnn = nn.Sequential(
            TDNNFBatchNorm(feat_dim, 512, 160, context_len=5, orthornomal_constraint=-1.0),
            TDNNFBatchNorm(512, 512, 160, context_len=3, orthornomal_constraint=-1.0),
            TDNNFBatchNorm(512, 512, 160, context_len=3, subsampling_factor=3, orthornomal_constraint=-1.0),
            TDNNFBatchNorm(512, 512, 160, context_len=3, orthornomal_constraint=-1.0),
            TDNNFBatchNorm(512, 512, 160, context_len=3, orthornomal_constraint=-1.0),
            TDNNFBatchNorm(512, 512, 160, context_len=3, orthornomal_constraint=-1.0),
        )
        self.lstm = nn.LSTM(512, 256, 2, batch_first=True)
        self.chain = nn.Sequential(
            TDNNFBatchNorm(256, 256, 160, context_len=1, orthornomal_constraint=-1.0),
            NaturalAffineTransform(256, output_dim),
        )
        self.xent = nn.Sequential(
            TDNNFBatchNorm(256, 256, 160, context_len=1, orthornomal_constraint=-1.0),
            NaturalAffineTransform(256, output_dim),
        )
        self.chain[-1].weight.data.zero_()
        self.chain[-1].bias.data.zero_()
        self.xent[-1].weight.data.zero_()
        self.xent[-1].bias.data.zero_()

    def forward(self, input):
        mb, T, D = input.shape
        x = self.tdnn(input)
        x, _ = self.lstm(x)
        x = F.relu(x)
        chain_out = self.chain(x)
        xent_out = self.xent(x)
        return chain_out, F.log_softmax(xent_out, dim=2)

if __name__ == '__main__':
    ChainModel(Net, cmd_line=True)

