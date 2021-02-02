#!/usr/bin/env python3
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Apoorv Vyas <apoorv.vyas@idiap.ch>
#             Srikanth Madikeri <srikanth.madikeri@idiap.ch>

# tg results on dev_clean
# %WER 7.78 [ 4233 / 54402, 460 ins, 502 del, 3271 sub
# after fg rescoring
# %WER 5.11 [ 2779 / 54402, 404 ins, 248 del, 2127 sub ]

import torch
import torch.nn.functional as F
import torch.nn as nn
import pkwrap
from pkwrap.nn import TDNNFBatchNorm, NaturalAffineTransform, OrthonormalLinear
from pkwrap.chain import ChainE2EModel
import numpy as np
from torch.nn.utils import clip_grad_value_
import logging
logging.basicConfig(level=logging.DEBUG)
import sys


class Net(nn.Module):

    def __init__(self,
                 feat_dim,
                 output_dim,
                 hidden_dim=1024,
                 bottleneck_dim=128,
                 prefinal_bottleneck_dim=256,
                 kernel_size_list=[3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3],
                 subsampling_factor_list=[1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1],
                 frame_subsampling_factor=3,
                 p_dropout=0.1):
        super().__init__()

        # at present, we support only frame_subsampling_factor to be 3
        assert frame_subsampling_factor == 3

        assert len(kernel_size_list) == len(subsampling_factor_list)
        num_layers = len(kernel_size_list)
        input_dim = feat_dim

        #input_dim = feat_dim * 3 + ivector_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_subsampling = frame_subsampling_factor

        # manually calculated
        self.padding = 27
        self.frame_subsampling_factor = frame_subsampling_factor

        self.tdnn1 = TDNNFBatchNorm(
            input_dim, hidden_dim,
            bottleneck_dim=bottleneck_dim,
            context_len=kernel_size_list[0],
            subsampling_factor=subsampling_factor_list[0],
            orthonormal_constraint=-1.0,
        )
        self.dropout1 = nn.Dropout(p_dropout)
        tdnnfs = []
        for i in range(1, num_layers):
            kernel_size = kernel_size_list[i]
            subsampling_factor = subsampling_factor_list[i]
            layer = TDNNFBatchNorm(
                hidden_dim,
                hidden_dim,
                bottleneck_dim=bottleneck_dim,
                context_len=kernel_size,
                subsampling_factor=subsampling_factor,
                orthonormal_constraint=-1.0,
            )
            tdnnfs.append(layer)
            dropout_layer = nn.Dropout(p_dropout)
            tdnnfs.append(dropout_layer)

        # tdnnfs requires [N, C, T]
        self.tdnnfs = nn.ModuleList(tdnnfs)

        # prefinal_l affine requires [N, C, T]
        self.prefinal_chain = TDNNFBatchNorm(
            hidden_dim, hidden_dim,
            bottleneck_dim=prefinal_bottleneck_dim,
            context_len=1,
            orthonormal_constraint=-1.0,
        )
        self.prefinal_xent = TDNNFBatchNorm(
            hidden_dim, hidden_dim,
            bottleneck_dim=prefinal_bottleneck_dim,
            context_len=1,
            orthonormal_constraint=-1.0,
        )
        self.chain_output = pkwrap.nn.NaturalAffineTransform(hidden_dim, output_dim)
        self.chain_output.weight.data.zero_()
        self.chain_output.bias.data.zero_()

        self.xent_output = pkwrap.nn.NaturalAffineTransform(hidden_dim, output_dim)
        self.xent_output.weight.data.zero_()
        self.xent_output.bias.data.zero_()
        self.validate_model()

    def validate_model(self):
        N = 1
        T = (10 * self.frame_subsampling_factor)
        #C = feat_dim * 3
        C = self.input_dim
        x = torch.arange(N * T * C).reshape(N, T, C).float()
        nnet_output, xent_output = self.forward(x)
        assert nnet_output.shape[1] == 10

    def pad_input(self, x):
        if self.padding > 0:
            N, T, C = x.shape
            left_pad = x[:,0:1,:].repeat(1,self.padding,1).reshape(N, -1, C)
            right_pad = x[:,-1,:].repeat(1,self.padding,1).reshape(N, -1, C)
            x = torch.cat([left_pad, x, right_pad], axis=1)
        return x

    def forward(self, x, dropout=0.):
        # input x is of shape: [batch_size, seq_len, feat_dim] = [N, T, C]
        assert x.ndim == 3
        x = self.pad_input(x)
        # at this point, x is [N, T, C]
        x = self.tdnn1(x)
        x = self.dropout1(x)

        # tdnnf requires input of shape [N, C, T]
        for i in range(len(self.tdnnfs)):
            x = self.tdnnfs[i](x)

        chain_prefinal_out = self.prefinal_chain(x)
        xent_prefinal_out = self.prefinal_xent(x)
        chain_out = self.chain_output(chain_prefinal_out)
        xent_out = self.xent_output(xent_prefinal_out)
        return chain_out, F.log_softmax(xent_out, dim=2)

if __name__ == '__main__':
    ChainE2EModel(Net, cmd_line=True)
