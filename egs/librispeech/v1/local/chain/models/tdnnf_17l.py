#!/usr/bin/env python3
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

# TODO: (1) move implementation of TDNN and TDNNBatchnorm, test their outputs w.r.t the old implementation of TDNN
#       this test will avoid testing the entire model
# TODO: (2) create a model class

"""
    This models is like model_1a but uses a pseudo-ivector sub-network
"""

import logging
logging.basicConfig(level=logging.DEBUG, format=f'{__name__} %(levelname)s: %(message)s')
import argparse
import os
from collections import Counter, OrderedDict
import torch
import torch.nn.functional as F
import torch.nn as nn
import pkwrap
from pkwrap.nn import TDNNFBatchNorm, NaturalAffineTransform
from pkwrap.chain import ChainModel


class Net(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super(Net, self).__init__()
        def get_tdnnf_layer(feat_dim, layer_dim):
            return TDNNFBatchNorm(
                feat_dim, 
                layer_dim, 
                context_len=3, 
                orthonormal_constraint=-1.0, 
                bypass_scale=0.75,
            )
        def get_tdnnf_layer_with_id(layer_id, feat_dim, layer_dim):
            return (f"tdnnf{layer_id}", get_tdnnf_layer(feat_dim, layer_dim))
        
        self.tdnnf_layers = nn.Sequential(
                OrderedDict([
                    get_tdnnf_layer(1, feat_dim, 1536),
                    *[get_tdnnf_layer(i, 1536, 1536) for i in range(2, 4)],
                    ["tdnn4", TDNNFBatchNorm(1536, 1536, context_len=3, subsampling_factor=3, orthonormal_constraint=-1.0)],
                    *[get_tdnnf_layer(i, 1536, 1536) for i in range(5, 18)],
                ])
        )
        self.chain_layers = nn.Sequential(
                OrderedDict([
                    ["prefinal_chain", TDNNFBatchNorm(512, 512, context_len=1, orthonormal_constraint=-1.0)],
                    ["chain_output", NaturalAffineTransform(512, output_dim)],
                ])
        )
        self.chain_layers[-1].weight.data.zero_()
        self.chain_layers[-1].bias.data.zero_()
        self.xent_layers = nn.Sequential(
                OrderedDict([
                    ["prefinal_xent", TDNNFBatchNorm(512, 512, context_len=1, orthonormal_constraint=-1.0)],
                    ["xent_output", NaturalAffineTransform(512, output_dim)],
                ])
        )
        self.xent_layers[-1].weight.data.zero_()
        self.xent_layers[-1].bias.data.zero_()
        self.output_dim = output_dim


    def forward(self, input): 
        x = self.tdnnf_layers(input)
        chain_out = self.chain_layers(x)
        xent_out = self.xent_layers(x)
        return chain_out, F.log_softmax(xent_out, dim=2)

if __name__ == '__main__':
    trainer = ChainModel(Net, cmd_line=True)
