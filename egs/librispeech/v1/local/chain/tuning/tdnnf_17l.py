#!/usr/bin/env python3
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

"""
    This models is like tdnn_1d in Kaldi
"""

import logging
logging.basicConfig(level=logging.DEBUG, format=f'{__name__} %(levelname)s: %(message)s')
import torch
import torch.nn.functional as F
import torch.nn as nn
from pkwrap.nn import TDNNFBatchNorm, NaturalAffineTransform, OrthonormalLinear
from pkwrap.chain import ChainModel


class Net(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super(Net, self).__init__()

        def get_tdnnf_layer(input_dim, layer_dim):
            return nn.Sequential(
                TDNNFBatchNorm(
                input_dim, 
                layer_dim, 
                context_len=3, 
                orthonormal_constraint=-1.0, 
                bypass_scale=0.75,
                bottleneck_dim=160,
                ),
                nn.Dropout(0.1)
        )
        
        self.tdnnf_layers = nn.Sequential(
                    TDNNFBatchNorm(feat_dim, 1536, context_len=3, orthonormal_constraint=-1.0, bottleneck_dim=160),
                    nn.Dropout(0.1),
                    *[get_tdnnf_layer(1536, 1536) for i in range(2, 4)],
                    TDNNFBatchNorm(1536, 1536, context_len=3, subsampling_factor=3, orthonormal_constraint=-1.0, bottleneck_dim=160),
                    nn.Dropout(0.1),
                    *[get_tdnnf_layer(1536, 1536) for i in range(5, 18)],
                    OrthonormalLinear(1536, 256, scale=-1.0),
                    nn.Dropout(0.1),
        )
        self.chain_layers = nn.Sequential(
                OrthonormalLinear(256, 1536, scale=-1.0),
                nn.Dropout(0.1),
                OrthonormalLinear(1536, 256, scale=-1.0),
                nn.Dropout(0.1),
                NaturalAffineTransform(256, output_dim),
        )
        self.chain_layers[-1].weight.data.zero_()
        self.chain_layers[-1].bias.data.zero_()
        self.xent_layers = nn.Sequential(
                OrthonormalLinear(256, 1536, scale=-1.0),
                nn.Dropout(0.1),
                OrthonormalLinear(1536, 256, scale=-1.0),
                nn.Dropout(0.1),
                NaturalAffineTransform(256, output_dim),
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
