# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

import torch
import torch.nn as nn
import torch.nn.functional as F
from _pkwrap import kaldi
from . import chain

class NaturalAffineTransform(nn.Module):
    """Linear layer wrapped in NG-SGD

    This is an implementation of NaturalGradientAffineTransform in Kaldi.
    It wraps the linear transformation with chain.OnlineNaturalGradient to 
    achieve this.

    Attributes:
        feat_dim: (int, required) input dimension of the transformation
        out_dim: (int, required) output dimension of the transformation
        bias: (bool, optional) set False to not use bias. True by default. 
        ngstate: a dictionary containing the following keys
            alpha: a floating point value (default is 4.0)
            num_samples_history: a floating point value (default is 2000.)
            update_period: an integer (default is 4)
    """
    def __init__(self, feat_dim, out_dim, bias=True, 
                 ngstate={'alpha': 4.0, 
                          'num_samples_history': 2000.0, 
                          'update_period': 4,
                         }
                ):
        """Initialize NaturalGradientAffineTransform layer

        The function initializes NG-SGD states and parameters of the layer

        Args:
            feat_dim: (int, required) input dimension of the transformation
            out_dim: (int, required) output dimension of the transformation
            bias: (bool, optional) set False to not use bias. True by default. 
            ngstate: a dictionary containing the following keys
                alpha: a floating point value (default is 4.0)
                num_samples_history: a floating point value (default is 2000.)
                update_period: an integer (default is 4)

        Returns:
            NaturalAffineTransform object
        """
        super(NaturalAffineTransform, self).__init__()
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.preconditioner_in = kaldi.nnet3.OnlineNaturalGradient()
        self.preconditioner_in.SetAlpha(ngstate['alpha'])
        self.preconditioner_in.SetNumSamplesHistory(ngstate['num_samples_history'])
        self.preconditioner_in.SetUpdatePeriod(ngstate['update_period'])
        self.preconditioner_out = kaldi.nnet3.OnlineNaturalGradient()
        self.preconditioner_out.SetAlpha(ngstate['alpha'])
        self.preconditioner_out.SetNumSamplesHistory(ngstate['num_samples_history'])
        self.preconditioner_out.SetUpdatePeriod(ngstate['update_period'])
        self.weight = nn.Parameter(torch.Tensor(out_dim, feat_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_dim))
        else:
            self.register_parameter('bias', None)
        self.init_parameters()
    
    def init_parameters(self):
        """Initialize the parameters (weight and bias) of the layer"""

        self.weight.data.normal_()
        self.weight.data.mul_(1.0/pow(self.feat_dim*self.out_dim, 0.5))
        self.bias.data.normal_()
    
    def forward(self, input):
        """Forward pass"""
        return chain.OnlineNaturalGradient.apply(input, 
                                                 self.weight,
                                                 self.bias,
                                                 self.preconditioner_in,
                                                 self.preconditioner_out)

class TDNN(nn.Module):
    """Naive implementation of Kaldi's TDNN module

    The TDNN layer takes a context and a subsampling factor and apply a linear
    transformation to the input w.r.t the context and removes output values according
    to the subsampling factor.

    It does not use NG-SGD (will be made optional in future release) 

    Attributes:
        feat_dim (int): dimension of input features
        out_dim (int): dimension of output
        context (optional, [int]): a list of indices to use as context. Default is [0]
        subsampling_factor (optional, int): subsampling value for this layer. Default is 1 i.e. no subsampling
        linear: Pytorch's nn.Linear layer created with feat_dim \times len(context), out_dim
    """
    def __init__(self, feat_dim, out_dim, context=[0], subsampling_factor=1, bias=True):
        """Initialize TDNN module
        
        Args:
            feat_dim (int): dimension of input features
            out_dim (int): dimension of output
            context (optional, [int]): a list of indices to use as context. Default is [0]
            subsampling_factor (optional, int): subsampling value for this layer. Default is 1 i.e. no subsampling
            linear: Pytorch's nn.Linear layer created with feat_dim \times len(context), out_dim
            bias (optional, bool): Set to False if we don't want to use bias parameters. Default is True
        
        Returns:
            TDNN object
        """
        super(TDNN, self).__init__()
        self.feat_dim = torch.Tensor(feat_dim)
        self.out_dim = out_dim
        self.context = sorted(context)
        self.subsampling_factor = subsampling_factor
        assert context is not None
        assert len(context) > 0
        self.linear = nn.Linear(self.feat_dim*len(self.context), self.out_dim)

    def forward(self, input, padded=False):
        """forward pass for TDNN module

        This implementation does not use unfold, but implements the context addition. 
        Currently, there is also an implementation
        in egs/minilibrespeech that uses unfold.

        Args:
            input: Tensor input to the layer
            padded (optional, bool): if set to True, the function doesn't add context, but
                simply passes the input through the linear layer. The default value is False.
        """
        l = len(self.context)
        if padded or l == 1:
            return self.linear(input)
        else:
            assert input.shape[-1] == self.feat_dim
            mb, t, d = input.shape
            x = torch.zeros(mb, t-self.subsampling_factor+1, l, d)
            start_index = 0
            for i in range(l):
                if i == 0:
                    start_index = 0
                else:
                    start_index += self.context[i]-self.context[i-1]
                x[:, :, i, :] = input[:, start_index::self.subsampling_factor, :].view(mb, -1, 1, d)
            return self.linear(x.view(mb, t-self.subsampling_factor+1, d*l))