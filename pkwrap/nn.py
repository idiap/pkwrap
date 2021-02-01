""" Neural network architectures and relevant utility functions"""
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

import torch
import torch.nn as nn
import torch.nn.functional as F
from _pkwrap import kaldi
from . import chain
from dataclasses import dataclass

@dataclass
class NGState:
    """NGState value container"""
    alpha: float = 4.0
    num_samples_history: float = 2000.0
    update_period: int = 4

def get_preconditioner_from_ngstate(ngstate):
    assert ngstate is not None
    preconditioner = kaldi.nnet3.OnlineNaturalGradient()
    preconditioner.SetAlpha(ngstate.alpha)
    preconditioner.SetNumSamplesHistory(ngstate.num_samples_history)
    preconditioner.SetUpdatePeriod(ngstate.update_period)
    return preconditioner

class NaturalAffineTransform(nn.Module):
    """Linear layer wrapped in NG-SGD

    This is an implementation of NaturalGradientAffineTransform in Kaldi.
    It wraps the linear transformation with chain.OnlineNaturalGradient to
    achieve this.

    Attributes:
        feat_dim: (int, required) input dimension of the transformation
        out_dim: (int, required) output dimension of the transformation
        bias: (bool, optional) set False to not use bias. True by default.
        ngstate: a dataclass of type NGState containing the following keys
            alpha: a floating point value (default is 4.0)
            num_samples_history: a floating point value (default is 2000.)
            update_period: an integer (default is 4)
    """
    def __init__(
            self,
            feat_dim,
            out_dim,
            bias=True,
            ngstate=None,
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
        if ngstate is None:
            ngstate = NGState()
        self.preconditioner_in = get_preconditioner_from_ngstate(ngstate)
        self.preconditioner_out = get_preconditioner_from_ngstate(ngstate)
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
        return chain.OnlineNaturalGradient.apply(
            input, 
            self.weight,
            self.bias,
            self.preconditioner_in,
            self.preconditioner_out
        )

class TDNN(nn.Module):
    """Naive implementation of Kaldi's TDNN module

    WARNING: changed implementation. context parameter is now context_len

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
    def __init__(self, feat_dim, output_dim, context_len=1, subsampling_factor=1):
        """Initialize TDNN module
        
        Args:
            feat_dim (int): dimension of input features
            out_dim (int): dimension of output
            context (optional): length of context to be used. Default is 1 i.e. no context
            subsampling_factor (optional, int): subsampling value for this layer. Default is 1 i.e. no subsampling
            linear: Pytorch's nn.Linear layer created with feat_dim \times len(context), out_dim
            bias (optional, bool): Set to False if we don't want to use bias parameters. Default is True
        
        Returns:
            TDNN object
        """
        super(TDNN, self).__init__()
        self.linear = NaturalAffineTransform(feat_dim*context_len, output_dim)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)
        self.feat_dim = torch.tensor(feat_dim, requires_grad=False)
        self.subsampling_factor = torch.tensor(subsampling_factor, requires_grad=False)
        self.context_len = torch.tensor(context_len, requires_grad=False)

    def forward(self, input):
        """forward pass for TDNN module

        This implementation does not use unfold, but implements the context addition. 
        Currently, there is also an implementation
        in egs/minilibrespeech that uses unfold.

        Args:
            input: Tensor input to the layer
            padded (optional, bool): if set to True, the function doesn't add context, but
                simply passes the input through the linear layer. The default value is False.
        """
        mb, T, D = input.shape
        l = self.context_len
        N = T-l+1
        padded_input = torch.zeros(mb, N, D*self.context_len, device=input.device)
        start_d = 0
        for i in range(l):
            end_d = start_d + D
            padded_input[:,:,start_d:end_d] = input[:,i:i+N,:]
            start_d = end_d
        if self.subsampling_factor>1:
            padded_input = padded_input[:,::self.subsampling_factor,:]
        return self.linear(padded_input)

class TDNNBatchNorm(nn.Module):
    def __init__(self, feat_dim, output_dim, context_len=1, subsampling_factor=1):
        super(TDNNBatchNorm, self).__init__()
        self.tdnn = TDNN(feat_dim, output_dim, context_len, subsampling_factor)
        self.bn = nn.BatchNorm1d(output_dim, affine=False)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)

    def forward(self, input):
        mb, T, D = input.shape
        x = self.tdnn(input)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        return x

@torch.no_grad()
def constrain_orthonormal(M, scale, update_speed=0.125):
    rows, cols = M.shape
    d = rows
    if rows < cols:
        M = M.T
        d = cols
    # we don't update it. we just compute the gradient
    P = M.mm(M.T)

    if scale < 0.:
        trace_P_Pt = P.pow(2.0).sum()
        trace_P = P.trace()
        ratio = trace_P_Pt/trace_P
        scale = ratio.sqrt()
        ratio = ratio * d / trace_P
        if ratio > 1.1:
            update_speed *= 0.25
        elif ratio > 1.02:
            update_speed *= 0.5
    scale2 = scale**2
    P[range(d), range(d)] -= scale2
    M.data.add_(P.mm(M), alpha=-4*update_speed/scale2)

class OrthonormalLinear(NaturalAffineTransform):
    def __init__(self, feat_dim, out_dim, bias=True, scale=0.0,
                 ngstate=NGState(),
                ):
        super(OrthonormalLinear, self).__init__(feat_dim, out_dim, bias=bias, ngstate=ngstate)
        self.scale = torch.tensor(scale, requires_grad=False)

    def forward(self, input):
        """Forward pass"""
        # do it before forward pass
        if self.training:
           with torch.no_grad():
               constrain_orthonormal(self.weight, self.scale)
        x = super().forward(input)
        return x

class TDNNF(nn.Module):
    def __init__(
        self,
        feat_dim,
        output_dim,
        bottleneck_dim,
        context_len=1,
        subsampling_factor=1,
        orthonormal_constraint=0.0,
        floating_scale=True,
        bypass_scale=0.66):
        super(TDNNF, self).__init__()
        # lets keep it context_len for now
        self.linearB = OrthonormalLinear(feat_dim*context_len, bottleneck_dim, scale=orthonormal_constraint)
        self.linearA = nn.Linear(bottleneck_dim, output_dim)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)
        self.bottleneck_dim = torch.tensor(bottleneck_dim, requires_grad=False)
        self.feat_dim = torch.tensor(feat_dim, requires_grad=False)
        self.subsampling_factor = torch.tensor(subsampling_factor, requires_grad=False)
        self.context_len = torch.tensor(context_len, requires_grad=False)
        self.orthonormal_constraint = torch.tensor(orthonormal_constraint, requires_grad=False)
        self.bypass_scale = torch.tensor(bypass_scale, requires_grad=False)
        if bypass_scale>0. and feat_dim == output_dim:
            self.use_bypass = True
            if self.context_len > 1:
                if self.context_len%2 == 1:
                    self.identity_lidx = self.context_len//2
                    self.identity_ridx = -self.identity_lidx
                else:
                    self.identity_lidx = self.context_len//2
                    self.identity_ridx = -self.identity_lidx+1
            else:
                self.use_bypass = False
        else:
            self.use_bypass = False


    def forward(self, input):
        mb, T, D = input.shape
        padded_input = input.reshape(mb, -1).unfold(1, D*self.context_len, D*self.subsampling_factor).contiguous()
        x = self.linearB(padded_input)
        x = self.linearA(x)
        if self.use_bypass:
            x = x + input[:,self.identity_lidx:self.identity_ridx:self.subsampling_factor,:]*self.bypass_scale
        return x

class TDNNFBatchNorm(nn.Module):
    def __init__(
        self, 
        feat_dim, 
        output_dim, 
        bottleneck_dim,
        context_len=1,
        subsampling_factor=1,
        orthonormal_constraint=0.0,
        bypass_scale=0.66
    ):
        super(TDNNFBatchNorm, self).__init__()
        self.tdnn = TDNNF(
            feat_dim,
            output_dim,
            bottleneck_dim,
            context_len=context_len,
            subsampling_factor=subsampling_factor,
            orthonormal_constraint=orthonormal_constraint,
            bypass_scale=bypass_scale,
        )
        self.bn = nn.BatchNorm1d(output_dim, affine=False)
        self.output_dim = torch.tensor(output_dim, requires_grad=False)

    def forward(self, input):
        mb, T, D = input.shape
        x = self.tdnn(input)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = F.relu(x)
        return x

