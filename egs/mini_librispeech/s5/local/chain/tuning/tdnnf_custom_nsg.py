#!/usr/bin/env python3
# Copyright 2020    Srikant Madikeri (Idiap Research Institute)

# RESULT
# cat exp/chain/tdnnf/decode_dev_clean_2_hires_iterfinal/wer_* | fgrep WER | sort -k2,2 | head -1
# %WER 17.22 [ 3468 / 20138, 359 ins, 516 del, 2593 sub ]

"""
    simple TDNNF implementation, but no randomization used. the updates
    are done every iteration.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import pkwrap
from pkwrap.chain import ChainModel
from pkwrap.nsg import OnlineNaturalGradient

class OnlineNaturalGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, in_state, out_state, in_state_orig, out_state_orig):
        ctx.save_for_backward(input, weight, bias)
        ctx.states = [in_state, out_state, in_state_orig, out_state_orig]
        # the code below is based on pytorch's F.linear
        if input.dim() == 2 and bias is not None:
            output = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias
        return output

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output):
        """Backward pass for NG-SGD layer

        We pass the gradients computed by Pytorch to Kaldi's precondition_directions
        given the states of the layer.
        """
        input, weight, _ = ctx.saved_tensors
        in_state, out_state, in_state_orig, out_state_orig = ctx.states
        if input.dim() == 3:
            mb, T, D = input.shape
            mb_T = mb*T
        else:
            mb_T, D = input.shape
        input_temp = torch.zeros(mb_T, D+1, device=input.device, requires_grad=False)
        input_temp[:,-1] = 1.0
        input_temp[:,:-1].copy_(input.reshape(mb_T, D))
        grad_weight = grad_bias = None
        if grad_output.dim() == 3:
            grad_input = grad_output.matmul(weight)
            grad_input = grad_input.reshape(mb, T, D)
        else:
            grad_input = grad_output.mm(weight)
        input_temp_copy = input_temp.clone().detach()
        in_scale = in_state.precondition_directions(input_temp)
        # in_scale_orig = pkwrap.kaldi.nnet3.precondition_directions(in_state_orig, input_temp_copy)
        # quit(0)
        out_dim = grad_output.shape[-1]
        grad_output_temp = grad_output.view(-1, out_dim)
        out_scale = out_state.precondition_directions(grad_output_temp) # hope grad_output is continguous!
        scale = in_scale*out_scale
        grad_output.data.mul_(scale)
        # TODO: check if we should use data member instead?
        grad_weight = grad_output_temp.t().mm(input_temp[:,:-1])
        grad_bias = grad_output_temp.t().mm(input_temp[:,-1].reshape(-1,1))
        grad_weight.data.mul_(scale)
        grad_bias.data.mul_(scale)
        return grad_input, grad_weight, grad_bias.t(), None, None, None, None

class NaturalAffineTransform(nn.Module):
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
        self.preconditioner_in = OnlineNaturalGradient()
        self.preconditioner_out = OnlineNaturalGradient()
        if ngstate is None:
            ngstate = pkwrap.nn.NGState()
        self.preconditioner_in_orig = pkwrap.nn.get_preconditioner_from_ngstate(ngstate)
        self.preconditioner_out_orig = pkwrap.nn.get_preconditioner_from_ngstate(ngstate)
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
        return OnlineNaturalGradientFunction.apply(
            input, 
            self.weight,
            self.bias,
            self.preconditioner_in,
            self.preconditioner_out,
            self.preconditioner_in_orig,
            self.preconditioner_out_orig
        )

class OrthonormalLinear(NaturalAffineTransform):
    def __init__(self, feat_dim, out_dim, bias=True, scale=0.0,
                 ngstate=None,
                ):
        super(OrthonormalLinear, self).__init__(feat_dim, out_dim, bias=bias, ngstate=ngstate)
        self.scale = torch.tensor(scale, requires_grad=False)

    def forward(self, input):
        """Forward pass"""
        # do it before forward pass
        if self.training:
           with torch.no_grad():
               pkwrap.nn.constrain_orthonormal(self.weight, self.scale)
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


class Net(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super(Net, self).__init__()
        self.input_dim = feat_dim
        self.output_dim = output_dim
        self.tdnn = nn.Sequential(
            TDNNFBatchNorm(feat_dim, 512, 160, context_len=5, orthonormal_constraint=-1.0),
            TDNNFBatchNorm(512, 512, 160, context_len=3, orthonormal_constraint=-1.0),
            TDNNFBatchNorm(512, 512, 160, context_len=3, subsampling_factor=3, orthonormal_constraint=-1.0),
            TDNNFBatchNorm(512, 512, 160, context_len=3, orthonormal_constraint=-1.0),
            TDNNFBatchNorm(512, 512, 160, context_len=3, orthonormal_constraint=-1.0),
            TDNNFBatchNorm(512, 512, 160, context_len=3, orthonormal_constraint=-1.0),
        )
        self.chain = nn.Sequential(
            TDNNFBatchNorm(512, 512, 160, context_len=1, orthonormal_constraint=-1.0),
            NaturalAffineTransform(512, output_dim),
        )
        self.xent = nn.Sequential(
            TDNNFBatchNorm(512, 512, 160, context_len=1, orthonormal_constraint=-1.0),
            NaturalAffineTransform(512, output_dim),
        )
        self.chain[-1].weight.data.zero_()
        self.chain[-1].bias.data.zero_()
        self.xent[-1].weight.data.zero_()
        self.xent[-1].bias.data.zero_()

    def forward(self, input):
        mb, T, D = input.shape
        x = self.tdnn(input)
        chain_out = self.chain(x)
        xent_out = self.xent(x)
        return chain_out, F.log_softmax(xent_out, dim=2)

if __name__ == '__main__':
    ChainModel(Net, cmd_line=True)
