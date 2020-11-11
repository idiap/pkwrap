#!/usr/bin/env python3

# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Srikanth Madikeri <srikanth.madikeri@idiap.ch>,
# Amrutha Prasad <amrutha.prasad@idiap.ch>


import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# TDNN implementation similar to the one in Kaldi where there is one big weight matrix
class TDNN(nn.Module):
    def __init__(self, input_features, output_features, context=[], params=[], subsampling_factor=1):
        super(TDNN, self).__init__()
        lcontext = 1
        self.context = torch.tensor(context)
        self.subsampling_factor = torch.tensor(subsampling_factor)
        if context is not None and len(context)>0:
            lcontext = len(context)
        else:
            self.context = torch.tensor([1])
        self.input_features = input_features
        self.output_features = output_features
        if params is not None and len(params) == 2:
            self.linear_params_ = nn.Parameter(params[0], requires_grad=True)
            self.bias_ = nn.Parameter(params[1].T, requires_grad=True)
        elif params is not None and len(params) == 1:
            self.linear_params_ = nn.Parameter(params[0], requires_grad=True)
            self.bias_ = None
        else:
            self.bias_ = nn.Parameter(torch.zeros(self.output_features, 1))
            self.linear_params_ = nn.Parameter(torch.ones(self.output_features, self.input_features * lcontext))
#            self.bias_.zero_()
#            self.linear_params_.zero_()
        

    def forward(self, input):
        mb, N, D = input.shape
        l = self.context.shape[0]
        expected_N = (N-l+1)
        padded_input = torch.zeros(mb, expected_N, D*l, device=input.device)
        start_d = 0
        for i,c in enumerate(self.context):
            end_d = start_d + D
            cint = int(c)
            padded_input[:,:,start_d:end_d] = input[:,i:i+expected_N,:]
            start_d = end_d
        if self.subsampling_factor>1:
            padded_input = padded_input[:,::self.subsampling_factor,:]

        # x = torch.mm(padded_input.view(-1, D*l), self.linear_params_.t()) 
        x = torch.mm(padded_input.reshape(-1, D*l), self.linear_params_.t()) 
        if self.bias_ is not None and self.bias_.shape[0] != 0:
            x += self.bias_.t()
        return x.view(mb, -1, self.output_features)

    def extra_repr(self):
        return 'in_features={}, out_features={}, weights={}'.format(
            self.input_features, self.output_features, self.linear_params_ 
        )

class TDNNQAT(nn.Module):
    def __init__(self, input_features, output_features, context=[], params=[], subsampling_factor=1):
        super(TDNNQAT, self).__init__()
        lcontext = 1
        self.context = torch.tensor(context)
        self.subsampling_factor = torch.tensor(subsampling_factor)
        if context is not None and len(context)>0:
            lcontext = len(context)
        else:
            self.context = torch.tensor([1])
        self.input_features = input_features
        self.output_features = output_features
        self.linear_layer = nn.Linear(self.input_features * lcontext, self.output_features)
        if params is not None and len(params) == 2:
            self.linear_params_ = nn.Parameter(params[0], requires_grad=True)
            self.bias_ = nn.Parameter(params[1].T, requires_grad=True)
            self.linear_layer.weight.data.copy_(self.linear_params_)
            self.linear_layer.bias.data.copy_(params[1].reshape(-1))
        elif params is not None and len(params) == 1:
            self.linear_params_ = nn.Parameter(params[0], requires_grad=True)
            self.bias_ = None
            self.linear_layer.weight.data.copy_(self.linear_params_)
        else:
            self.bias_ = nn.Parameter(torch.zeros(self.output_features, 1))
            self.linear_params_ = nn.Parameter(torch.ones(self.output_features, self.input_features * lcontext))
#            self.bias_.zero_()
#            self.linear_params_.zero_()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, input):
        mb, N, D = input.shape
        l = self.context.shape[0]
        expected_N = (N-l+1)
        padded_input = torch.zeros(mb, expected_N, D*l, device=input.device)
        start_d = 0
        for i,c in enumerate(self.context):
            end_d = start_d + D
            cint = int(c)
            padded_input[:,:,start_d:end_d] = input[:,i:i+expected_N,:]
            start_d = end_d
        if self.subsampling_factor>1:
            padded_input = padded_input[:,::self.subsampling_factor,:]

        padded_input = self.quant(padded_input)
        # sys.stderr.write("In TDNNQAT forward ")
        
        # sys.stderr.write(str(padded_input.dtype))
        # sys.stderr.write('\n')
        # sys.stderr.write(str(padded_input))
        # x = torch.mm(padded_input.view(-1, D*l), self.linear_params_.t()) 
        x = self.linear_layer(padded_input)
        x = F.relu(x.view(mb, -1, self.output_features))
        x = self.dequant(x)
        # return x.view(mb, -1, self.output_features)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, weights={}'.format(
            self.input_features, self.output_features, self.linear_params_ 
        )

class TDNNF(nn.Module):
    def __init__(self, feat_dim, output_dim, bottleneck_dim, 
                linear_context=[], affine_context=[], params={}):
        super(TDNNF, self).__init__()
        # lets keep it context_len for now
        # self.linearB = OrthonormalLinear(feat_dim*context_len, bottleneck_dim, scale=orthornomal_constraint)
        # self.linearA = nn.Linear(bottleneck_dim, output_dim)
        
        context_len = len(linear_context)
        self.linearB = TDNN(feat_dim*context_len, bottleneck_dim, linear_context, params['linear'])
        self.linearA = TDNN(bottleneck_dim*len(affine_context), output_dim, affine_context, params['affine'])

        self.context_len = torch.tensor(context_len, requires_grad=True)


    def forward(self, input):
        x = self.linearB(input)
        x = self.linearA(x)
        return x

class TDNNFBatchnorm(nn.Module):
    def __init__(self, feat_dim, output_dim, bottleneck_dim, 
                linear_context=[], affine_context=[], context_len=1, subsampling_factor=1, 
                bypass_scale=0.75, tdnnf_params={}, bn_params=[]):
        super(TDNNFBatchnorm, self).__init__()

        self.tdnnf = TDNNF(feat_dim, output_dim, bottleneck_dim, linear_context=linear_context, affine_context=affine_context,
                            params=tdnnf_params)
        self.bn = BatchNorm(feat_dim, bn_params)

        self.subsampling_factor = torch.tensor(subsampling_factor, requires_grad=True)
        self.bypass_scale = torch.tensor(bypass_scale, requires_grad=True)
        if bypass_scale>0. and feat_dim == output_dim:
            self.use_bypass = True
        else:
            self.use_bypass = False

    def forward(self, input):
        x = self.tdnnf(input)
        x = F.relu(x)
        x = self.bn(x)

        if self.use_bypass:
            try:
                x = x + input[:,1:x.shape[1]+1,:] * self.bypass_scale
            except:
                x = x + input[:,0:x.shape[1],:] * self.bypass_scale
        # sub-sampling factor or time-stride is 3 for layer 6. so its done here before passing input to 6th layer
        if self.subsampling_factor > 1:
            x = x[:,::self.subsampling_factor,:]

        return x

class FixedAffineLayer(nn.Module):        
    def __init__(self, feat_dim, affine_matrix=torch.Tensor([]), context=[]):
        super(FixedAffineLayer, self).__init__()
        lcontext = 1
        self.context = torch.tensor(context)
        if context is not None and len(context)>0:
            lcontext = len(context)
        else:
            self.context = torch.tensor([1])
        if affine_matrix is not None and len(affine_matrix) !=0 :
            self.affine_matrix = nn.Parameter(affine_matrix, requires_grad=True)
        else:
            self.affine_matrix = nn.Parameter(torch.zeros(feat_dim * lcontext, feat_dim * lcontext + 1))


    def forward(self, input):
        mb, N, D = input.shape
        l = self.context.shape[0]
        expected_N = N-l+1
        padded_input = torch.zeros(mb, expected_N, D*l+1, device=input.device)
        padded_input[:,:,-1] = 1.0
        start_d = 0
        for i,c in enumerate(self.context):
            end_d = start_d+D
            cint = int(c)
            padded_input[:,:,start_d:end_d] = input[:,i:i+expected_N,:]
            start_d = end_d
        x = torch.mm(padded_input.view(-1, D*l+1), self.affine_matrix.t()).view(mb,expected_N, -1)
        return x



class BatchNorm(nn.Module):
    def __init__(self, input_size, params=[]):
        super(BatchNorm, self).__init__()
        self.input_size = input_size
        self.params = params
        if self.params is not None and len(params) == 2:
            self.scale_ = nn.Parameter(self.params[0].clone().detach(), requires_grad=True)
            self.offset_ = nn.Parameter(self.params[1].clone().detach(), requires_grad=True)
        else:
            self.offset_ = nn.Parameter(torch.zeros(1, input_size), requires_grad=True)
            self.scale_ = nn.Parameter(torch.ones(1, input_size), requires_grad=True)
        

    def forward(self, input):
        mb, T, D = input.shape
        assert D == self.input_size, "In BatchNorm: {} != {}".format(D, self.input_size)
        x = input.view(-1,D) * self.scale_
        x = x+self.offset_
        return x.view(mb,T,D)


