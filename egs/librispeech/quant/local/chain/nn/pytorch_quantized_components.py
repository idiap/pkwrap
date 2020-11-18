#!/usr/bin/env python3

# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Amrutha Prasad <amrutha.prasad@idiap.ch>


import torch
import torch.nn as nn
import torch.nn.functional as F

def calcScaleZeroPoint(min_val, max_val,num_bits=8):
  # Calc Scale and zero point of next 
    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale_next = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale_next
    zero_point_next = 0
    if initial_zero_point < qmin:
        zero_point_next = qmin
    elif initial_zero_point > qmax:
        zero_point_next = qmax
    else:
        zero_point_next = initial_zero_point

    zero_point_next = int(zero_point_next)

    return scale_next, zero_point_next

def calcScaleZeroPointInt8(min_val, max_val,num_bits=8):
  # Calc Scale and zero point of next 
    qmin = -127
    qmax = 127

    scale_next = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale_next
    zero_point_next = 0
    if initial_zero_point < qmin:
        zero_point_next = qmin
    elif initial_zero_point > qmax:
        zero_point_next = qmax
    else:
        zero_point_next = initial_zero_point

    zero_point_next = int(zero_point_next)

    return scale_next, zero_point_next

def quantize_tensor_int8(tensor, scale, zp):
    t = tensor/scale + zp
    t = t.round()
    t = t.to(torch.int8)
    return (t, scale, zp)

def quantize_tensor_uint8(tensor, scale, zp):
    t = tensor/scale + zp
    t = t.round()
    t = t.to(torch.uint8)
    return (t, scale, zp)

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
            self.linear_params_ = nn.Parameter(params[0], requires_grad=False)
            self.bias_ = nn.Parameter(params[1].T, requires_grad=False)
        elif params is not None and len(params) == 1:
            self.linear_params_ = nn.Parameter(params[0], requires_grad=False)
            self.bias_ = None
        else:
            self.bias_ = nn.Parameter(torch.zeros(self.output_features, 1))
            self.linear_params_ = nn.Parameter(torch.ones(self.output_features, self.input_features * lcontext))
       

    def forward(self, input, use_int8=True, input_min=None, input_max=None):
        mb, N, D = input.shape
        l = self.context.shape[0]
        expected_N = (N-l+1)
        padded_input = torch.zeros(mb, expected_N, D*l, device=input.device)
        start_d = 0
        for i,c in enumerate(self.context):
            end_d = start_d+D
            cint = int(c)
            padded_input[:,:,start_d:end_d] = input[:,i:i+expected_N,:]
            start_d = end_d
        if self.subsampling_factor>1:
            padded_input = padded_input[:,::self.subsampling_factor,:]
        
        scale, zero_point = calcScaleZeroPointInt8(self.linear_params_.data.min(), self.linear_params_.data.max())

        self.qweight = torch.quantize_per_tensor(self.linear_params_.data, scale, zero_point, torch.qint8)

        qweight_int8 = quantize_tensor_int8(self.linear_params_, scale, zero_point)
        dequant_weight = (qweight_int8[0].float() - zero_point) * scale
        # print("printing norm b/w weight and its dequantized version")
        # print((self.linear_params_ - dequant_weight).norm())
        if use_int8:
            input_scale, input_zpt = calcScaleZeroPointInt8(input_min, input_max)
            padded_input_quant = quantize_tensor_int8(padded_input, input_scale, input_zpt)
            padded_input_32b = padded_input_quant[0].to(torch.int32)
            dequant_input = (padded_input_quant[0].float() - input_zpt) * input_scale
            # print("printing norm b/w activation and its dequantized version")
            # print((padded_input - dequant_input).norm())
        else:
            # print("Using uint8 for activation quantization")
            input_scale, input_zpt = calcScaleZeroPoint(input_min, input_max)
            padded_input_quant = quantize_tensor_uint8(padded_input, input_scale, input_zpt)
            # padded_input_32b = padded_input_quant[0].to(torch.uint32)    # torch doesn't support uint32
            padded_input_32b = padded_input_quant[0].to(torch.int32)

        qweight_int32 = qweight_int8[0].to(torch.int32)
        # padded_input_32b = padded_input_quant[0].to(torch.int32)
        x1 = padded_input_32b.matmul(qweight_int32.t())
        x2 = input_zpt * qweight_int32.sum(1)
        x3 = zero_point * padded_input_32b.sum(2)
        N = self.linear_params_.shape[1]
        x = (x1 - x2).squeeze(0).t() - x3.reshape(-1)
        x = x + N * zero_point * input_zpt
        x = x.t().unsqueeze(0).float() * (input_scale * scale)
        if self.bias_ is not None and self.bias_.shape[0] != 0:
            # x += self.bias_.reshape(1, -1)
            x += self.bias_.t()
        # x = x + self.bias_.reshape(1, -1)
        return x, [dequant_weight, self.bias_]


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class TDNNTime(nn.Module):
    def __init__(self, input_features, output_features, context=[], params=[], subsampling_factor=1):
        super(TDNNTime, self).__init__()
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
            self.linear_params_ = nn.Parameter(params[0], requires_grad=False)
            self.bias_ = nn.Parameter(params[1].T, requires_grad=False)
        elif params is not None and len(params) == 1:
            self.linear_params_ = nn.Parameter(params[0], requires_grad=False)
            self.bias_ = None
        else:
            self.bias_ = nn.Parameter(torch.Tensor(1,self.output_features))
            self.linear_params_ = torch.ones(torch.Tensor(self.output_features, self.input_features))
            self.bias_.zero_()
            self.linear_params_.zero_()
       

    def forward(self, input, use_int8=True, input_min=None, input_max=None):
        mb, N, D = input.shape
        l = self.context.shape[0]
        expected_N = (N-l+1)
        padded_input = torch.zeros(mb, expected_N, D*l, device=input.device)
        start_d = 0
        for i,c in enumerate(self.context):
            end_d = start_d+D
            cint = int(c)
            padded_input[:,:,start_d:end_d] = input[:,i:i+expected_N,:]
            start_d = end_d
        if self.subsampling_factor>1:
            padded_input = padded_input[:,::self.subsampling_factor,:]
        
        scale, zero_point = calcScaleZeroPointInt8(self.linear_params_.data.min(), self.linear_params_.data.max())

        self.qweight = torch.quantize_per_tensor(self.linear_params_.data, scale, zero_point, torch.qint8)

        qweight_int8 = quantize_tensor_int8(self.linear_params_, scale, zero_point)
        dequant_weight = (qweight_int8[0].float() - zero_point) * scale
        if use_int8:
            input_scale, input_zpt = calcScaleZeroPointInt8(input_min, input_max)
            padded_input_quant = quantize_tensor_int8(padded_input, input_scale, input_zpt)
            # dequant_input = (padded_input_quant[0].float() - input_zpt) * input_scale
        else:
            input_scale, input_zpt = calcScaleZeroPoint(input_min, input_max)
            padded_input_quant = quantize_tensor_uint8(padded_input, input_scale, input_zpt)

        qweight_int32 = qweight_int8[0].to(torch.int32)
        padded_input_32b = padded_input_quant[0].to(torch.int32)
        x1 = padded_input_32b.matmul(qweight_int32.t())
        x2 = input_zpt * qweight_int32.sum(1)
        x3 = zero_point * padded_input_32b.sum(2)
        N = self.linear_params_.shape[1]
        x = (x1 - x2).squeeze(0).t() - x3.reshape(-1)
        x = x + N * zero_point * input_zpt
        x = x.t().unsqueeze(0).float() * (input_scale * scale)
        if self.bias_ is not None and self.bias_.shape[0] != 0:
            # x += self.bias_.reshape(1, -1)
            x += self.bias_.t()
        # x = x + self.bias_.reshape(1, -1)
        return x


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
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

        self.context_len = torch.tensor(context_len, requires_grad=False)
        self.quantized_params = params
        self.use_int8 = True

    def forward(self, input):
        x_min, x_max = input.min().item(), input.max().item()
        x, q_linear_params = self.linearB(input, self.use_int8, x_min, x_max)

        x_min, x_max = x.min().item(), x.max().item()
        x, q_affine_params = self.linearA(x, self.use_int8, x_min, x_max)

        self.quantized_params['linear'] = q_linear_params
        self.quantized_params['affine'] = q_affine_params

        return x


class TDNNFBatchnorm(nn.Module):
    def __init__(self, feat_dim, output_dim, bottleneck_dim, 
                linear_context=[], affine_context=[], context_len=1, subsampling_factor=1, 
                bypass_scale=0.75, tdnnf_params={}, bn_params=[]):
        super(TDNNFBatchnorm, self).__init__()

        self.tdnnf = TDNNF(feat_dim, output_dim, bottleneck_dim, linear_context=linear_context, 
                            affine_context=affine_context, params=tdnnf_params)
        self.bn = BatchNorm(feat_dim, bn_params)

        self.subsampling_factor = torch.tensor(subsampling_factor, requires_grad=False)
        self.bypass_scale = torch.tensor(bypass_scale, requires_grad=False)
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

        # print("Print in TDNNF batchnorm")
        # print(type(self.tdnnf.quantized_params['affine']))
        return x, self.tdnnf.quantized_params


class FixedAffineLayer(nn.Module):        
    def __init__(self, affine_matrix, context=[]):
        super(FixedAffineLayer, self).__init__()
        lcontext = 1
        self.context = torch.tensor(context)
        if context is not None and len(context)>0:
            lcontext = len(context)
        else:
            self.context = torch.tensor([1])
        self.affine_matrix = nn.Parameter(affine_matrix, requires_grad=False)


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
            self.scale_ = self.params[0].clone().detach()
            self.offset_ = self.params[1].clone().detach()
        else:
            self.offset_ = torch.zeros(input_size)
            self.scale_ = torch.ones(input_size)
           

    def forward(self, input):
        mb, T, D = input.shape
        assert D == self.input_size, "In BatchNorm: {} != {}".format(D, self.input_size)
        
        x = input.view(-1,D) * self.scale_
        x = x+self.offset_
        return x.view(mb,T,D)


