#!/usr/bin/env python3

# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Srikanth Madikeri <srikanth.madikeri@idiap.ch>,
# Amrutha Prasad <amrutha.prasad@idiap.ch>


import torch
import torch.nn as nn
import math

qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.backends.quantized.engine = 'qnnpack'


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
        else:
            self.bias_ = nn.Parameter(torch.Tensor(1,self.output_features))
            self.linear_params_ = torch.ones(self.Tensor(self.output_features, self.input_features))
            self.bias_.zero_()
            self.linear_params_.zero_()
        self.qfn = torch.nn.quantized.QFunctional()

        self.quantized_linear = nn.quantized.Linear(self.input_features, self.output_features)
        self.quantized_relu = nn.quantized.ReLU()

    def forward(self, input, use_int8=False, input_scale=None, input_zpt=None):
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
            expected_N = math.ceil(expected_N / float(self.subsampling_factor))
            padded_input = padded_input[:,::self.subsampling_factor,:]

        scale, zero_point = calcScaleZeroPointInt8(self.linear_params_.data.min(), self.linear_params_.data.max())
        self.qweight = torch.quantize_per_tensor(self.linear_params_.data, scale, zero_point, torch.qint8)

        qweight_int8 = quantize_tensor_int8(self.linear_params_, scale, zero_point)
        dequant_weight = (qweight_int8[0].float() - zero_point) * scale

        if use_int8:
            input_scale, input_zpt = calcScaleZeroPointInt8(input.data.min(), input.data.max())
            quantized_padded_input = torch.quantize_per_tensor(padded_input, input_scale, input_zpt, torch.qint8)
        else:
            input_scale, input_zpt = calcScaleZeroPoint(input.data.min(), input.data.max())
            quantized_padded_input = torch.quantize_per_tensor(padded_input, input_scale, input_zpt, torch.quint8)
        
        self.quantized_linear.set_weight_bias(self.qweight, self.bias_[:,0])
        x = self.quantized_linear(quantized_padded_input)
        return torch.dequantize(x).view(mb,expected_N, -1 ), [dequant_weight, self.bias_]


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



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
        self.quantized_batchnorm = torch.nn.quantized.BatchNorm2d(self.input_size)
           

    def forward(self, input):
        mb, T, D = input.shape[0], input.shape[1], input.shape[2]
        # mb, T, D = input.shape
        assert D == self.input_size, "In BatchNorm: {} != {}".format(D, self.input_size)

        scale, zero_point = calcScaleZeroPointInt8(self.scale_.data.min(), self.scale_.data.max())
        self.qweight = torch.quantize_per_tensor(self.scale_.data, scale, zero_point, torch.qint8)

        input_scale, input_zpt = calcScaleZeroPoint(input.data.min(), input.data.max())
        quantized_input = torch.quantize_per_tensor(input, input_scale, input_zpt, torch.quint8)

        x = input.view(-1,D) * self.scale_
        x = x+self.offset_
        return x.view(mb,T,D)


