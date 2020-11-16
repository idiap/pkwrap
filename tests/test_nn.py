import pytest
import torch
import pkwrap.nn as pnn
torch.no_grad()

def test_create_tdnn():
    """instantiate a TDNN"""
    pnn.TDNN(40, 100)

def test_create_tdnn_batchnorm():
    """create a TDNNBatchNorm"""
    pnn.TDNNBatchNorm(40, 100)

def test_create_tdnnf():
    """instantiate a TDNNF"""
    pnn.TDNNF(40, 100, bottleneck_dim=50)

def test_create_tdnnf_batchnorm():
    """instantiate a TDNNFBatchNorm"""
    pnn.TDNNFBatchNorm(40, 100, 50)

def test_create_orthonormal_linear():
    """instantiate a OrthonormalLinear"""
    pnn.OrthonormalLinear(40, 100)

def test_fwdpass_orthonormallinear():
    """instantiate a OrthonormalLinear and run forward pass"""
    seq_len = 150
    feat_dim = 40
    mb_size = 32
    out_dim = 100
    model = pnn.OrthonormalLinear(feat_dim, out_dim)
    x = model(torch.randn(mb_size, seq_len, feat_dim))
    assert x is not None
    assert x.shape[0] == mb_size
    assert x.shape[1] == seq_len
    assert x.shape[2] == out_dim
