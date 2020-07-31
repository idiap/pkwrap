# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

import torch
from _pkwrap import kaldi

def add_context(feats, left_context, right_context, mode='edge'):
    """Simple function to add context to features

    This function is implemented to wrap different types of context
    addition to features. It support two modes currently: edges and zeros.

    Args:
        feats: Input features; these are pytorch Tensors
        left_context (int): number of features to be added to the left
        right_context (int): number of features to be added to the right
        mode (str): either 'edge' to add features at the edge for context or
            'zeros' to simply add zeros
    
    Returns:
        a pytorch Tensor with context added
    Raises:
        Exception if mode is unsupported
    """
    D = feats.shape[1]
    if mode == 'edge':
        left_feats = feats[0, :].repeat(left_context).reshape(-1, D)
        right_feats = feats[-1,:].repeat(right_context).reshape(-1, D)
        return torch.cat([left_feats, feats, right_feats])
    elif mode == 'zeros':
        left_feats = torch.zeros_like(feats[:, 0]).repeat(left_context).reshape(-1, D)
        right_feats = torch.zeros_like(feats[:, 0]).repeat(right_context).reshape(-1, D)
        return torch.cat([left_feats, feats, right_feats])
    else:
        raise Exception("in add_context: mode={} not supported".format(mode))
