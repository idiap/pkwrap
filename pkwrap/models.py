# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

"""
    This module will contain different types of layers useful for acoustic
    modelling. At this point, it is not used in any of the pkwrap recipes
"""

import torch
import torch.nn.functional as F

class ReluRenormLayer(torch.nn.Module):
    """An implementation of Conv1d+ReLu+BatchNorm layer

    This is a simple extension of existing Pytorch modules.
    Based on the input_dim, output_dim, kernel_size and stride
    we can have one layer of Conv1d+ReLu+BatchNorm transformation.

    Attributes:
        layer: a ModuleList of Conv1d, Relu and BatchNorm1d
    """
    def __init__(self,
            input_dim,
            output_dim,
            kernel_size=1,
            stride=1):
        """Function to initialize ReluRenormLayer

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output
            kernel_size: kernel_size of Conv1d
            stride: size of filters in Conv1d. Useful to implement subsampling_factor
        
        Returns:
            ReluRenormLayer module object
        """
        super(ReluRenormLayer, __self__).__init__()
        self.layer = torch.nn.ModuleList(
                torch.nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, stride=stride),
                torch.nn.Relu(),
                torch.nn.BatchNorm1d(output_dim, affine=False))
    def forward(self, input):
        """forward pass; simply pass the input to self.layer"""
        return self.layer(input)