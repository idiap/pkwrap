#!/usr/bin/env python3
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

import torch
import pkwrap

x = torch.zeros(3,3).uniform_()
x = x.cuda()
pkwrap.kaldi.InstantiateKaldiCuda()
x_kaldi = pkwrap.kaldi.matrix.TensorToKaldiCuSubMatrix(x)
print("CuSubMatrix created")