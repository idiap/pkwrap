#!/usr/bin/env python3
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

import torch
import pkwrap

t = pkwrap.matrix.ReadKaldiMatrixFile("lfmmi_deriv_0")
print(t[0,:])
