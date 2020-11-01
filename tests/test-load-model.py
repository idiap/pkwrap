#!/usr/bin/env python3

// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Amrutha Prasad <amrutha.prasad@idiap.ch>

import torch
import pkwrap

model = 'exp/chain_cleaned/tdnn_7k_1a_sp/final.mdl'

model_params = pkwrap.kaldi.nnet3.GetNNet3Components(model)
# params = {name: param for index, name, param in model_params}

params = []
for i, x in enumerate(model_params):
    # print(i, x[0], x[1][0].shape, x[1][1].shape)
    if x[1].endswith('affine'):
        if len(x[2]) != 2:
            print("Affine componeny has {} params instead of 2 params".format(len(x[1])))
            quit(1)
        # if x[0].endswith('affine'):
            #print(x[0])
        lp = x[2][0]
        bp = x[2][1].T
        param = (x[0], x[1], [lp, bp])
        params.append(param)
    if x[1].endswith('batchnorm'):
        if len(x[2]) != 2:
            print("Batchnorm component has {} paramns instead of 2 params".format(len(x[1])))
            quit(1)
        # if x[0].endswith('batchnorm'):
        scale = x[2][0].T
        offset = x[2][1].T
        param = (x[0], x[1], [scale, offset])
        params.append(param)
    if x[1].endswith('lda'):
        if len(x[2]) != 2:
            print("LDA component has {} params instead of 2 params".format(len(x[1])))
            quit(1)
        # if x[0].endswith('lda'):
        lp = x[2][0]
        bp = x[2][1].T
        param = (x[0], x[1], [lp, bp])
        params.append(param)

for (idx, name, param) in params:
    print( idx, name, param[0].shape, param[1].shape)

pkwrap.kaldi.nnet3.SaveNNet3Components(model, 'test_model.mdl', params)
print("Model read and written was successful!!")
quit(0)
