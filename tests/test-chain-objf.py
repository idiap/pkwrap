#!/usr/bin/env python3
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

import torch
import pkwrap

pkwrap.InstantiateKaldiCuda()
input_mat = pkwrap.matrix.ReadKaldiMatrixFile("tests/data/lfmmi_input_0")
expected_out = pkwrap.matrix.ReadKaldiMatrixFile("tests/data/lfmmi_deriv_0")

den_graph = pkwrap.chain.LoadDenominatorGraph('tests/data/den.fst', 3304)
training_opts = pkwrap.chain.CreateChainTrainingOptions(5e-05, 0.01, 0.25, 0.025);
# supervision = pkwrap.chain.ReadOneSupervisionFile("ark:train_egs.ark")
supervision = pkwrap.chain.ReadSupervisionFromFile("supervision_0")

input_mat_gpu = input_mat.cuda()
objf = torch.zeros(1)
l2_term = torch.zeros(1)
weight = torch.zeros(1)
nnet_output_deriv = torch.zeros(input_mat.size(0), input_mat.size(1)).cuda()
xent_deriv = torch.zeros_like(nnet_output_deriv).cuda()
pkwrap.chain.ComputeChainObjfAndDeriv(training_opts, den_graph, supervision, input_mat_gpu, 
        objf,
        l2_term,
        weight,
        nnet_output_deriv,
        xent_deriv)
print(nnet_output_deriv)

