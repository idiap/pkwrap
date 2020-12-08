#!/usr/bin/env python3

# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Srikanth Madikeri <srikanth.madikeri@idiap.ch>,
# Amrutha Prasad <amrutha.prasad@idiap.ch>

usage="""
    Load Kaldi acoustic model and run forward pass to compute the log-likelihoods.
"""

import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
nn_path = os.path.join(dir_path, '../nn')
sys.path.insert(0, nn_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pkwrap
import numpy as np
from pytorch_components import TDNN as TDNN
from pytorch_components import FixedAffineLayer as FixedAffineLayer
from pytorch_components import BatchNorm as BatchNorm
import argparse


class Net(nn.Module):
    def __init__(self, model_path='', lda_path='', num_outputs=41):
        super(Net, self).__init__()
        model_params = pkwrap.kaldi.nnet3.GetNNet3Components(model_path)
        params = {name: param for index, name, param in model_params}
        # params = dict(model_params)
        self.outputsize = nn.Parameter(torch.Tensor(1,))
        self.outputsize[0] = num_outputs
        self.lda = pkwrap.kaldi.nnet3.LoadAffineTransform(lda_path)
        self.fal = FixedAffineLayer(120, self.lda, [-1,0,1])
        self.tdnn1 = TDNN(120, 625, [0], params['tdnn1.affine'])
        self.bn1 = BatchNorm(625, params['tdnn1.batchnorm'])
        self.tdnn2 = TDNN(625, 625, [-1,0,1], params['tdnn2.affine'])
        self.bn2 = BatchNorm(625, params['tdnn2.batchnorm'])
        self.tdnn3 = TDNN(625, 625, [-1,0,1], params['tdnn3.affine'], subsampling_factor=3)
        self.bn3 = BatchNorm(625, params['tdnn3.batchnorm'])
        self.tdnn4 = TDNN(625, 625, [-3,0,3], params['tdnn4.affine'])
        self.bn4 = BatchNorm(625, params['tdnn4.batchnorm'])
        self.tdnn5 = TDNN(625, 625, [-3,0,3], params['tdnn5.affine'])
        self.bn5 = BatchNorm(625, params['tdnn5.batchnorm'])
        self.tdnn6 = TDNN(625, 625, [-3,0,3], params['tdnn6.affine'])
        self.bn6 = BatchNorm(625, params['tdnn6.batchnorm'])
        self.tdnn7 = TDNN(625, 625, [-3,0,3], params['tdnn7.affine'])
        self.bn7 = BatchNorm(625, params['tdnn7.batchnorm'])
        self.prefinal_chain = TDNN(625, 625, [0], params['prefinal-chain.affine'])
        self.prefinal_chain_bn = BatchNorm(625, params['prefinal-chain.batchnorm'])
        self.output = TDNN(625, num_outputs, [0], params['output.affine'])

    def forward(self, input):
        mb, T, D = input.shape
        x = self.fal(input)

        x = self.tdnn1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.tdnn2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.tdnn3(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.tdnn4(x)
        x = F.relu(x)
        x = self.bn4(x)

        x = self.tdnn5(x)
        x = F.relu(x)
        x = self.bn5(x)

        x = self.tdnn6(x)
        x = F.relu(x)
        x = self.bn6(x)

        x = self.tdnn7(x)
        x = F.relu(x)
        x = self.bn7(x)

        x = self.prefinal_chain(x)
        x = F.relu(x)
        x = self.prefinal_chain_bn(x)

        x = self.output(x)  
        return x

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument("-i", "--input", help="input features file (feats.scp)",
                        nargs=1, dest="featsFile", required=True)
    parser.add_argument("-o", "--output", help="output file",
                        nargs=1, dest="outputFile", required=True)
    parser.add_argument("-md", "--modelDir", help="final.mdl directory path", nargs=1,
                        dest="modelDir", required=True)
    parser.add_argument("-n", "--numOutput", help="num of ouptuts of the model ",
                        nargs=1, dest="numOutput", default=41, required=True)
    

    # Parse arguments
    args = parser.parse_args()
    featsFile = args.featsFile[0]
    outputFile = args.outputFile[0]
    modelDir = args.modelDir[0]
    numOutput = int(args.numOutput[0])

    feats_scp = 'scp:'+ featsFile
    print ('Input feats file is:', feats_scp)

    log_likelihoods = []
    lda_file = os.path.join(modelDir + '/configs/lda.mat')
    model_file = os.path.join(modelDir + '/final.mdl')
    print('lda file:', lda_file)
    print('model file:', model_file)
    lc = 15
    rc = 15
    nnet = Net(model_file, lda_file, numOutput)
    # nnet.to('cuda')
    nnet.eval()

    with torch.no_grad():
        for key, feat in pkwrap.script_utils.feat_reader_gen(feats_scp):
            x = torch.tensor(np.pad(feat,[(lc,rc),(0,0)], 'edge'))
            x = x.unsqueeze(0)

            y = nnet(x)

            log_likelihoods.append((key, y.squeeze(0)))

    print('Writing output to: {0}'.format(outputFile))         
    pkwrap.kaldi.matrix.WriteFeatures('ark,t:' + outputFile, log_likelihoods)
