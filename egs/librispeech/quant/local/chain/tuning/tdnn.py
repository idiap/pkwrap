#!/usr/bin/env python3

# %WER 20.06 [ 4040 / 20138, 370 ins, 638 del, 3032 sub ]
# not a big difference compared to 1d or 1f. 

description = """
    this is 1c, but also uses residue
    """

import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import pkwrap
import numpy as np
from torch.nn.utils import clip_grad_value_
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
tdnn_class_path = '/idiap/temp/aprasad/pkwrap/egs/am_quantization/local/model_classes'
sys.path.insert(0, tdnn_class_path)

from pytorch_components import TDNN as TDNN
from pytorch_components import FixedAffineLayer as FixedAffineLayer
from pytorch_components import BatchNorm as BatchNorm


def train_lfmmi_one_iter(model, egs_file, den_fst_path, training_opts, feat_dim, 
    minibatch_size="64", use_gpu=True, lr=0.0001, weight_decay=0.25, frame_shift=0, print_interval=10):
    pkwrap.kaldi.InstantiateKaldiCuda()
    if training_opts is None:
        training_opts = pkwrap.kaldi.chain.CreateChainTrainingOptionsDefault()
    den_graph = pkwrap.kaldi.chain.LoadDenominatorGraph(den_fst_path, model.output_dim)
    criterion = pkwrap.chain.KaldiChainObjfFunction.apply
    if use_gpu:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    acc_sum = torch.tensor(0., requires_grad=False)
    for mb_id, merged_egs in enumerate(pkwrap.chain.prepare_minibatch(egs_file, minibatch_size)):
        features = pkwrap.kaldi.chain.GetFeaturesFromEgs(merged_egs)
        chunk_size = 140
        left_context = 15
        right_context = 15
        features = features[:,frame_shift:frame_shift+chunk_size+left_context+right_context:,:]
        # features = features[:,1+frame_shift:1+140+25+frame_shift,:]
        features = features.cuda()
        output, xent_output = model(features)
        print(output.shape)
        sup = pkwrap.kaldi.chain.GetSupervisionFromEgs(merged_egs)
        deriv = criterion(training_opts, den_graph, sup, output, xent_output)
        acc_sum.add_(deriv[0])
        if mb_id>0 and mb_id%print_interval==0:
            sys.stderr.write("Overall objf={}\n".format(acc_sum/print_interval))
            acc_sum.zero_()
        optimizer.zero_grad()
        deriv.backward()
        clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()
    sys.stdout.flush()
    model = model.cpu()
    return model

class Net(nn.Module):
    def __init__(self, output_dim, feat_dim, params=None, lda_matrix=None):
        super(Net, self).__init__()
        self.input_dim = feat_dim
        self.output_dim = output_dim

        if params is not None:
            self.fal = FixedAffineLayer(self.input_dim, lda_matrix, [-1,0,1])
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
            self.prefinal_xent = TDNN(625, 625, [0], params['prefinal-xent.affine'])
            self.prefinal_xent_batchnorm = BatchNorm(625, params['prefinal-xent.batchnorm'])
            self.output_xent_affine = TDNN(625, num_outputs, [0], params['output-xent.affine'])
        else:
            self.fal = FixedAffineLayer(self.input_dim, context=[-1,0,1])
            self.tdnn1 = TDNN(120, 625, [0])
            self.bn1 = BatchNorm(625)
            self.tdnn2 = TDNN(625, 625, [-1,0,1])
            self.bn2 = BatchNorm(625)
            self.tdnn3 = TDNN(625, 625, [-1,0,1], subsampling_factor=3)
            self.bn3 = BatchNorm(625)
            self.tdnn4 = TDNN(625, 625, [-3,0,3])
            self.bn4 = BatchNorm(625)
            self.tdnn5 = TDNN(625, 625, [-3,0,3])
            self.bn5 = BatchNorm(625)
            self.tdnn6 = TDNN(625, 625, [-3,0,3])
            self.bn6 = BatchNorm(625)
            self.tdnn7 = TDNN(625, 625, [-3,0,3])
            self.bn7 = BatchNorm(625)
            self.prefinal_chain = TDNN(625, 625, [0])
            self.prefinal_chain_bn = BatchNorm(625)
            self.output = TDNN(625, num_outputs, [0])
            self.prefinal_xent = TDNN(625, 625, [0])
            self.prefinal_xent_batchnorm = BatchNorm(625)
            self.output_xent_affine = TDNN(625, num_outputs, [0])


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
        tdnn7_out = self.bn7(x)

        prefinal_chain_out = self.prefinal_chain(tdnn7_out)
        prefinal_chain_out = F.relu(prefinal_chain_out)
        prefinal_chain_out = self.prefinal_chain_bn(prefinal_chain_out)

        chain_out = self.output(prefinal_chain_out)  
        prefinal_xent_out = self.prefinal_xent(tdnn7_out)
        prefinal_xent_out = F.relu(prefinal_xent_out)
        prefinal_xent_out = self.prefinal_xent_batchnorm(prefinal_xent_out)

        xent_out = self.output_xent_affine(prefinal_xent_out)
        return chain_out, F.log_softmax(xent_out, dim=2)

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--mode", default="init")
        parser.add_argument("--dir", default="")
        parser.add_argument("--lr", default=0.0001, type=float)
        parser.add_argument("--egs", default="")
        parser.add_argument("--new-model", default="")
        parser.add_argument("--l2-regularize", default=1e-4, type=float)
        parser.add_argument("--l2-regularize-factor", default=1.0, type=float) # this is the weight_decay in pytorch
        parser.add_argument("--out-of-range-regularize", default=0.01, type=float)
        parser.add_argument("--xent-regularize", default=0.025, type=float)
        parser.add_argument("--leaky-hmm-coefficient", default=0.1, type=float)
        parser.add_argument("--minibatch-size", default="32", type=str)
        parser.add_argument("--decode-feats", default="data/test/feats.scp", type=str)
        parser.add_argument("--decode-output", default="-", type=str)
        parser.add_argument("--decode-iter", default="final", type=str)
        parser.add_argument("--frame-shift", default=0, type=int)
        parser.add_argument("base_model")

        args = parser.parse_args()
        dirname = args.dir
        num_outputs = None
        with open(os.path.join(dirname, "num_pdfs")) as ipf:
            num_outputs = int(ipf.readline().strip())
        assert num_outputs is not None
        feat_dim = None
        with open( os.path.join(dirname, "feat_dim")) as ipf:
            feat_dim = int(ipf.readline().strip())
        assert feat_dim is not None

        if args.mode == 'init':
            old_model_dir = '/idiap/temp/aprasad/kaldi/egs/librispeech/s5a/exp/chain_cleaned/tdnn_7k_1a_sp'
            lda_path = os.path.join(old_model_dir, 'configs', 'lda.mat')
            lda_matrix = pkwrap.kaldi.nnet3.LoadAffineTransform(lda_path)
            model_path = os.path.join(old_model_dir, 'final.mdl')
            model_params = pkwrap.kaldi.nnet3.GetNNet3Components(model_path)
            params = {name: param for index, name, param in model_params}
            model = Net(num_outputs, feat_dim, params=params, lda_matrix=lda_matrix)
            torch.save(model.state_dict(), args.base_model)
        elif args.mode == 'training':
            lr = args.lr
            den_fst_path = os.path.join(dirname, "den.fst")

#           load model
            model = Net(num_outputs, feat_dim)
            base_model = args.base_model
            loader = torch.load(base_model)
            model.load_state_dict(torch.load(base_model))
            sys.stderr.write("Loaded base model from {}".format(base_model))

            training_opts = pkwrap.kaldi.chain.CreateChainTrainingOptions(args.l2_regularize, 
                                                                          args.out_of_range_regularize, 
                                                                          args.leaky_hmm_coefficient, 
                                                                          args.xent_regularize) 
            new_model = train_lfmmi_one_iter(
                            model,
                            args.egs, 
                            den_fst_path, 
                            training_opts, 
                            feat_dim, 
                            minibatch_size=args.minibatch_size, 
                            lr=args.lr,
                            weight_decay=args.l2_regularize_factor,
                            frame_shift=args.frame_shift)
            torch.save(new_model.state_dict(), args.new_model)
        elif args.mode == 'diagnostic':
            # TODO: implement diagnostics
            pass
        elif args.mode == 'merge':
            with torch.no_grad():
                base_models = args.base_model.split(',')
                assert len(base_models)>0
                model0 = Net(num_outputs, feat_dim)
                model0.load_state_dict(torch.load(base_models[0]))
                model_acc = dict(model0.named_parameters())
                for mdl_name in base_models[1:]:
                    this_mdl = Net(num_outputs, feat_dim)
                    this_mdl.load_state_dict(torch.load(mdl_name))
                    for name, params in this_mdl.named_parameters():
                        model_acc[name].data.add_(params.data)
                weight = 1.0/len(base_models)
                for name in model_acc:
                    model_acc[name].data.mul_(weight)
                print(list(model0.parameters())[0])
                torch.save(model0.state_dict(), args.new_model)

        elif args.mode == 'decode':
            with torch.no_grad():
                model = Net(num_outputs, feat_dim)
                base_model = args.base_model
                try:
                    model.load_state_dict(torch.load(base_model))
                except:
                    sys.stderr.write("Cannot load model {}".format(base_model))
                    quit(1)
                model.eval()
                writer_spec = "ark,t:{}".format(args.decode_output)
                writer = pkwrap.script_utils.feat_writer(writer_spec)
                for key, feats in pkwrap.script_utils.feat_reader_gen(args.decode_feats):
                    feats_with_context = pkwrap.matrix.add_context(feats, 15, 15).unsqueeze(0)
                    post, _ = model(feats_with_context)
                    post = post.squeeze(0).contiguous()
                    tensor_to_kmat = pkwrap.kaldi.matrix.TensorToKaldiMatrix(post)
                    sys.stderr.write(str(type(tensor_to_kmat)))
                    sys.stderr.write('\n')
                    # quit(1)
                    writer.Write(key, pkwrap.kaldi.matrix.TensorToKaldiMatrix(post))
                    sys.stderr.write("Wrote {}\n ".format( key))
                    sys.stderr.flush()
                writer.Close()
                sys.stdout.flush()
