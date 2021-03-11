# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

import sys
import os
import random
from collections import OrderedDict, Counter
import logging
import argparse
from dataclasses import dataclass
from librosa.core.constantq import __num_two_factors
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
import torch.optim as optim
from _pkwrap import kaldi
from . import chain
from . import matrix
from . import script_utils
from collections import defaultdict, namedtuple
import librosa
import subprocess
import io
from math import ceil
from .objf import train_lfmmi_one_iter

@dataclass
class TrainerOpts:
    mode: str = "init"
    dir: str = ""
    lr: float = 0.001
    minibatch_size: int = 32
    base_model: str = ''

@dataclass
class DecodeOpts:
    decode_feats: str = 'data/test/feats.scp'
    decode_output: str = '-'

@dataclass
class ChainModelOpts(TrainerOpts, DecodeOpts):
    egs: str = ""
    new_model: str = ""
    l2_regularize: float = 1e-4
    l2_regularize_factor: float = 1.0
    out_of_range_regularize: float = 0.01
    leaky_hmm_coefficient: float = 0.1
    xent_regularize: float = 0.025
    minibatch_size: str = "32"
    frame_shift: int = 0
    output_dim: int = 1
    feat_dim: int = 1
    context: int = 0
    frame_subsampling_factor: int = 3

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self


class ChainModel(nn.Module):
    def __init__(self, model_cls, cmd_line=False, **kwargs):
        """initialize a ChainModel

        The idea behind this class is to split the various functionalities
        across methods so that we can reuse whataver is required and reimplement
        only that which is necessary
        """
        super(ChainModel, self).__init__()
        assert model_cls is not None
        if cmd_line:
            args = self.load_cmdline_args()
            self.chain_opts = ChainModelOpts()
            self.chain_opts.load_from_config(vars(args))
        else:
            self.chain_opts = ChainModelOpts()
            self.chain_opts.load_from_config(kwargs)

        self.Net = model_cls
        self.call_by_mode()

    def call_by_mode(self):
        """A that calls appropriate method based on the value of chain_opts.mode

        So far the modes supported are:
            - init
            - context
            - merge
            - train (or training)
            - validate (or diagnostic)
            - infer (or decode)
        """
        self.reset_dims()
        if self.chain_opts.mode != 'context':
            self.load_context()
        if self.chain_opts.mode == 'init':
            self.init()
        elif self.chain_opts.mode == 'context':
            self.context()
        elif self.chain_opts.mode == 'merge':
            self.merge()
        elif self.chain_opts.mode in ['validate', 'diagnostic'] :
            self.validate()
        elif self.chain_opts.mode in ['train', 'training']:
            self.train()
        elif self.chain_opts.mode in ['decode', 'infer']:
            self.infer()
        elif self.chain_opts.mode == 'final_combination':
            self.combine_final_model()

    def init(self):
        """Initialize the model and save it in chain_opts.base_model"""
        model = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
        torch.save(model.state_dict(), self.chain_opts.base_model)

    def train(self):
        """Run one iteration of LF-MMI training

        This is called by
        >>> self.train()

        It will probably be renamed as self.fit() since this seems to be
        the standard way other libraries call the training function.
        """
        chain_opts = self.chain_opts
        lr = chain_opts.lr
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")

        # load model
        model = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
        model.load_state_dict(torch.load(chain_opts.base_model))

        training_opts = kaldi.chain.CreateChainTrainingOptions(
                chain_opts.l2_regularize,
                chain_opts.out_of_range_regularize,
                chain_opts.leaky_hmm_coefficient,
                chain_opts.xent_regularize,
        )
        context = chain_opts.context
        new_model = train_lfmmi_one_iter(
            model,
            chain_opts.egs,
            den_fst_path,
            training_opts,
            chain_opts.feat_dim,
            minibatch_size=chain_opts.minibatch_size,
            left_context=context,
            right_context=context,
            lr=chain_opts.lr,
            weight_decay=chain_opts.l2_regularize_factor,
            frame_shift=chain_opts.frame_shift
        )
        torch.save(new_model.state_dict(), chain_opts.new_model)

    @torch.no_grad()
    def validate(self):
        kaldi.InstantiateKaldiCuda()
        chain_opts = self.chain_opts
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")

#           load model
        model = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
        model.load_state_dict(torch.load(chain_opts.base_model))
        model.eval()

        training_opts = kaldi.chain.CreateChainTrainingOptions(
                chain_opts.l2_regularize,
                chain_opts.out_of_range_regularize,
                chain_opts.leaky_hmm_coefficient,
                chain_opts.xent_regularize,
        )
        compute_chain_objf(
            model,
            chain_opts.egs,
            den_fst_path,
            training_opts,
            minibatch_size="1:64",
            left_context=chain_opts.context,
            right_context=chain_opts.context,
        )

    @torch.no_grad()
    def merge(self):
        chain_opts = self.chain_opts
        base_models = chain_opts.base_model.split(',')
        assert len(base_models)>0
        model0 = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
        model0.load_state_dict(torch.load(base_models[0]))
        model_acc = dict(model0.named_parameters())
        for mdl_name in base_models[1:]:
            this_mdl = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
            this_mdl.load_state_dict(torch.load(mdl_name))
            for name, params in this_mdl.named_parameters():
                model_acc[name].data.add_(params.data)
        weight = 1.0/len(base_models)
        for name in model_acc:
            model_acc[name].data.mul_(weight)
        torch.save(model0.state_dict(), chain_opts.new_model)

    @torch.no_grad()
    def infer(self):
        chain_opts = self.chain_opts
        model = self.Net(chain_opts.feat_dim, chain_opts.output_dim)
        base_model = chain_opts.base_model
        try:
            model.load_state_dict(torch.load(base_model))
        except Exception as e:
            logging.error(e)
            logging.error("Cannot load model {}".format(base_model))
            quit(1)
        # TODO(srikanth): make sure context is a member of chain_opts
        context = chain_opts.context
        model.eval()
        writer_spec = "ark,t:{}".format(chain_opts.decode_output)
        writer = script_utils.feat_writer(writer_spec)
        for key, feats in script_utils.feat_reader_gen(chain_opts.decode_feats):
            feats_with_context = matrix.add_context(feats, context, context).unsqueeze(0)
            post, _ = model(feats_with_context)
            post = post.squeeze(0)
            writer.Write(key, kaldi.matrix.TensorToKaldiMatrix(post))
            logging.info("Wrote {}".format(key))
        writer.Close()

    def context(self):
        """Find context by brute force

        WARNING: it only works for frame_subsampling_factor=3
        """

        logging.warning("context function called. it only works for frame_subsampling_factor=3")
        visited = Counter()
        with torch.no_grad():
          feat_dim = 40
          num_pdfs = 300
          model = self.Net(40, 300)
          chunk_sizes = [(150,50), (50, 17), (100, 34), (10, 4), (20, 7)]
          frame_shift = 0
          left_context = 0
          logging.info("Searching for context...")
          while True:
              right_context = left_context
              found = []
              for chunk_len, output_len in chunk_sizes:
                  feat_len = chunk_len+left_context+right_context
                  assert feat_len > 0
                  try:
                      test_feats = torch.zeros(32, feat_len, feat_dim)
                      y = model(test_feats)
                  except Exception as e:
                      visited[left_context] += 1
                      if visited[left_context] > 10:
                          break
                  if y[0].shape[1] == output_len:
                      found.append(True)
                  else:
                      found.append(False)
                      break
              if all(found):
                      self.save_context(left_context)
                      return
              left_context += 1
              if left_context >= 100:
                  raise NotImplementedError("more than context of 100")
          raise Exception("No context found")

    def save_context(self, value):
          logging.info(f"Left_context = {value}")
          with open(os.path.join(self.chain_opts.dir, 'context'), 'w') as opf:
              opf.write(f'{value}')
              opf.close()

    def load_context(self):
        self.chain_opts.context = script_utils.read_single_param_file(
                os.path.join(self.chain_opts.dir, 'context'),
        )

    def reset_dims(self):
        # what if the user wants to pass it? Just override this function
        num_pdfs_filename = os.path.join(
            self.chain_opts.dir,
            "num_pdfs"
        )
        self.chain_opts.output_dim = script_utils.read_single_param_file(num_pdfs_filename)

        feat_dim_filename = os.path.join(
            self.chain_opts.dir,
            "feat_dim"
        )
        # checking this because we don't always need feat_dim (e.g. when
        # generating context)
        if os.path.isfile(feat_dim_filename):
            self.chain_opts.feat_dim = script_utils.read_single_param_file(feat_dim_filename)


    def load_model_context(self):
        context_file_name = os.path.join(self.chain_opts.dir, 'context')
        context = script_utils.read_single_param_file(context_file_name)

    def load_cmdline_args(self):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--mode", default="init")
        parser.add_argument("--dir", default="")
        parser.add_argument("--lr", default=0.001, type=float)
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
        return args

    @torch.no_grad()
    def combine_final_model(self):
        """Implements Kaldi-style model ensembling"""
        kaldi.InstantiateKaldiCuda()
        chain_opts = self.chain_opts
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")
        base_models = chain_opts.base_model.split(',')
        assert len(base_models)>0
        training_opts = kaldi.chain.CreateChainTrainingOptions(
                chain_opts.l2_regularize,
                chain_opts.out_of_range_regularize,
                chain_opts.leaky_hmm_coefficient,
                chain_opts.xent_regularize,
        )

        moving_average = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
        best_mdl =  self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
        moving_average.load_state_dict(torch.load(base_models[0]))
        moving_average.cuda()
        best_mdl = moving_average
        compute_objf = lambda mdl: compute_chain_objf(
            mdl,
            chain_opts.egs,
            den_fst_path,
            training_opts,
            minibatch_size="1:64", # TODO: this should come from a config
            left_context=chain_opts.context,
            right_context=chain_opts.context,
            frame_shift=chain_opts.frame_shift,
        )

        _, init_objf = compute_objf(moving_average)
        best_objf = init_objf

        model_acc = dict(moving_average.named_parameters())
        num_accumulated = torch.Tensor([1.0]).reshape(1).cuda()
        best_num_to_combine = 1
        for mdl_name in base_models[1:]:
            this_mdl = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
            logging.info("Combining model {}".format(mdl_name))
            this_mdl.load_state_dict(torch.load(mdl_name))
            this_mdl = this_mdl.cuda()
            # TODO(srikanth): check why is this even necessary
            moving_average.cuda()
            num_accumulated += 1.
            for name, params in this_mdl.named_parameters():
                model_acc[name].data.mul_((num_accumulated-1.)/(num_accumulated))
                model_acc[name].data.add_(params.data.mul_(1./num_accumulated))
            _, this_objf = compute_objf(moving_average)
            if this_objf > best_objf:
                best_objf = this_objf
                best_mdl = moving_average
                best_num_to_combine = int(num_accumulated.clone().detach())
                logging.info("Found best model")
            else:
                logging.info("Won't update best model")

        logging.info("Combined {} models".format(best_num_to_combine))
        logging.info("Initial objf = {}, Final objf = {}".format(init_objf, best_objf))
        best_mdl.cpu()
        torch.save(best_mdl.state_dict(), chain_opts.new_model)
        return self

class ChainE2EModel(ChainModel):
    """Extension of ChainModel to handle Chain E2E training"""
    def get_optimizer(self, model, lr=0.01, weight_decay=0.001, **kwargs):
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        return optimizer

    def train(self):
        """Run one iteration of LF-MMI training

        This is called by
        >>> self.train()

        It will probably be renamed as self.fit() since this seems to be
        the standard way other libraries call the training function.
        """
        kaldi.InstantiateKaldiCuda()
        chain_opts = self.chain_opts
        lr = chain_opts.lr
        den_fst_path = os.path.join(chain_opts.dir, "den.fst")

#           load model
        model = self.Net(self.chain_opts.feat_dim, self.chain_opts.output_dim)
        model.load_state_dict(torch.load(chain_opts.base_model))

        training_opts = kaldi.chain.CreateChainTrainingOptions(
                chain_opts.l2_regularize,
                chain_opts.out_of_range_regularize,
                chain_opts.leaky_hmm_coefficient,
                chain_opts.xent_regularize,
        )
        logging.info("xent passed as {}".format(chain_opts.xent_regularize))
        context = chain_opts.context
        model = model.cuda()
        optimizer = self.get_optimizer(model, lr=chain_opts.lr, weight_decay=chain_opts.l2_regularize_factor)
        new_model = train_lfmmi_one_iter(
            model,
            chain_opts.egs,
            den_fst_path,
            training_opts,
            chain_opts.feat_dim,
            minibatch_size=chain_opts.minibatch_size,
            left_context=context,
            right_context=context,
            lr=chain_opts.lr,
            weight_decay=chain_opts.l2_regularize_factor,
            frame_shift=chain_opts.frame_shift,
            optimizer=optimizer,
            e2e = True
        )
        torch.save(new_model.state_dict(), chain_opts.new_model)

    def context(self):
        """Write context of the model to 0 because the Net is designed to pad its own context"""
        self.save_context(0)


