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
from .. import chain
from .. import matrix
from .. import script_utils
from collections import defaultdict, namedtuple
import librosa
import subprocess
import io
from math import ceil

@dataclass
class EgsInfo:
    name: str = ''
    wav: str = ''
    fst: str = ''
    duration: float = 0.
    num_output_frames: int = 0

def prepare_e2e_minibatch():
    pass

class Wav2vec2BatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        batch_by_length = defaultdict(list)
        for idx in self.sampler:
            l = idx.duration
            batch_by_length[l].append(idx)
            if len(batch_by_length[l]) == self.batch_size:
                # we will have to normalize and merge egs
                yield prepare_e2e_minibatch(batch_by_length[l])
                batch_by_length[l] = []
        for l in batch_by_length:
            if batch_by_length[l] and not self.drop_last:
                yield prepare_e2e_minibatch(batch_by_length[l])

class Wav2vec2EgsDataset(torch.utils.data.Dataset):
    def __init__(self, egs_folder, cegs_idx, transition_model, normalization_fst, sampling_rate=16000):
        """instantiates a Pytorch Dataset for E2E training

        Args:
            egs_folder: the folder containing egs
            cegs_idx: index of egs, s.t. egs.cegs_idx.scp exists
            transition_model: transition model that maps transition ids to pdf ids
            normalization_fst: fst to normalize when supervision is created
            sampling_rate: sampling rate of the audio files
        """
        self.transition_model = transition_model
        self.normalization_fst = normalization_fst
        self.sampling_rate = sampling_rate
        self.prepare_egs(egs_folder, cegs_idx)

    def __len__(self):
        return len(self.egs_holder)

    # TODO: test this function
    def process_egs(self, egs):
        """Given an egs of type EgsInfo return a supervision

        This function is exposed so that someone can override its functionality easily

        Args:
            egs: an object of type EgsInfo
        """
        min_diff = 100
        len_extend_context = 0
        # first read the wavfile
        try:
            p = subprocess.Popen(egs.wav, stdout=subprocess.PIPE)
            samples, sampling_rate = librosa.load(io.BytesIO(p.communicate()[0]), sr=self.sampling_rate)
            samples = samples.reshape(1, -1)
        except:
            raise IOError("Error processing {}".format(egs.name))
        duration = samples.shape[1]/sampling_rate
        if duration not in self.allowed_durations:
            raise ValueError("Cannot find duration for {}".format(egs.name))
        num_samples = samples.shape[1]
        num_output_frames = egs.num_output_frames
        fst = kaldi.fst.StdVectorFst()
        kaldi.fst.ReadFstKaldi(egs.fst, fst)
        supervision = kaldi.chain.Supervision()
        # TODO: make sure transition model is loaded
        if not kaldi.chain.TrainingGraphToSupervisionE2e(fst, self.trans_mdl, num_output_frames, supervision):
            raise Exception("Cannot convert fst to supervision for {}".format(egs.name))
        # TODO: add num_states function to Fst
        if self.normalization_fst is not None and self.normalization_fst.num_states() > 0:
            if not kaldi.chain.AddWeightToSupervisionFst(self.normalization_fst, supervision):
                logging.warning("FST was empty for utterance {}".format(egs.name))
        return supervision


    def __item__(self, i):
        # we may need to return wav and normalized fst instead
        return self.process_egs(self.egs_holder[i])

    # TODO: test this function
    def prepare_egs(self, egs_folder, cegs_index):
        # egs file is wav.scp file
        egs_file = "{}/cegs.{}.ark".format(egs_folder, cegs_index)
        fst_file = "{}/fst.{}.scp".format(egs_folder, cegs_index)
        utt2dur_file = "{}/utt2dur.{}".format(egs_folder, cegs_index)
        utt2len_file = "{}/utt2len.{}".format(egs_folder, cegs_index)
        utt2wav = Wav2vec2EgsDataset.read_wav_scp(egs_file)
        utt2dur = Wav2vec2EgsDataset.read_utt2dur_file(utt2dur_file)
        utt2len = Wav2vec2EgsDataset.read_utt2len_file(utt2len_file)
        egs_holder = Wav2vec2EgsDataset.get_egs_holder(fst_file, utt2wav, utt2dur, utt2len)

        allowed_durations_filename = "{}/allowed_durations.txt".format(egs_folder)
        self.allowed_durations = set([float(ln) for ln in open(allowed_durations_filename)])
        self.allowed_durations_sorted = sorted(list(self.allowed_durations))
        self.egs_holder = egs_holder

    @staticmethod
    def read_wav_scp(wav_scp):
        utt2wav = {}
        with open(wav_scp) as ipf:
            for ln in ipf:
                lns = ln.strip().rstrip('|').split()
                uttname = lns[0]
                utt2wav[uttname] = lns[1:]
        return utt2wav

    # TODO: refactor this into some utilility that can read 2 column files
    @staticmethod
    def read_utt2dur_file(utt2dur_file):
        """read utt2dur file"""
        utt2dur = {}
        with open(utt2dur_file) as utt2dur_f:
            for ln in utt2dur_f:
                lns = ln.strip().split()
                utt2dur[lns[0]] = float(lns[1])
        return utt2dur

    @staticmethod
    def read_utt2len_file(utt2len_file):
        """read utt2len file, second column is the number of output frames"""
        utt2len = {}
        with open(utt2len_file) as utt2len_f:
            for ln in utt2len_f:
                lns = ln.strip().split()
                utt2len[lns[0]] = float(lns[1])
        return utt2len

    @staticmethod
    def get_egs_holder(self, fst_file, utt2wav, utt2dur, utt2len):
        egs_holder = []
        with open(fst_file) as ipf:
            total, done, skipped = 0, 0, 0
            for ln in ipf:
                lns = ln.strip().split()
                total += 1
                try:
                    uttname, fstscp = lns
                except:
                    logging.error("Excepted fst file {} to have only 2 columns".format(fst_file))
                if uttname not in utt2wav:
                    skipped += 1
                    continue
                if uttname not in utt2dur:
                    logging.warning("Cannot find duration for {}".format(uttname))
                    skipped += 1
                    continue
                if uttname not in utt2len:
                    logging.warning("Cannot find number of output frames for {}".format(uttname))
                    skipped += 1
                    continue
                this_egs_info = EgsInfo(uttname, utt2wav[uttname], fstscp, utt2dur[uttname], utt2len[uttname])
                egs_holder.append(this_egs_info)
                done += 1
            logging.info("In get_egs_holder: total={}, done={}, skipped={}".format(total, done, skipped))
        return egs_holder

