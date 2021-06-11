"""Module to prepare egs for e2e-lfmmi training with wav2vec2 models"""
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

import logging
import os
from collections import defaultdict
import subprocess
import io
import random
import torch
from _pkwrap import kaldi
# NOTE: soundfile was exceptionally faster than librosa
import soundfile


class EgsInfo(object):
    """EgsInfo objects hole information about each example"""

    def __init__(self, name, wav, fstscp, num_output_frames):
        self.name = name
        self.wav = wav
        self.fst = kaldi.fst.StdVectorFst()
        kaldi.fst.ReadFstKaldi(fstscp, self.fst)
        self.num_output_frames = int(num_output_frames)
        self.supervision = None

    def create_supervision(self, trans_mdl):
        """Creates supervision object given transition model

        Calls TrainingGraphToSupervisionE2e from Kaldi library

        Args:
            trans_mdl: Transition model object

        Returns:
            None
        """
        self.supervision = kaldi.chain.Supervision()
        kaldi.chain.TrainingGraphToSupervisionE2e(
            self.fst,
            trans_mdl,
            self.num_output_frames,
            self.supervision
        )

    # TODO(srikanth): check if the supervision is already normalized to avoid
    # applying normalization more than once
    def normalize_supervision(self, normalize_fst):
        """Normalize supervision with fst"""
        if self.supervision is not None:
            kaldi.chain.AddWeightToSupervisionFst(normalize_fst, self.supervision)

def prepare_e2e_minibatch(batch):
    """returns a tuple of minibatch tensor and merged supervision

    Args:
        batch: a list of egs objects

    Returns:
        feats: a Tensor of size minibatch_size x time x dimension
        merged_sup: Supervision object to be used as target for LF-MMI

    Raises:
        IOError: when something wrong while read a file
    """
    # load the audio
    if not batch:
        return None, None
    feat_list = []
    devnull = open(os.devnull, 'w')
    for egs in batch:
        try:
            wav_read_process = subprocess.Popen(
                ' '.join(egs.wav),
                stdout=subprocess.PIPE,
                shell=True,
                stderr=devnull
            )
            samples, _ = soundfile.read(
                io.BytesIO(wav_read_process.communicate()[0])
            )
            feat_list.append(samples)
        except Exception:
            raise IOError("Error processing {}".format(egs.name))
    merged_sup = kaldi.chain.Supervision()
    kaldi.chain.MergeSupervisionE2e([egs.supervision for egs in batch], merged_sup)
    feats_torch = torch.tensor(feat_list, dtype=torch.float32, requires_grad=False)
    return (feats_torch, merged_sup)


class Wav2vec2BatchSampler(torch.utils.data.BatchSampler): # pylint: disable=too-few-public-methods
    """An extension of BatchSampler to handle raw egs"""
    def __iter__(self):
        batch_by_length = defaultdict(list)
        #for i in torch.utils.data.RandomSampler(range(len(self.sampler))):
        for i in range(len(self.sampler)):
            idx = self.sampler[i]
            num_output_frames = idx.num_output_frames
            batch_by_length[num_output_frames].append(idx)
            if len(batch_by_length[num_output_frames]) == self.batch_size:
                # we will have to normalize and merge egs
                yield prepare_e2e_minibatch(batch_by_length[num_output_frames])
                batch_by_length[num_output_frames] = []
        for num_output_frames in batch_by_length:
            if batch_by_length[num_output_frames] and not self.drop_last:
                yield prepare_e2e_minibatch(batch_by_length[num_output_frames])


class Wav2vec2EgsDataset(torch.utils.data.Dataset):
    """A Pytorch Dataset class to prepare Egs for Wav2vec2 models"""

    # TODO(srikanth): should reduce the number of parameters or reconfigure pylint
    def __init__(
            self,
            egs_folder,
            cegs_idx,
            transition_model_filename,
            normalization_fst_rxfilename,
            sampling_rate=16000,
            transition_model_binary_mode=True,
            shuffle=False,
    ):
        """instantiates a Pytorch Dataset for E2E training

        Args:
            egs_folder: the folder containing egs
            cegs_idx: index of egs, s.t. egs.cegs_idx.scp exists
            transition_model: transition model that maps transition ids to pdf ids
            normalization_fst: fst to normalize when supervision is created
            sampling_rate: sampling rate of the audio files
        """
        self.transition_model = kaldi.hmm.TransitionModel()
        kaldi.hmm.ReadTransitionModel(
            self.transition_model,
            transition_model_filename,
            transition_model_binary_mode
        )
        self.normalization_fst = kaldi.fst.StdVectorFst()
        kaldi.fst.ReadFstKaldi(normalization_fst_rxfilename, self.normalization_fst)
        self.sampling_rate = sampling_rate
        self.prepare_egs(egs_folder, cegs_idx)
        if shuffle:
            random.shuffle(self.egs_holder)

    def __len__(self):
        return len(self.egs_holder)

    def __getitem__(self, idx):
        return self.egs_holder[idx]

    def __item__(self, i):
        # we may need to return wav and normalized fst instead
        return self.egs_holder[i]

    def prepare_egs(self, egs_folder, cegs_index):
        """Method to prepare egs_holder"""
        # egs file is wav.scp file
        egs_file = "{}/egs.{}.ark".format(egs_folder, cegs_index)
        fst_file = "{}/fst.{}.scp".format(egs_folder, cegs_index)
        utt2len_file = "{}/utt2len.{}.ark".format(egs_folder, cegs_index)
        utt2wav = Wav2vec2EgsDataset.read_wav_scp(egs_file)
        utt2len = Wav2vec2EgsDataset.read_utt2len_file(utt2len_file)
        egs_holder = self.get_egs_holder(fst_file, utt2wav, utt2len)

        allowed_durations_filename = "{}/allowed_durs.txt".format(egs_folder)
        self.allowed_durations = set([float(ln) for ln in open(allowed_durations_filename)])
        self.allowed_durations_sorted = sorted(list(self.allowed_durations))
        self.egs_holder = egs_holder

    @staticmethod
    def read_wav_scp(wav_scp):
        """Reads wav.scp file and returns a dictionary

        Args:
            wav_scp: a string, contains the path to wav.scp

        Returns:
            utt2wav: a dictionary, keys are the first column of wav.scp
                and values are the second column
        """
        utt2wav = {}
        with open(wav_scp) as ipf:
            for line in ipf:
                lns = line.strip().rstrip('|').split()
                uttname = lns[0]
                utt2wav[uttname] = lns[1:]
        return utt2wav

    # TODO: refactor this into some utilility that can read 2 column files
    @staticmethod
    def read_utt2dur_file(utt2dur_file):
        """read utt2dur file and return a dictionary"""
        utt2dur = {}
        with open(utt2dur_file) as utt2dur_f:
            for line in utt2dur_f:
                lns = line.strip().split()
                utt2dur[lns[0]] = float(lns[1])
        return utt2dur

    @staticmethod
    def read_utt2len_file(utt2len_file):
        """read utt2len file, second column is the number of output frames"""
        utt2len = {}
        with open(utt2len_file) as utt2len_f:
            for line in utt2len_f:
                lns = line.strip().split()
                utt2len[lns[0]] = float(lns[1])
        return utt2len

    def get_egs_holder(self, fst_file, utt2wav, utt2len):
        """Populate egs_holder"""
        egs_holder = []
        with open(fst_file) as ipf:
            total, done, skipped = 0, 0, 0
            for line in ipf:
                lns = line.strip().split()
                total += 1
                try:
                    uttname, fstscp = lns
                except Exception:
                    logging.error("Excepted fst file %s to have only 2 columns", fst_file)
                if uttname not in utt2wav:
                    skipped += 1
                    continue
                if uttname not in utt2len:
                    logging.warning("Cannot find number of output frames for %s", uttname)
                    skipped += 1
                    continue
                this_egs_info = EgsInfo(uttname, utt2wav[uttname], fstscp, utt2len[uttname])
                min_path_length = kaldi.chain.FindMinimumLengthPathFromFst(this_egs_info.fst)
                if min_path_length > this_egs_info.num_output_frames:
                    logging.warning(
                        "get_egs_holder, %s has more labels than frames",
                        this_egs_info.name
                    )
                    skipped += 1
                    continue

                this_egs_info.create_supervision(self.transition_model)
                this_egs_info.normalize_supervision(self.normalization_fst)
                egs_holder.append(this_egs_info)
                done += 1
            logging.info(
                "In get_egs_holder: total=%d, done=%d, skipped=%d",
                total,
                done,
                skipped
            )
        return egs_holder
