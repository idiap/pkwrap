// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>
//
#ifndef PKWRAP_COMMON_H_
#define PKWRAP_COMMON_H_
#include <pybind11/pybind11.h> 
#include <typeinfo>
#include <string>

// Kaldi related imports
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-normalize-component.h"
#include "cudamatrix/cu-matrix.h"

// Pytorch imports
#include <torch/extension.h>
#endif
