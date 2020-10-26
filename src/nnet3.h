// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

#ifndef PKWRAP_NNET3_H_
#define PKWRAP_NNET3_H_
#include <typeinfo>
#include <string>
#include "common.h"
#include "matrix.h"
#include <torch/extension.h>
#include "nnet3/natural-gradient-online.h"
#include "nnet3/nnet-convolutional-component.h"
kaldi::BaseFloat precondition_directions(kaldi::nnet3::OnlineNaturalGradient &state, torch::Tensor &grad);
std::vector<std::tuple<int, std::string,std::vector<torch::Tensor> > > GetNNet3Components(std::string model_path);
void SaveNNet3Components(std::string model_path,
                         std::string new_model_path,
                         std::vector<std::tuple<int, std::string,std::vector<torch::Tensor> > > & new_params);
torch::Tensor LoadAffineTransform(std::string matrix_path);
#endif
