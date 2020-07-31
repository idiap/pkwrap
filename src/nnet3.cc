// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

#include "nnet3.h"

// A wrapper to PreconditionDirections of OnlineNaturalGradient: it gets a pointer
// to the Tensor in the GPU and passed it to PreconditionDirections
kaldi::BaseFloat precondition_directions(kaldi::nnet3::OnlineNaturalGradient &state, torch::Tensor &grad) {
    kaldi::BaseFloat scale_;
    auto grad_cumat = TensorToKaldiCuSubMatrix(grad);
    state.PreconditionDirections(&grad_cumat, &scale_);
    return scale_;
}