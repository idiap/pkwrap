#ifndef PKWRAP_HMM_H_
#define PKWRAP_HMM_H_
#include "common.h"

void ReadTransitionModel(kaldi::TransitionModel &trans_model, std::string trans_model_rxfilename, bool binary);

#endif
