#include "hmm.h"

void ReadTransitionModel(kaldi::TransitionModel &trans_model, std::string trans_model_rxfilename, bool binary) {
    ReadKaldiObject(trans_model_rxfilename, &trans_model);
}
