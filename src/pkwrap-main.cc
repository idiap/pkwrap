// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>
//
#include "pkwrap-main.h"

static bool instantiated = false;
inline void InstantiateKaldiCuda() {
    if(!instantiated) {
        kaldi::CuDevice::Instantiate().SelectGpuId("yes");
        instantiated = true;
    }
}
