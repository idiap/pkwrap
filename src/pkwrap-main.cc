// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>
//
#include "pkwrap-main.h"

inline void InstantiateKaldiCuda() {
    kaldi::CuDevice::Instantiate().SelectGpuId("yes");
}
