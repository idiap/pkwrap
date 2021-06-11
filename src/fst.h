// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

#ifndef PKWRAP_FST_H_
#define PKWRAP_FST_H_
#include "common.h"
#include "fstext/fstext-lib.h"

void ReadFstKaldi(std::string rxfilename, fst::StdVectorFst &fst);

#endif
