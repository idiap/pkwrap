// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

#include "fst.h"

void ReadFstKaldi(std::string rxfilename, fst::StdVectorFst &fst) {
    try {
      fst::ReadFstKaldi(rxfilename, &fst);
    } catch (...) {
        std::cerr << "Error opening " << rxfilename << std::endl;
        return;
    }
    return;
}
