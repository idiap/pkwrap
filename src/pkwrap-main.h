// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

#ifndef PKWRAP_H_
#define PKWRAP_H_
#include "common.h"
#include "matrix.h"
#include "chain.h"
#include "nnet3.h"
// This is required to make sure kaldi CuMatrix and the likes are actually 
// in the GPU. We don't handle the behavior of the function being called twice though.
inline void InstantiateKaldiCuda();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    auto kaldi_module = m.def_submodule("kaldi");
    kaldi_module.def("InstantiateKaldiCuda", &InstantiateKaldiCuda);

    // matrix related functions
    auto matrix = kaldi_module.def_submodule("matrix");
    matrix.def("KaldiMatrixToTensor", &KaldiMatrixToTensor);
    matrix.def("KaldiCudaMatrixToTensor", &KaldiCudaMatrixToTensor);
    matrix.def("KaldiCudaVectorToTensor", &KaldiCudaVectorToTensor);
    matrix.def("TensorToKaldiCuSubMatrix",&TensorToKaldiCuSubMatrix);
    matrix.def("TensorToKaldiCuSubVector",&TensorToKaldiCuSubVector);
    matrix.def("TensorToKaldiMatrix",&TensorToKaldiMatrix);
    matrix.def("ReadKaldiMatrixFile", &ReadKaldiMatrixFile);
    py::class_<kaldi::Matrix<kaldi::BaseFloat> >(matrix, "Matrix");
    py::class_<kaldi::CuSubMatrix<kaldi::BaseFloat> >(matrix, "CuSubMatrix");
    py::class_<kaldi::SequentialBaseFloatMatrixReader>(matrix, "SequentialBaseFloatMatrixReader")
        .def(py::init<std::string>())
        .def("Next", &kaldi::SequentialBaseFloatMatrixReader::Next)
        .def("Done", &kaldi::SequentialBaseFloatMatrixReader::Done)
        .def("Key", &kaldi::SequentialBaseFloatMatrixReader::Key)
        .def("Value", &kaldi::SequentialBaseFloatMatrixReader::Value);
    py::class_<kaldi::RandomAccessBaseFloatMatrixReader>(matrix, "RandomAccessBaseFloatMatrixReader")
        .def(py::init<std::string>())
        .def("HasKey", &kaldi::RandomAccessBaseFloatMatrixReader::HasKey)
        .def("Value", &kaldi::RandomAccessBaseFloatMatrixReader::Value);
    py::class_<BaseFloatMatrixWriter>(matrix, "BaseFloatMatrixWriter")
        .def(py::init<std::string>())
        .def("Write", &BaseFloatMatrixWriter::Write)
        .def("Close", &BaseFloatMatrixWriter::Close);

    auto nnet3 = kaldi_module.def_submodule("nnet3");
    py::class_<kaldi::nnet3::SequentialNnetChainExampleReader>(nnet3, "SequentialNnetChainExampleReader")
        .def(py::init<std::string>())
        .def("Next", &kaldi::nnet3::SequentialNnetChainExampleReader::Next)
        .def("Done", &kaldi::nnet3::SequentialNnetChainExampleReader::Done)
        .def("Key", &kaldi::nnet3::SequentialNnetChainExampleReader::Key)
        .def("Value", &kaldi::nnet3::SequentialNnetChainExampleReader::Value);
    py::class_<kaldi::nnet3::OnlineNaturalGradient>(nnet3, "OnlineNaturalGradient")
        .def(py::init<>())
        .def("SetRank", &kaldi::nnet3::OnlineNaturalGradient::SetRank)
        .def("SetUpdatePeriod", &kaldi::nnet3::OnlineNaturalGradient::SetUpdatePeriod)
        .def("SetNumSamplesHistory", &kaldi::nnet3::OnlineNaturalGradient::SetNumSamplesHistory)
        .def("SetNumMinibatchesHistory", &kaldi::nnet3::OnlineNaturalGradient::SetNumMinibatchesHistory)
        .def("SetAlpha", &kaldi::nnet3::OnlineNaturalGradient::SetAlpha)
        .def("TurnOnDebug", &kaldi::nnet3::OnlineNaturalGradient::TurnOnDebug)
        .def("GetNumSamplesHistory", &kaldi::nnet3::OnlineNaturalGradient::GetNumSamplesHistory)
        .def("GetNumMinibatchesHistory", &kaldi::nnet3::OnlineNaturalGradient::GetNumMinibatchesHistory)
        .def("GetAlpha", &kaldi::nnet3::OnlineNaturalGradient::GetAlpha)
        .def("GetRank", &kaldi::nnet3::OnlineNaturalGradient::GetRank)
        .def("GetUpdatePeriod", &kaldi::nnet3::OnlineNaturalGradient::GetUpdatePeriod)
        .def("Freeze", &kaldi::nnet3::OnlineNaturalGradient::Freeze)
        .def("PreconditionDirections", &kaldi::nnet3::OnlineNaturalGradient::PreconditionDirections);
    nnet3.def("precondition_directions", &precondition_directions);

    auto chain = kaldi_module.def_submodule("chain");
    // classes from Kaldi
    py::class_<kaldi::chain::DenominatorGraph>(chain, "DenominatorGraph");
    py::class_<kaldi::chain::ChainTrainingOptions>(chain, "ChainTrainingOptions")
        .def_readwrite("xent_regularize", &kaldi::chain::ChainTrainingOptions::xent_regularize)
        .def_readwrite("leaky_hmm_coefficient", &kaldi::chain::ChainTrainingOptions::leaky_hmm_coefficient)
        .def_readwrite("out_of_range_regularize", &kaldi::chain::ChainTrainingOptions::out_of_range_regularize)
        .def_readwrite("l2_regularize", &kaldi::chain::ChainTrainingOptions::l2_regularize);
        
    py::class_<kaldi::nnet3::NnetChainExample>(chain, "NnetChainExample");
    py::class_<kaldi::chain::Supervision>(chain, "Supervision");

    // custom functions
    chain.def("CreateChainTrainingOptions", &CreateChainTrainingOptions);
    chain.def("LoadDenominatorGraph", &LoadDenominatorGraph);
    chain.def("TestLoadDenominatorGraph", &TestLoadDenominatorGraph);
    chain.def("ComputeChainObjfAndDeriv", &ComputeChainObjfAndDeriv);
    chain.def("ReadOneSupervisionFile", &ReadOneSupervisionFile);
    chain.def("ReadSupervisionFromFile", &ReadSupervisionFromFile);
    chain.def("ReadChainEgsFile", &ReadChainEgsFile);
    chain.def("ShuffleEgs", &ShuffleEgs);
    chain.def("MergeChainEgs", &MergeChainEgs);
    chain.def("ShiftEgsVector", &ShiftEgsVector);
    chain.def("GetFeaturesFromEgs", &GetFeaturesFromEgs);
    chain.def("GetFeaturesFromCompressedEgs", &GetFeaturesFromCompressedEgs);
    chain.def("GetIvectorsFromEgs", &GetIvectorsFromEgs);
    chain.def("GetFramesPerSequence", &GetFramesPerSequence);
    chain.def("GetSupervisionFromEgs", &GetSupervisionFromEgs);
    chain.def("PrintSupervisionInfoE2E", &PrintSupervisionInfoE2E);
}
#endif
