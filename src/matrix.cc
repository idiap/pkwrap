// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

#include "matrix.h"

torch::Tensor KaldiMatrixToTensor(kaldi::Matrix<kaldi::BaseFloat> &mat) {
    int32 nr = mat.NumRows(), nc = mat.NumCols();
    torch::Tensor t = torch::from_blob(mat.Data(), 
                                      {nr, nc}, 
                                      {mat.Stride(),1}, 
                                      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat));
    return t.clone().detach();
}


torch::Tensor KaldiCudaMatrixToTensor(kaldi::CuMatrix<kaldi::BaseFloat>& kmat) {
    if(kaldi::CuDevice::Instantiate().Enabled()) {
        // we are sure that kmat is in CUDA. otherwise it will segfault
        kaldi::BaseFloat* data_ptr = kmat.Data();
        torch::Tensor x = torch::from_blob((void*)data_ptr, 
                                            {kmat.NumRows(),kmat.NumCols()}, 
                                            {kmat.Stride(), 1}, 
                                            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
        return x;
    }
    else {
        // this is a simpler case. everything is on the host side
        kaldi::BaseFloat* data_ptr = kmat.Data();
        // we trust kaldi to have the same stride as pytorch
        // TODO: check kaldi's stride type 
        torch::Tensor x = torch::from_blob((void*)data_ptr, 
                                            {kmat.NumRows(),kmat.NumCols()}, 
                                            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU));
        return x;
    }
}

torch::Tensor KaldiCudaVectorToTensor(const kaldi::CuVector<kaldi::BaseFloat>& kvec) {
    if(kaldi::CuDevice::Instantiate().Enabled()) {
        // we are sure that kmat is in CUDA. otherwise it will segfault
        auto data_ptr = kvec.Data();
        torch::Tensor x = torch::from_blob((void*)data_ptr, 
                                            {1, kvec.Dim()}, 
                                            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
        return x;
    } 
    else {
        auto data_ptr = kvec.Data();
        torch::Tensor x = torch::from_blob((void*)data_ptr, 
                                            {1, kvec.Dim()}, 
                                            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU));
        return x;
    }
}

kaldi::CuSubMatrix<kaldi::BaseFloat> TensorToKaldiCuSubMatrix(torch::Tensor &t) {
    if(kaldi::CuDevice::Instantiate().Enabled()) {
        if(t.is_contiguous())  {
            kaldi::CuSubMatrix<kaldi::BaseFloat> cumat((kaldi::BaseFloat *)t.data_ptr(), t.size(0), t.size(1), t.size(1));
            // we can return this because cumat only holds a pointer
            return cumat;
        }
        else {
            std::cout << "ERROR: Cannot convert to CuSubMatrix because tensor is not contiguous" << std::endl;
            exit(1);
        }
    }
    else {
        std::cout << "ERROR: Cannot convert to CuSubMatrix because GPU is not enabled" << std::endl;
        exit(1);
    }
    // TODO: handle the else condition
}

// TODO: check if t is really one-dimensional
kaldi::CuSubVector<kaldi::BaseFloat> TensorToKaldiCuSubVector(torch::Tensor &t) {
    if(kaldi::CuDevice::Instantiate().Enabled()) {
        if(t.is_contiguous())  {
            kaldi::CuSubVector<kaldi::BaseFloat> kvec((kaldi::BaseFloat *)t.data_ptr(), t.size(0));
            return kvec;
        }
    }
    else {
        // we can't really do much b/c we don't have GPU
        // TODO: we need to make sure t is in CPU else raise an exception. right now the code
        // below is really not what the user expects to happen b/c if t is not already in CPU
        // we return an object that is pointing to a completely different tensor
        auto t_cpu = t.to(torch::kCPU);
        if(t.is_contiguous())  {
            kaldi::CuSubVector<kaldi::BaseFloat> cukvec((kaldi::BaseFloat *)t_cpu.data_ptr(), t_cpu.size(0));
            return cukvec;
        }
    }
}

torch::Tensor ReadKaldiMatrixFile(const std::string &file_name) {
    kaldi::Matrix<kaldi::BaseFloat> mat;
    kaldi::ReadKaldiObject(file_name, &mat);
    //return torch::zeros({2,2}).clone().detach();
    return KaldiMatrixToTensor(mat).clone().detach();
}

kaldi::Matrix<kaldi::BaseFloat> TensorToKaldiMatrix(torch::Tensor &t) {
    // TODO: check if the device is CPU
    if(t.is_contiguous()) {
        kaldi::SubMatrix<kaldi::BaseFloat> submat((kaldi::BaseFloat *)t.data_ptr(), t.size(0), t.size(1), t.size(1));
        // TODO: optimize this. too much copying happening.
        kaldi::Matrix<kaldi::BaseFloat> mat(submat);
        return mat;
    } else {
        kaldi::Matrix<kaldi::BaseFloat> mat(t.size(0), t.size(1));
        for(int32 r=0; r<t.size(0); r++){
            auto row = mat.RowData(r);
            for(int32 c=0; c<t.size(1); c++) {
                row[c] = t[r][c].item<float>();
            }
        }
        return mat;
    }
}
