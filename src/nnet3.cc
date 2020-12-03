// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>
// Written by Amrutha Prasad <amrutha.prasad@idiap.ch>

#include "nnet3.h"
#include <typeinfo>
#include <tuple>
#include <map>
// from https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
// Under Creative Commons License
inline bool ends_with(std::string const & value, std::string const & ending)
{
        if (ending.size() > value.size()) return false;
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

// A wrapper to PreconditionDirections of OnlineNaturalGradient: it gets a pointer
// to the Tensor in the GPU and passed it to PreconditionDirections
kaldi::BaseFloat precondition_directions(kaldi::nnet3::OnlineNaturalGradient &state, torch::Tensor &grad) {
    kaldi::BaseFloat scale_;
    auto grad_cumat = TensorToKaldiCuSubMatrix(grad);
    state.PreconditionDirections(&grad_cumat, &scale_);
    return scale_;
}

std::vector<std::tuple<int, std::string,std::vector<torch::Tensor> > > GetNNet3Components(std::string model_path) {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    kaldi::nnet3::Nnet nnet;
    ReadKaldiObject(model_path, &nnet);
    auto nc = nnet.NumComponents();
    std::vector<std::tuple<int, std::string, std::vector<torch::Tensor> > > model;
    for(int32 i=0; i<nc; i++) {
        auto c = nnet.GetComponent(i);
        auto name = nnet.GetComponentName(i);
        if((name.rfind("tdnnf", 0) == 0) &&  (ends_with(name, "affine") || ends_with(name, "linear")) ){
            kaldi::nnet3::TdnnComponent* cg = (kaldi::nnet3::TdnnComponent*) c;
            torch::Tensor t_lp = KaldiCudaMatrixBaseToTensor(cg->LinearParams()).clone().detach();
            torch::Tensor t_bp = KaldiCudaVectorToTensor(cg->BiasParams()).clone().detach();
            if (cg->BiasParams().Dim()) {
                torch::Tensor t_bp = KaldiCudaVectorToTensor(cg->BiasParams()).clone().detach();
            }
            std::vector<torch::Tensor> params;
            params.push_back(t_lp);
            params.push_back(t_bp);
            model.push_back(std::make_tuple(i, name, params));
        }
        else if(ends_with(name, "affine") ) {
            kaldi::nnet3::NaturalGradientAffineComponent* cg = (kaldi::nnet3::NaturalGradientAffineComponent*) c;
            torch::Tensor t_lp = KaldiCudaMatrixToTensor(cg->LinearParams()).clone().detach();
            torch::Tensor t_bp = KaldiCudaVectorToTensor(cg->BiasParams()).clone().detach();
            std::vector<torch::Tensor> params;
            params.push_back(t_lp);
            params.push_back(t_bp);
            model.push_back(std::make_tuple(i, name, params));
        }
        else if (name.find("batchnorm") != std::string::npos) {
        // else if(ends_with(name, "batchnorm")) {
            kaldi::nnet3::BatchNormComponent* cg = (kaldi::nnet3::BatchNormComponent*) c;
            cg->SetTestMode(true);
            torch::Tensor scale = KaldiCudaVectorToTensor(cg->Scale()).clone().detach();
            torch::Tensor offset = KaldiCudaVectorToTensor(cg->Offset()).clone().detach();
            std::vector<torch::Tensor> params;
            params.push_back(scale);
            params.push_back(offset);
            model.push_back(std::make_tuple(i, name, params));
        }
        else if(ends_with(name, "lda")) {
            kaldi::nnet3::FixedAffineComponent* cg = (kaldi::nnet3::FixedAffineComponent*) c;
            torch::Tensor t_lp = KaldiCudaMatrixToTensor(cg->LinearParams()).clone().detach();
            torch::Tensor t_bp = KaldiCudaVectorToTensor(cg->BiasParams()).clone().detach();
            std::vector<torch::Tensor> params;
            params.push_back(t_lp);
            params.push_back(t_bp);
            model.push_back(std::make_tuple(i, name, params));
        }
        else if(ends_with(name, "prefinal-l") || ends_with(name, "prefinal-chain.linear")) {
            kaldi::nnet3::LinearComponent* cg = (kaldi::nnet3::LinearComponent*) c;
            torch::Tensor t_lp = KaldiCudaMatrixBaseToTensor(cg->Params()).clone().detach();
            // torch::Tensor t_bp = KaldiCudaVectorToTensor(cg->BiasParams()).clone().detach();
            std::vector<torch::Tensor> params;
            params.push_back(t_lp);
            // params.push_back(t_bp);
            model.push_back(std::make_tuple(i, name, params));
        }
    }
    return model;
}

void SaveNNet3Components(std::string model_path,
                         std::string new_model_path,
                         std::vector<std::tuple<int, std::string,std::vector<torch::Tensor> > > & new_params) {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    kaldi::nnet3::Nnet nnet;
    ReadKaldiObject(model_path, &nnet);
    auto nc = nnet.NumComponents();

    std::map<int, int> cmp2new_params;
    for(int j=0; j<new_params.size(); j++) {
        int i = std::get<0>(new_params[j]);
        cmp2new_params[i] = j;
    }
    for(int32 i=0; i<nc; i++) {
        auto c = nnet.GetComponent(i);
        auto name = nnet.GetComponentName(i);
        if (name =="prefinal-xent.affine") {
            break;
        }
        // don't change the component if it is not there in components_to_change
        auto key_value = cmp2new_params.find(i);
        if(key_value == cmp2new_params.end()) {
            continue;
        }
        int j = key_value->second;

        if((name.rfind("tdnnf", 0) == 0) &&  (ends_with(name, "affine") || ends_with(name, "linear")) ){
            auto c_params = new_params[j];
            if((std::get<1>(c_params)).compare(name)!=0) {
                std::cout << "ERROR: expected " << name << " but found " << std::get<1>(c_params) << std::endl;
                return;
            }
            auto params_array = std::get<2>(c_params);
            torch::Tensor lp_float32 = params_array[0];
            torch::Tensor bp_float32 = params_array[1];
            auto nr_lp = lp_float32.size(0), nc_lp = lp_float32.size(1),
                  nr_bp = bp_float32.size(0), nc_bp = bp_float32.size(1);
            torch::Tensor new_params_tensor = torch::zeros({nr_lp*nc_lp+nr_bp*nc_bp,1});
            new_params_tensor.narrow(0, 0, nr_lp*nc_lp).copy_(lp_float32.reshape({nr_lp*nc_lp,1}));
            new_params_tensor.narrow(0, nr_lp*nc_lp, nr_bp).copy_(bp_float32.reshape({nr_bp,1}));
            kaldi::Vector<kaldi::BaseFloat> vec(static_cast<int32>(new_params_tensor.size(0)));
            TensorToKaldiVector(new_params_tensor, vec);
            kaldi::nnet3::TdnnComponent* cg = (kaldi::nnet3::TdnnComponent*) c;
            cg->UnVectorize(vec);
        }
        else if(ends_with(name, "prefinal-l") || ends_with(name, "prefinal-chain.linear"))  {
            auto c_params = new_params[j];
            if((std::get<1>(c_params)).compare(name)!=0) {
                std::cout << "ERROR: expected " << name << " but found " << std::get<1>(c_params) << std::endl;
                return;
            }
            auto params_array = std::get<2>(c_params);
            torch::Tensor lp_float32 = params_array[0];
            auto nr_lp = lp_float32.size(0), nc_lp = lp_float32.size(1);
            torch::Tensor new_params_tensor = torch::zeros({nr_lp*nc_lp,1});
            new_params_tensor.narrow(0, 0, nr_lp*nc_lp).copy_(lp_float32.reshape({nr_lp*nc_lp,1}));
            kaldi::Vector<kaldi::BaseFloat> vec(static_cast<int32>(new_params_tensor.size(0)));
            TensorToKaldiVector(new_params_tensor, vec);
            kaldi::nnet3::LinearComponent* cg = (kaldi::nnet3::LinearComponent*) c;
            cg->UnVectorize(vec);
        }
        else if(ends_with(name, "affine") || ends_with(name, "linear")) {
            auto c_params = new_params[j];
            if((std::get<1>(c_params)).compare(name)!=0) {
                std::cout << "ERROR: expected " << name << " but found " << std::get<1>(c_params) << std::endl;
                return;
            }
            auto params_array = std::get<2>(c_params);
            torch::Tensor lp_float32 = params_array[0];
            torch::Tensor bp_float32 = params_array[1];
            auto nr_lp = lp_float32.size(0), nc_lp = lp_float32.size(1),
                  nr_bp = bp_float32.size(0), nc_bp = bp_float32.size(1);
            torch::Tensor new_params_tensor = torch::zeros({nr_lp*nc_lp+nr_bp*nc_bp,1});
            new_params_tensor.narrow(0, 0, nr_lp*nc_lp).copy_(lp_float32.reshape({nr_lp*nc_lp,1}));
            new_params_tensor.narrow(0, nr_lp*nc_lp, nr_bp).copy_(bp_float32.reshape({nr_bp,1}));
            kaldi::Vector<kaldi::BaseFloat> vec(static_cast<int32>(new_params_tensor.size(0)));
            TensorToKaldiVector(new_params_tensor, vec);
            kaldi::nnet3::NaturalGradientAffineComponent* cg = (kaldi::nnet3::NaturalGradientAffineComponent*) c;
            cg->UnVectorize(vec);
        }
        // else if()
    }
    WriteKaldiObject(nnet, new_model_path, true);
    std::cout <<"Succesfully wrote model!" << std::endl;
}

torch::Tensor LoadAffineTransform(std::string matrix_path) {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    kaldi::Matrix<kaldi::BaseFloat> mat;
    ReadKaldiObject(matrix_path, &mat);
    return KaldiMatrixToTensor(mat);
}
