// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

// NOTE: ChainExampleMerger comes from Kaldi. While it has been slightly modified here, the original
// code is copyrighted as follows
// Copyright      2015    Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0

#include "chain.h"


kaldi::chain::ChainTrainingOptions CreateChainTrainingOptions(float l2, float oor, float lhc, float xentr) { 
    kaldi::chain::ChainTrainingOptions copts; 
    copts.l2_regularize = l2; 
    copts.out_of_range_regularize = oor; 
    copts.leaky_hmm_coefficient = lhc; 
    copts.xent_regularize = xentr; 
    return copts; 
} 

// Return with default values
kaldi::chain::ChainTrainingOptions CreateChainTrainingOptionsDefault() { 
    kaldi::chain::ChainTrainingOptions copts; 
    return copts; 
} 

kaldi::chain::DenominatorGraph LoadDenominatorGraph(std::string fst_path, int32 num_pdfs) { 
      fst::StdVectorFst den_fst;
      fst::ReadFstKaldi(fst_path, &den_fst);
      kaldi::chain::DenominatorGraph den_graph(den_fst, num_pdfs);
      return den_graph;
}

bool TestLoadDenominatorGraph(std::string fst_path, int32 num_pdfs) {
    auto fst = LoadDenominatorGraph(fst_path, num_pdfs);
    return true;
}

kaldi::chain::Supervision ReadOneSupervisionFile(std::string &file_name) {
    kaldi::nnet3::SequentialNnetChainExampleReader example_reader(file_name);
    auto key = example_reader.Key();
    auto eg = example_reader.Value();
    return eg.outputs[0].supervision;
}

kaldi::chain::Supervision ReadSupervisionFromFile(std::string &file_name) {
    /* kaldi::Input ip(file_name, true); */
    kaldi::chain::Supervision sup;
    kaldi::ReadKaldiObject<kaldi::chain::Supervision>(file_name, &sup);
    /* sup.Read(ip.Stream(), true); */
    return sup;
}

void PrintSupervisionInfoE2E(const kaldi::chain::Supervision &supervision) {
    std::cout << "Number of sequences: " << supervision.num_sequences << "\n";
    std::cout << "Number of frames per sequence " << supervision.frames_per_sequence << "\n";
}

// TODO, DONE: 6/8
// 7. Check derivative size
/* // 8. Check xent_deriv_size */
bool ComputeChainObjfAndDeriv(const kaldi::chain::ChainTrainingOptions &opts,
                              const kaldi::chain::DenominatorGraph &den_graph,
                              const kaldi::chain::Supervision &supervision,
                              torch::Tensor &nnet_output,
                              torch::Tensor &objf,
                              torch::Tensor &l2_term,
                              torch::Tensor &weight,
                              torch::Tensor &nnet_output_deriv,
                              torch::Tensor &xent_output_deriv) {
    if(den_graph.NumPdfs() != nnet_output.size(1)) {
        std::cout << "ERROR: In pkwrap's ComputeChainObjfAndDeriv, den_graph is not compatible with the output matrix" << std::endl;
        std::cout << "den_graph has " << den_graph.NumPdfs() << " pdfs and output has " << nnet_output.size(1)  << " columns" << std::endl;
        return false;
    }
    if (objf.size(0) != 1) {
        std::cout << "ERROR: In pkwrap's ComputeChainObjfAndDeriv, objf should be a scalar" << std::endl;
        return false;
    }
    if (weight.size(0) != 1) {
        std::cout << "ERROR: In pkwrap's ComputeChainObjfAndDeriv, weight should be a scalar" << std::endl;
        return false;
    }
    if (l2_term.size(0) != 1) {
        std::cout << "ERROR: In pkwrap's ComputeChainObjfAndDeriv, l2_term should be a scalar" << std::endl;
        return false;
    }

    kaldi::BaseFloat objf_;
    kaldi::BaseFloat l2_term_;
    kaldi::BaseFloat weight_;
    if(kaldi::CuDevice::Instantiate().Enabled()) {
        auto nnet_output_cumat = TensorToKaldiCuSubMatrix(nnet_output);
        kaldi::CuMatrix<BaseFloat> xent_output_deriv_cumat;
        auto nnet_output_deriv_cumat = TensorToKaldiCuSubMatrix(nnet_output_deriv);
        kaldi::chain::ComputeChainObjfAndDeriv(opts, den_graph, supervision, 
                nnet_output_cumat, &objf_, &l2_term_, &weight_, 
                &nnet_output_deriv_cumat,
                &xent_output_deriv_cumat);
        //xent_output_deriv_cumat.Scale(opts.xent_regularize);
        auto xent_output_torch = TensorToKaldiCuSubMatrix(xent_output_deriv);
        xent_output_torch.CopyFromMat(xent_output_deriv_cumat);
    }
	 
    objf[0]= objf_;
    l2_term[0] = l2_term_;
    weight[0] = weight_;
    return true;
}

std::vector<kaldi::nnet3::NnetChainExample> ReadChainEgsFile(std::string egs_file_path, int32 frame_shift) {
    kaldi::nnet3::SequentialNnetChainExampleReader example_reader(egs_file_path);
    std::vector<kaldi::nnet3::NnetChainExample> out;
    std::vector<std::string> exclude_names; 
    exclude_names.push_back(std::string("ivector"));
    int32 num_read = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      const std::string &key = example_reader.Key();
      kaldi::nnet3::NnetChainExample &eg = example_reader.Value();
      if(frame_shift>0) {
        kaldi::nnet3::ShiftChainExampleTimes(frame_shift, exclude_names, &eg);
      }
      out.push_back(eg);
    }
    return out;
}

void ShiftEgsVector(std::vector<kaldi::nnet3::NnetChainExample> &egs, int32 frame_shift) {
    std::vector<std::string> ex_names;
    for(auto iter = egs.begin(); iter != egs.end(); iter++) {
        kaldi::nnet3::NnetChainExample ex = *iter;
        kaldi::nnet3::ShiftChainExampleTimes(frame_shift, ex_names, &ex);
    }
}

void ShuffleEgs(std::vector<kaldi::nnet3::NnetChainExample> &egs) {
    std::random_shuffle(egs.begin(), egs.end());
}

std::vector<kaldi::nnet3::NnetChainExample> MergeChainEgs(std::vector<kaldi::nnet3::NnetChainExample> &egs, std::string minibatch_size) {
    ExampleMergingConfig merging_config("64");  // 64 is default minibatch size.
    merging_config.minibatch_size = minibatch_size;
    merging_config.ComputeDerived();
    std::vector<kaldi::nnet3::NnetChainExample> merged_eg;
    ChainExampleMerger merger(merging_config, merged_eg);
    for(auto iter = egs.begin(); iter != egs.end(); iter++) {
      merger.AcceptExample(new kaldi::nnet3::NnetChainExample(*iter));
    }
    merger.Finish();
    return merged_eg;
}

ChainExampleMerger::ChainExampleMerger(const ExampleMergingConfig &config, std::vector<kaldi::nnet3::NnetChainExample> &merged_eg):
    finished_(false), num_egs_written_(0),
    config_(config), merged_eg_(merged_eg) { }

void ChainExampleMerger::AcceptExample(kaldi::nnet3::NnetChainExample *eg) {
  if(finished_)
      return;
  // If an eg with the same structure as 'eg' is already a key in the
  // map, it won't be replaced, but if it's new it will be made
  // the key.  Also we remove the key before making the vector empty.
  // This way we ensure that the eg in the key is always the first
  // element of the vector.
  std::vector<kaldi::nnet3::NnetChainExample*> &vec = eg_to_egs_[eg];
  vec.push_back(eg);
  int32 eg_size = GetNnetChainExampleSize(*eg),
      num_available = vec.size();
  bool input_ended = false;
  int32 minibatch_size = config_.MinibatchSize(eg_size, num_available,
                                               input_ended);
  if (minibatch_size != 0) {  // we need to write out a merged eg.
    std::vector<kaldi::nnet3::NnetChainExample*> vec_copy(vec);
    eg_to_egs_.erase(eg);

    // MergeChainExamples() expects a vector of NnetChainExample, not of pointers,
    // so use swap to create that without doing any real work.
    std::vector<kaldi::nnet3::NnetChainExample> egs_to_merge(minibatch_size);
    for (int32 i = 0; i < minibatch_size; i++) {
      egs_to_merge[i].Swap(vec_copy[i]);
      delete vec_copy[i];  // we owned those pointers.
    }
    WriteMinibatch(&egs_to_merge);
  }
}

void ChainExampleMerger::WriteMinibatch(
    std::vector<kaldi::nnet3::NnetChainExample> *egs) {
  int32 eg_size = GetNnetChainExampleSize((*egs)[0]);
  kaldi::nnet3::NnetChainExampleStructureHasher eg_hasher;
  size_t structure_hash = eg_hasher((*egs)[0]);
  int32 minibatch_size = egs->size();
  stats_.WroteExample(eg_size, structure_hash, minibatch_size);
  kaldi::nnet3::NnetChainExample merged_eg;
  kaldi::nnet3::MergeChainExamples(config_.compress, egs, &merged_eg);
  merged_eg_.push_back(merged_eg);
}

void ChainExampleMerger::Finish() {
  if (finished_) return;  // already finished.
  finished_ = true;

  // we'll convert the map eg_to_egs_ to a vector of vectors to avoid
  // iterator invalidation problems.
  std::vector<std::vector<kaldi::nnet3::NnetChainExample*> > all_egs;
  all_egs.reserve(eg_to_egs_.size());

  MapType::iterator iter = eg_to_egs_.begin(), end = eg_to_egs_.end();
  for (; iter != end; ++iter)
    all_egs.push_back(iter->second);
  eg_to_egs_.clear();

  for (size_t i = 0; i < all_egs.size(); i++) {
    int32 minibatch_size;
    std::vector<kaldi::nnet3::NnetChainExample*> &vec = all_egs[i];
    int32 eg_size = GetNnetChainExampleSize(*(vec[0]));
    bool input_ended = true;
    while (!vec.empty() &&
           (minibatch_size = config_.MinibatchSize(eg_size, vec.size(),
                                                   input_ended)) != 0) {
      // MergeChainExamples() expects a vector of
      // NnetChainExample, not of pointers, so use swap to create that
      // without doing any real work.
      std::vector<kaldi::nnet3::NnetChainExample> egs_to_merge(minibatch_size);
      for (int32 i = 0; i < minibatch_size; i++) {
        egs_to_merge[i].Swap(vec[i]);
        delete vec[i];  // we owned those pointers.
      }
      vec.erase(vec.begin(), vec.begin() + minibatch_size);
      WriteMinibatch(&egs_to_merge);
    }
    if (!vec.empty()) {
      int32 eg_size = GetNnetChainExampleSize(*(vec[0]));
      kaldi::nnet3::NnetChainExampleStructureHasher eg_hasher;
      size_t structure_hash = eg_hasher(*(vec[0]));
      int32 num_discarded = vec.size();
      stats_.DiscardedExamples(eg_size, structure_hash, num_discarded);
      for (int32 i = 0; i < num_discarded; i++)
        delete vec[i];
      vec.clear();
    }
  }
  stats_.PrintStats();
}

int32 GetNnetChainExampleSize(const kaldi::nnet3::NnetChainExample &a) {
  int32 ans = 0;
  for (size_t i = 0; i < a.inputs.size(); i++) {
    int32 s = a.inputs[i].indexes.size();
    if (s > ans)
      ans = s;
  }
  for (size_t i = 0; i < a.outputs.size(); i++) {
    int32 s = a.outputs[i].indexes.size();
    if (s > ans)
      ans = s;
  }
  return ans;
}

int32 ExampleMergingConfig::IntSet::LargestValueInRange(int32 max_value) const {
  int32 ans = 0, num_ranges = ranges.size();
  for (int32 i = 0; i < num_ranges; i++) {
    int32 possible_ans = 0;
    if (max_value >= ranges[i].first) {
      if (max_value >= ranges[i].second)
        possible_ans = ranges[i].second;
      else
        possible_ans = max_value;
    }
    if (possible_ans > ans)
      ans = possible_ans;
  }
  return ans;
}

bool ExampleMergingConfig::ParseIntSet(const std::string &str,
                                       ExampleMergingConfig::IntSet *int_set) {
  std::vector<std::string> split_str;
  kaldi::SplitStringToVector(str, ",", false, &split_str);
  if (split_str.empty())
    return false;
  int_set->largest_size = 0;
  int_set->ranges.resize(split_str.size());
  for (size_t i = 0; i < split_str.size(); i++) {
    std::vector<int32> split_range;
    kaldi::SplitStringToIntegers(split_str[i], ":", false, &split_range);
    if (split_range.size() < 1 || split_range.size() > 2 ||
        split_range[0] > split_range.back() || split_range[0] <= 0)
      return false;
    int_set->ranges[i].first = split_range[0];
    int_set->ranges[i].second = split_range.back();
    int_set->largest_size = std::max<int32>(int_set->largest_size,
                                            split_range.back());
  }
  return true;
}

void ExampleMergingConfig::ComputeDerived() {
  if (measure_output_frames != "deprecated") {
    std::cout << "The --measure-output-frames option is deprecated "
        "and will be ignored.";
  }
  if (discard_partial_minibatches != "deprecated") {
    std::cout << "The --discard-partial-minibatches option is deprecated "
        "and will be ignored.";
  }
  std::vector<std::string> minibatch_size_split;
  kaldi::SplitStringToVector(minibatch_size, "/", false, &minibatch_size_split);
  if (minibatch_size_split.empty()) {
      std::cout << "Invalid option --minibatch-size=" << minibatch_size;
      exit(1);
  }

  rules.resize(minibatch_size_split.size());
  for (size_t i = 0; i < minibatch_size_split.size(); i++) {
    int32 &eg_size = rules[i].first;
    IntSet &int_set = rules[i].second;
    // 'this_rule' will be either something like "256" or like "64-128,256"
    // (but these two only if  minibatch_size_split.size() == 1, or something with
    // an example-size specified, like "256=64-128,256"
    std::string &this_rule = minibatch_size_split[i];
    if (this_rule.find('=') != std::string::npos) {
      std::vector<std::string> rule_split;  // split on '='
      kaldi::SplitStringToVector(this_rule, "=", false, &rule_split);
      if (rule_split.size() != 2) {
          std::cout << "Could not parse option --minibatch-size="
                  << minibatch_size;
          exit(1);
      }
      if (!kaldi::ConvertStringToInteger(rule_split[0], &eg_size) ||
          !ParseIntSet(rule_split[1], &int_set)) {
        std::cout << "Could not parse option --minibatch-size="
                  << minibatch_size;
          exit(1);
      }

    } else {
      if (minibatch_size_split.size() != 1) {
        std::cout << "Could not parse option --minibatch-size="
                  << minibatch_size << " (all rules must have "
                  << "eg-size specified if >1 rule)";
        exit(1);
      }
      if (!ParseIntSet(this_rule, &int_set)) {
        std::cout << "Could not parse option --minibatch-size="
                  << minibatch_size;
        exit(1);
      }
    }
  }
  {
    // check that no size is repeated.
    std::vector<int32> all_sizes(minibatch_size_split.size());
    for (size_t i = 0; i < minibatch_size_split.size(); i++)
      all_sizes[i] = rules[i].first;
    std::sort(all_sizes.begin(), all_sizes.end());
    if (!kaldi::IsSortedAndUniq(all_sizes)) {
      std::cout << "Invalid --minibatch-size=" << minibatch_size
                << " (repeated example-sizes)";
        exit(1);
    }
  }
}

int32 ExampleMergingConfig::MinibatchSize(int32 size_of_eg,
                                          int32 num_available_egs,
                                          bool input_ended) const {
  //KALDI_ASSERT(num_available_egs > 0 && size_of_eg > 0);
  int32 num_rules = rules.size();
  if (num_rules == 0) {
    std::cout << "You need to call ComputeDerived() before calling "
        "MinibatchSize().";
        exit(1);
  }
  int32 min_distance = std::numeric_limits<int32>::max(),
      closest_rule_index = 0;
  for (int32 i = 0; i < num_rules; i++) {
    int32 distance = std::abs(size_of_eg - rules[i].first);
    if (distance < min_distance) {
      min_distance = distance;
      closest_rule_index = i;
    }
  }
  if (!input_ended) {
    // until the input ends, we can only use the largest available
    // minibatch-size (otherwise, we could expect more later).
    int32 largest_size = rules[closest_rule_index].second.largest_size;
    if (largest_size <= num_available_egs)
      return largest_size;
    else
      return 0;
  } else {
    int32 s = rules[closest_rule_index].second.LargestValueInRange(
        num_available_egs);
    //KALDI_ASSERT(s <= num_available_egs);
    return s;
  }
}

torch::Tensor GetFeaturesFromEgs(const kaldi::nnet3::NnetChainExample &egs) {
    auto mat = egs.inputs[0].features.GetFullMatrix();
    int32 mb_size = egs.outputs[0].supervision.num_sequences;
    int32 feat_dim = mat.NumCols();
    return KaldiMatrixToTensor(mat).clone().detach().reshape({mb_size, -1, feat_dim});
}

torch::Tensor GetFeaturesFromCompressedEgs(kaldi::nnet3::NnetChainExample &egs) {
    if(egs.inputs.size() != 1) {
        std::cout << "We do not support the egs to have more than 1 input features" << std::endl;
        exit(1);
    }
    egs.inputs[0].features.Uncompress();
    auto mat = egs.inputs[0].features.GetFullMatrix();
    int32 mb_size = egs.outputs[0].supervision.num_sequences;
    int32 feat_dim = mat.NumCols();
    return KaldiMatrixToTensor(mat).clone().detach().reshape({mb_size, -1, feat_dim});
}

torch::Tensor GetIvectorsFromEgs(const kaldi::nnet3::NnetChainExample &egs) {
    for(int32 i=0; i<egs.inputs.size(); i++) {
      if (egs.inputs[i].name != "ivector") {
        continue;
      }
      auto mat = egs.inputs[1].features.GetFullMatrix();
      int32 mb_size = egs.outputs[0].supervision.num_sequences;
      int32 feat_dim = mat.NumCols();
      return KaldiMatrixToTensor(mat).clone().detach().reshape({mb_size, -1, feat_dim});
    }
    std::cerr << "GetIvectorsFromEgs: Cannot find ivectors in egs!";
    return torch::zeros({0});
}

int32 GetFramesPerSequence(const kaldi::nnet3::NnetChainExample &egs) {
  return (int) egs.outputs[0].supervision.frames_per_sequence;
}

kaldi::chain::Supervision GetSupervisionFromEgs(kaldi::nnet3::NnetChainExample &egs) {
    return egs.outputs[0].supervision;
}
