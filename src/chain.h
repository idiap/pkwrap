// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

#ifndef PKWRAP_CHAIN_H_
#define PKWRAP_CHAIN_H_
#include "common.h"
#include "matrix.h"
#include<set>
#include "util/kaldi-io.h"
#include "util/text-utils.h"
#include "util/stl-utils.h"
#include "nnet3/nnet-chain-example.h"
#include "chain/chain-denominator.h"
#include "chain/chain-training.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-example-utils.h"

kaldi::chain::ChainTrainingOptions CreateChainTrainingOptions(float l2, float oor, float lhc, float xentr);
kaldi::chain::ChainTrainingOptions CreateChainTrainingOptionsDefault();
kaldi::chain::DenominatorGraph LoadDenominatorGraph(std::string fst_path, int32 num_pdfs);
bool TestLoadDenominatorGraph(std::string fst_path, int32 num_pdfs);
kaldi::chain::Supervision ReadOneSupervisionFile(std::string &file_name);
kaldi::chain::Supervision ReadSupervisionFromFile(std::string &file_name);
void PrintSupervisionInfoE2E(const kaldi::chain::Supervision &supervision);
bool ComputeChainObjfAndDeriv(const kaldi::chain::ChainTrainingOptions &opts,
                              const kaldi::chain::DenominatorGraph &den_graph,
                              const kaldi::chain::Supervision &supervision,
                              torch::Tensor &nnet_output,
                              torch::Tensor &objf,
                              torch::Tensor &l2_term,
                              torch::Tensor &weight,
                              torch::Tensor &nnet_output_deriv,
                              torch::Tensor &xent_output_deriv);
std::vector<kaldi::nnet3::NnetChainExample> ReadChainEgsFile(std::string egs_file_path, int32 frame_shift);
void ShiftEgsVector(std::vector<kaldi::nnet3::NnetChainExample> &egs, int32 frame_shift);
void ShuffleEgs(std::vector<kaldi::nnet3::NnetChainExample> &egs);
std::vector<kaldi::nnet3::NnetChainExample> MergeChainEgs(std::vector<kaldi::nnet3::NnetChainExample> &egs, std::string minibatch_size);
//std::vector<kaldi::nnet3::NnetChainExample> MergeChainEgs(std::vector<kaldi::nnet3::NnetChainExample> &egs, std::string minibatch_size);

class ExampleMergingConfig {
public:
  // The following configuration values are registered on the command line.
  bool compress;
  std::string measure_output_frames;  // for back-compatibility, not used.
  std::string minibatch_size;
  std::string discard_partial_minibatches;   // for back-compatibility, not used.
  bool multilingual_eg; // add language information as a Query (e.g. ?lang=query) to the merged egs's name

  ExampleMergingConfig(const char *default_minibatch_size = "256"):
      compress(false),
      measure_output_frames("deprecated"),
      minibatch_size(default_minibatch_size),
      discard_partial_minibatches("deprecated"),
      multilingual_eg(false)
      { }

  // this function computes the derived (private) parameters; it must be called
  // after the command-line parameters are read and before MinibatchSize() is
  // called.
  void ComputeDerived();

  /// This function tells you what minibatch size should be used for this eg.

  ///  @param [in] size_of_eg   The "size" of the eg, as obtained by
  ///                           GetNnetExampleSize() or a similar function (up
  ///                           to the caller).
  ///  @param [in] num_available_egs   The number of egs of this size that are
  ///                            currently available; should be >0.  The
  ///                            value returned will be <= this value, possibly
  ///                            zero.
  ///  @param [in] input_ended   True if the input has ended, false otherwise.
  ///                            This is important because before the input has
  ///                            ended, we will only batch egs into the largest
  ///                            possible minibatch size among the range allowed
  ///                            for that size of eg.
  ///  @return                   Returns the minibatch size to use in this
  ///                            situation, as specified by the configuration.
  int32 MinibatchSize(int32 size_of_eg,
                      int32 num_available_egs,
                      bool input_ended) const;


 private:
  // struct IntSet is a representation of something like 16:32,64, which is a
  // nonempty list of either positive integers or ranges of positive integers.
  // Conceptually it represents a set of positive integers.
  struct IntSet {
    // largest_size is the largest integer in any of the ranges (64 in this
    // example).
    int32 largest_size;
    // e.g. would contain ((16,32), (64,64)) in this example.
    std::vector<std::pair<int32, int32> > ranges;
    // Returns the largest value in any range (i.e. in the set of
    // integers that this struct represents), that is <= max_value,
    // or 0 if there is no value in any range that is <= max_value.
    // In this example, this function would return the following:
    // 128->64, 64->64, 63->32, 31->31, 16->16, 15->0, 0->0
    int32 LargestValueInRange(int32 max_value) const;
  };
  static bool ParseIntSet(const std::string &str, IntSet *int_set);

  // 'rules' is derived from the configuration values above by ComputeDerived(),
  // and are not set directly on the command line.  'rules' is a list of pairs
  // (eg-size, int-set-of-minibatch-sizes); If no explicit eg-sizes were
  // specified on the command line (i.e. there was no '=' sign in the
  // --minibatch-size option), then we just set the int32 to 0.
  std::vector<std::pair<int32, IntSet> > rules;
};

class ChainExampleMerger {
 public:
  ChainExampleMerger(const ExampleMergingConfig &config,
          std::vector<kaldi::nnet3::NnetChainExample> &merged_eg);
  

  // This function accepts an example, and if possible, writes a merged example
  // out.  The ownership of the pointer 'a' is transferred to this class when
  // you call this function.
  void AcceptExample(kaldi::nnet3::NnetChainExample *a);

  // This function announces to the class that the input has finished, so it
  // should flush out any smaller-sized minibatches, as dictated by the config.
  // This will be called in the destructor, but you can call it explicitly when
  // all the input is done if you want to; it won't repeat anything if called
  // twice.  It also prints the stats.
  void Finish();

  // returns a suitable exit status for a program.
  int32 ExitStatus() { Finish(); return (num_egs_written_ > 0 ? 0 : 1); }

  ~ChainExampleMerger() { Finish(); };
 private:
  // called by Finish() and AcceptExample().  Merges, updates the stats, and
  // writes.  The 'egs' is non-const only because the egs are temporarily
  // changed inside MergeChainEgs.  The pointer 'egs' is still owned
  // by the caller.
  void WriteMinibatch(std::vector<kaldi::nnet3::NnetChainExample> *egs);

  bool finished_;
  int32 num_egs_written_;
  std::vector<kaldi::nnet3::NnetChainExample>& merged_eg_;
  const ExampleMergingConfig &config_;
  kaldi::nnet3::ExampleMergingStats stats_;

  // Note: the "key" into the egs is the first element of the vector.
  typedef unordered_map<kaldi::nnet3::NnetChainExample*,
                        std::vector<kaldi::nnet3::NnetChainExample*>,
                        kaldi::nnet3::NnetChainExampleStructureHasher,
                        kaldi::nnet3::NnetChainExampleStructureCompare> MapType;
MapType eg_to_egs_;
};

int32 GetNnetChainExampleSize(const kaldi::nnet3::NnetChainExample &a);
torch::Tensor GetFeaturesFromEgs(const kaldi::nnet3::NnetChainExample &egs);
torch::Tensor GetFeaturesFromCompressedEgs(kaldi::nnet3::NnetChainExample &egs);
torch::Tensor GetIvectorsFromEgs(const kaldi::nnet3::NnetChainExample &egs);
int32 GetFramesPerSequence(const kaldi::nnet3::NnetChainExample &egs);
kaldi::chain::Supervision GetSupervisionFromEgs(kaldi::nnet3::NnetChainExample &egs);
#endif
