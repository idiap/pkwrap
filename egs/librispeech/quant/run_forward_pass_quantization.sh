#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
stage=0
affix=_7k_1a_sp

. path.sh
. cmd.sh
. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: "
  echo "  $0 [options] <expdir>"
  echo "e.g.:"
  echo " $0 $KALDI_ROOT/egs/librispeech/s5 "
  echo "Options"
  echo "   --affix=<prefix>     # Affix for the model dir. ex: _7k_1a_sp"
  exit 1;
fi

exp_dir=$1

model_dir=$exp_dir/exp/chain_cleaned/tdnn${affix}
graph_dir=$model_dir/graph_tgsmall
eval_data_dir=$exp_dir/data/test_clean_hires

feats='${eval_data_dir}/feats.scp'
filename=$(basename  "$eval_data_dir")

out_dir=decode_7k_${filename}_pytorch_qi8_1a

echo "i/p dir is $eval_data_dir & o/p dir is $out_dir"

# Check if output dir exists and create dir 
[ -d $out_dir ] || mkdir -p $out_dir/log

nj=50
# Quantize the model params and activations in Pytorch 
# and run the forward pass in Pytorch
if [ $stage -le 1 ]; then
    echo "$0: Computing log-likelihoods"
    num_outputs=$(tree-info $model_dir/tree | grep 'num-pdfs' | awk '{print $2}')
    $decode_cmd  JOB=1:$nj $out_dir/log/ll.JOB.log \
    python $DIR/local/chain/inference/run_tdnn_quant.py \
        -n $num_outputs -i $eval_data_dir/split${nj}/JOB/feats.scp -md $model_dir \
        -o $out_dir/ll.JOB.txt  -qm $out_dir/dequantized.mdl
fi

if [ $stage -le 2 ]; then
    echo "$0: Generating lattices from log-likelihoods"
    $decode_cmd  JOB=1:$nj $out_dir/log/lat.JOB.log \
    bash $DIR/local/shutil/decode/generate-lattices.sh \
        $graph_dir/words.txt $model_dir/0.trans_mdl \
        $graph_dir/HCLG.fst $out_dir/ll.JOB.txt $out_dir/lat.JOB.gz

    echo "$0: Computing the score (WER)"
    bash $DIR/local/shutil/decode/score.sh $eval_data_dir $graph_dir $out_dir

    # Find the best WER
    grep WER $out_dir/wer_* | best_wer.sh
fi

exit 0
