#!/bin/bash

# words='/idiap/temp/aprasad/kaldi/egs/fisher_swbd/s5/exp/chain/tdnn_7k_swbd_6eps_tree4k_lang7k/graph_fsh_sw1_tg/words.txt'
# trans_mdl='/idiap/temp/aprasad/kaldi/egs/fisher_swbd/s5/exp/chain/tdnn_7k_swbd_6eps_tree4k_lang7k/0.trans_mdl'
# graph='/idiap/temp/aprasad/kaldi/egs/fisher_swbd/s5/exp/chain/tdnn_7k_swbd_6eps_tree4k_lang7k/graph_fsh_sw1_tg/HCLG.fst'
# llf=$1
# latf=$2

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "$0 $@"  
. ./utils/parse_options.sh

if [ $# -ne 5 ]; then
    echo "Usage: $0 words trans_mdl graph output"
    echo ""
    echo "words: words.txt file in the lang folder"
    echo "trans_mdl: transition model"
    echo "graph: HCLG.fst file"
    echo "output: output file, usually filename of type lat.x.gz, where x is some number"
    exit 1
fi


words=$1
trans_mdl=$2
graph=$3

llf=$4
latf=$5

latgen-faster-mapped \
    --acoustic-scale=1 \
    --max-active=7000 \
    --min-active=200 \
    --beam=15.0 \
    --lattice-beam=8.0 \
    --allow-partial=true \
    --word-symbol-table=$words \
    $trans_mdl $graph ark,t:$llf ark:"| lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >$latf" 

#ark,t:test_woq_wfsf_words_qf16_v1.txt

