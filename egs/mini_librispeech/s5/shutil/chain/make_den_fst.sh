#!/usr/bin/env bash
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>,
# Srikanth Madikeri <srikanth.madikeri@idiap.ch>


cmd=run.pl
num_extra_lm_states=2000

echo "$0 $@"  

. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
    echo "Usage: $0 tree_dir gmm_dir dir"
    exit 1
fi

if [ -f path.sh ]; then
    . path.sh
fi

tree_dir=$1
gmm_dir=$2
dir=$3

cp $tree_dir/tree $dir/tree
copy-transition-model $tree_dir/final.mdl $dir/0.trans_mdl

echo "$0: creating phone language-model"
$cmd $dir/log/make_phone_lm_default.log \
  chain-est-phone-lm --num-extra-lm-states=$num_extra_lm_states \
     "ark:gunzip -c $gmm_dir/ali.*.gz | ali-to-phones $gmm_dir/final.mdl ark:- ark:- |" \
     $dir/phone_lm.fst || exit 1

echo "$0: creating denominator FST"
$cmd $dir/log/make_den_fst.log \
    chain-make-den-fst $dir/tree $dir/0.trans_mdl $dir/phone_lm.fst \
   $dir/den.fst $dir/normalization.fst || exit 1;
