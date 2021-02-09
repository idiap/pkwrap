#!/usr/bin/env bash
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>,
# Srikanth Madikeri <srikanth.madikeri@idiap.ch>


set -e
cmd=run.pl
num_extra_lm_states=2000 
echo "$0 $@"  

. ./utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 tree_dir dir"
  exit 1
fi

if [ -f path.sh ]; then
  . path.sh
fi

tree_dir=$1
dir=$2

cp $tree_dir/tree $dir/tree
copy-transition-model $tree_dir/final.mdl $dir/0.trans_mdl

echo "$0: creating denominator FST"
$cmd $dir/log/make_den_fst.log \
  chain-make-den-fst $dir/tree $dir/0.trans_mdl $dir/phone_lm.fst \
  $dir/den.fst $dir/normalization.fst || exit 1;
