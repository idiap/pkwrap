#!/usr/bin/env bash
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Apoorv Vyas <apoorv.vyas@idiap.ch>,
# Srikanth Madikeri <srikanth.madikeri@idiap.ch>


if [ "$#" -ne 2 ]; then
    echo "Usage: $0 old_lang new_lang"
    echo ""
    exit 1
fi
old_lang=$1
lang=$2

echo "$0 $@"  

if [ -d $lang ]; then
    if [ $lang/L.fst -nt $old_lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than $old_lang ..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
else
    # put an extra / to make sure soft links are copied too
    cp -r $old_lang/ $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi
