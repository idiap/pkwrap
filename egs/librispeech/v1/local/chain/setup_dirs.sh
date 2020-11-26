#!/bin/bash
# Copyright 2020 	Srikanth Madikeri (Idiap Research Institute)
# 
# Script to make data directory for librispeech 

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 kaldi-egs-dir"
    echo "$0: This script sets up data dirs that have been prepared with Kaldi."
    echo "The Kaldi folder would be something like $KALDI_ROOT/egs/librispeech/s5/, that"
    echo "contains data/, exp/ folders."
    exit 1
fi

kaldi_folder=$1


link() {
    # create only if source exists and dest doesn't
    src=$1
    dest=$2
    if [ -e $src -a ! -e $dest ]; then
        ln -r -s $src $dest
    elif [ ! -e $src ]; then
        echo "$0: $src does not exist. Skipping ..."
    else
        echo "$0: $dest exists. Skipping ..."
    fi
}

echo "$0: Copying data folders"
for data_name in \
    train_960_cleaned_sp \
    train_960_cleaned_sp_hires \
    dev_clean_hires \
    dev_other_hires \
    test_clean_hires \
    test_other_hires; 
do
    if [ -e data/$data_name ]; then
        echo "$0: data/$data_name already exists. Delete the folder to re-create it"
        echo "$0: Continuing to copy other folders..."
        continue
    fi
    echo "$0: Copying $data_name"
    utils/copy_data_dir.sh $kaldi_folder/data/$data_name data/$data_name
done

echo "$0: Linking lang folders"
for lang_name in lang_nosp_test_tgmed lang_nosp_test_tgsmall lang_test_tgmed; do
    link $kaldi_folder/data/$lang_name data/$lang_name
done

echo "$0: Linking exp folders"
[ ! -e exp ] && mkdir exp
# gmm dir
link $kaldi_folder/exp/tri6b_cleaned exp/tri6b_cleaned
# ali dir
link $kaldi_folder/exp/tri6b_cleaned_ali_train_960_cleaned_sp exp/tri6b_cleaned_ali_train_960_cleaned_sp
