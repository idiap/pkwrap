#!/bin/bash
# Copyright 2020 	Srikanth Madikeri (Idiap Research Institute)

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 kaldi-egs-dir"
    echo "$0: This script sets up data dirs that have been prepared with Kaldi."
    echo "The Kaldi folder would be something like $KALDI_ROOT/egs/mini_librispeech/s5c/, that"
    echo "contains data/, exp/ folders."
    exit 1
fi

kaldi_folder=$1

link_if_exists() {
    # create only if source exists and dest doesn't
    src=$1
    dest=$2
    if [ -e $src -a ! -e $dest ]; then
        ln -r -s $src $dest
    # Give proper message
    elif [ ! -e $src ]; then
        echo "$0: $src does not exist. Skipping ..."
    else
        echo "$0: $dest exists. Skipping ..."
    fi
}

if [ ! -e utils  -o ! -e steps ]; then
    echo "$0: utils/ does not exists. Run make_links.sh before running this folder"
    exit 1
fi

echo "$0: Copying data folders"
for data_name in train_clean_5  train_clean_5_sp train_clean_5_sp_hires \
                dev_clean_2 dev_clean_2_hires;  do
    if [ -e data/$data_name ]; then
        echo "$0: data/$data_name already exists. Delete the folder to re-create it"
        echo "$0: Continuing to copy other folders..."
        continue
    fi
    utils/copy_data_dir.sh $kaldi_folder/data/$data_name data/$data_name
done

echo "$0: Linking lang folders"
for lang_name in lang lang_chain lang_nosp_test_tglarge lang_nosp_test_tgmed lang_nosp_test_tgsmall lang_test_tglarge lang_test_tgmed lang_test_tgsmall; do
    link_if_exists $kaldi_folder/data/$lang_name data/$lang_name
done
