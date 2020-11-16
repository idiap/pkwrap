#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 kaldi-egs-dir"
    echo "$0: This script sets up data dirs that have been prepared with Kaldi."
    echo "The Kaldi folder would be something like $KALDI_ROOT/egs/swbd/s5c/, that"
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

echo "$0: Copying data folders"
for data_name in train_nodup_sp train_nodup_sp_hires eval2000_hires; do
    if [ -e data/$data_name ]; then
        echo "$0: data/$data_name already exists. Delete the folder to re-create it"
        echo "$0: Continuing to copy other folders..."
        continue
    fi
    utils/copy_data_dir.sh $kaldi_folder/data/$data_name data/$data_name
done

echo "$0: Linking lang folders"
for lang_name in lang lang_chain; do
    link_if_exists $kaldi_folder/data/$lang_name data/$lang_name
done
[ ! -e exp ] && mkdir exp
link_if_exists $kaldi_folder/tri4 exp/tri4
link_if_exists $kaldi_folder/tri4_ali_nodup_sp exp/tri4_ali_nodup_sp

# TODO: copy i-vector extractor and features if they exist
