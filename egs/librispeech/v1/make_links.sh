#!/bin/bash

link_files () {
        ln -s $KALDI_ROOT/egs/wsj/s5/{utils,steps} .
        ln -r -s $KALDI_ROOT/egs/librispeech/s5/local/score.sh local/
}

if [ $# -lt 1 ]; then
    if [ -z $KALDI_ROOT ]; then
        echo "$0: KALDI_ROOT is not defined and no argument found to the script".
        echo "Usage: $0 KALDI_ROOT"
        # we don't want to quit here because it might close the terminal
    else
        link_files
    fi
else
    KALDI_ROOT=$1
    link_files
fi
