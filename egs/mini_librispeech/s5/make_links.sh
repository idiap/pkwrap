#!/bin/bash

if [ $# -lt 1 ]; then
    if [ -z $KALDI_ROOT ]; then
        echo "$0: KALDI_ROOT is not defined and no argument found to the script".
        exit 1
    fi
else
    KALDI_ROOT=$1
fi

ln -s $KALDI_ROOT/egs/wsj/s5/{utils,steps} .
ln -r -s $KALDI_ROOT/egs/wsj/s5/local/score.sh local/score.sh
