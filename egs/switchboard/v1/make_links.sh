#!/bin/bash

if [ $# -lt 1 ]; then
    if [ -z $KALDI_ROOT ]; then
        echo "$0: KALDI_ROOT is not defined and no argument found to the script".
        echo "Usage: $0 KALDI_ROOT"
        # we don't want to quit here because it might close the terminal
    else
        ln -s $KALDI_ROOT/egs/wsj/s5/{utils,steps} .
    fi
else
    KALDI_ROOT=$1
    ln -s $KALDI_ROOT/egs/wsj/s5/{utils,steps} .
fi

