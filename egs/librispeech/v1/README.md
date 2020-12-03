# README

This folder contains librispeech recipes to train with pkwrap. 

## Chain recipe

The default config in ``configs/default`` trains a model
with 17 TDNNF layers. To run the recipe first setup the data folders

```
export KALDI_ROOT=...
. path.sh
# run this only if links are not already there
./make_links.sh

local/chain/train.py
```
