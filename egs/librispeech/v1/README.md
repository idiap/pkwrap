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

local/chain/setup_dirs_960h.sh
local/chain/train.py
```

### RESULTS

| Model | config | dev-clean | 
| ---------- | ---------- | ---------- | 
| TDNNF 17 layers with dropout | ``tdnnf_17l``| 6.8 |
