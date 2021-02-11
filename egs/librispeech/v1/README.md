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

## E2E Chain recipe

To run the 100h setup, follow the steps below

1. Change the path to librispeech database in ``conf/local.conf``

2. Prepare data directories with

```
local/chain/e2e/prepare_data.sh
```

Note that in the future we expect the above script to be done by pkwrap library.

3. Train (and test ``dev_clean``) using the configuration below

```
local/chain/train.py --stage 4 --conf configs/tdnnf_e2e
```

The model is stored in ``exp/chain/e2e_tdnnf/``. Once the entire script finishes successfully, the expected WER on ``dev_clean`` is

```
# cat exp/chain/e2e_tdnnf/decode_dev_clean_fbank_hires_iterfinal/wer_* | fgrep WER | sort -k2,2 -g | head -1
%WER 7.78 [ 4233 / 54402, 460 ins, 502 del, 3271 sub ]
```

4. To rescore with a 4gram LM, run:

```
. cmd.sh
steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_lp_test_{tgsmall,fglarge} data/dev_clean_fbank_hires/ exp/chain/e2e_tdnnf/decode_dev_clean_fbank_hires_iterfinal/ exp/chain/e2e_tdnnf/decode_dev_clean_fbank_hires_iterfinal_fg
```

The expected WER is 
```
%WER 5.11 [ 2779 / 54402, 404 ins, 248 del, 2127 sub ]
```
