# README

This folder is used to replicate results for minilibrespeech dataset.
This is useful to proptype systems and test code.

## HOW TO USE

Of course, we assume that the package has been compiled and installed. Otherwise,
see [here](../../../README.md)


### Setup directory

1. Start by creating links to Kaldi's ``utils``, ``steps`` folders

```
./make_links.sh
```

2. Define KALDI_ROOT variable for your environment. This is important, otherwise
``path.sh`` won't work

```
export KALDI_ROOT=<enter-path-here>
. path.sh
```

### Train model

The way the folder is arranged is such that one can run
``local/tuning/run_tdnn.py model_file``, where ``model_file`` is one of the files
in ``local/chian/tuning/models``.

Each model script is copied from the previous one, hoping that the WER gets
better as the versions progress. The scripts do all of the following (mimicing
Kaldi's ``train.py`` and lately, chain2's train.sh)

1. create ``lang_chain`` folder
2. Create supervision lattice by calling ``steps/align_fmllr_lats.sh``
3. Build tree by calling ``steps/nnet3/chain/build_tree.sh``
4. Create denominator fst with ``shutil/chain/make_den_fst.sh``
5. Create egs file susingKaldi script
6. Train model with pytorch
7. Decode

Script used for (6), the model file, gives a good idea of how to use pkwrap to train acoustic models in pytorch with MMI cost function.

To run one of the scripts, say 1b, simply run

```
local/chain/tuning/run_tdnn.py local/chain/tuning/models/1b.py
```