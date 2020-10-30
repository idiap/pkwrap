# Switchboard v1 setup

This folder contains code to train LF-MMI systems for Switchboard. Compared to the current version of the ``mini_librespeech``
recipe, we use a different approach to training. We use ``local/chain/train.py`` that takes a config file
to figure out the details of training. See ``config/default`` for a sample config. Thus, in order to train and test 
a model, pass the config file as follows:

```
local/chain/train.py --config configs/default
```

## Results with tri-gram LM

| System | config | SWBD | CALLHM | Average|
| --------- | -------- | ---------- |---------- |---------- |
| TDNN | ``default`` | 13.6 | 26.4 | 20.0 |
| + i-vectors | ``tdnn_kaldi_ivector`` | 12.0 | 23.0 | 17.5 |
