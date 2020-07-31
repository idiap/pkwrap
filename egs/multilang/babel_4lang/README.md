# README

This recipe trains multilingual model for 4 languages

1. Tagalog
2. Swahili
3. Zulu
4. Turkish

The idea is to replicate the results of multilingual modelling in Kaldi and use
that as a starting point for further improvements.

In this recipe, we assume that we already have access to egs folders.
To train the model simply run

```
# set the model file. 
model_file=local/chain/tuning/models/1c.py
# useful to separate the model folders with affixes
affix=_1c

# train: we start from stage 6 because the examples are already
# there
local/chain/run_tdnn.py \
    --stage 6 \
    --train-stage 70 \
    --left-context 13 --right-context 13 \
    --num-jobs-initial 2 \
    --num-jobs-final 6 \
    --num-epochs 4 \
    --affix $affix \
    --train-set train \
    --gmm tri5 \
    --chunk-width 150,100,50,20,10 \
    --lr-initial 0.0005 \
    --lr-final 0.00005 \
    $model_file  
```

### Important

- The minibatch sizes are already set in the script. In the future, we will make it an argument to the training command
- It is critical to make sure that the model contexts are correct. This changes with every model configuration.
