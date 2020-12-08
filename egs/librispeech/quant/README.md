# Quantization setup

The details for the recipe provided here can be found in [this](http://arxiv.org/abs/2006.09054) paper, 
titled:

## Quantization of Acoustic Model Parameters in Automatic Speech Recognition Framework

This folder contains the recipe to load Kaldi AM in PyTorch and 
- run inference in PyTorch
- run post training quantization with custom quantization
- run post training quantization with PyTorch functions
- run quantization aware training (QAT) using PyTorch functionalities

### Example scripts of inference of Kaldi models in PyTorch w/ and w/o quantization:
1. In order to run inference in PyTroch, run
```
    bash run_forward_pass.sh <exp_foler>
    ex: bash run_forward_pass.sh $KALDI_ROOT/egs/librispeech/s5
```
The above script, uses `local/chain/inference/run_tdnn.py` to load the Kaldi AM parameters and the MFCC features to compute the log-likelihoods.
These log-likelihoods are then passed to `local/shutil/decode/generate-lattices.sh` to generate the lattice and compute the WER.

2. The recipe to run custom post-training quantization and inference is:
```
    bash run_forward_pass_quantization.sh <exp_foler>
    ex: bash run_forward_pass_quantization.sh $KALDI_ROOT/egs/librispeech/s5
```

3. The recipe to use PyTorch's post-training quantization functions for inference is:
```
    bash run_forward_pass_quantization_w_torch.sh <exp_foler>
    ex: bash run_forward_pass_quantization_w_torch.sh $KALDI_ROOT/egs/librispeech/s5
```

### Example scripts training AM models in PyTorch w/ and w/o QAT:
This folder also contains recipe to load a trained Kaldi AM and run fine-tuning or run QAT using PyTorch modules.

1. To run fine tuning of a trained AM, the script used:
```
    python local/chain/run_tdnn.py --stage 5 --lr-initial 1e-5 --lr-final 5e-6 \
        --graph-dir exp/chain_cleaned/tdnn_7k_1a_sp/graph_tgsmall
```

2. To run QAT of a trained AM, the script used:
```
    python local/chain/run_tdnn_qat.py --stage 5 --lr-initial 1e-5 --lr-final 5e-6 \
        --graph-dir exp/chain_cleaned/tdnn_7k_1a_pt_sp/graph_tgsmall  \
        --kaldi-model-dir exp/chain_cleaned/tdnn_7k_1a_sp
```

**NOTE:** 
1. The egs_dir, graph_dir, den.fst, normalization.fst, tree can be linked from the previously trained <Kaldi model_dir> to the <PyTorch model_dir>
2. In order to use PyTorch quantization and QAT modules, QNNPACK and XNNPACK must be installed.
To install them, clone [QNNPACK](https://github.com/pytorch/QNNPACK) and [XNNPACK](https://github.com/google/XNNPACK) go to resp dir and run
    ``` 
        git clone https://github.com/pytorch/QNNPACK.git
        git clone https://github.com/google/XNNPACK.git
        scripts/build-local.sh
    ```

