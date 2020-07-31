# README

This is a (yet another!) python wrapper for Kaldi. The main goal is to be able to train acoustic models in Pytorch
so that we can

1. use MMI cost function during training 
2. use NG-SGD for affine transformations, which enables multi-GPU training with SGE

## MOTIVATION 

The main motivation of this project is to run MMI training in Pytorch. The idea is to use existing 
functionality in Kaldi so that we don't have to re-implement anything. 

### Why not use the existing wrappers?

**Pykaldi** is good for exposing high-level functionalties. However, many things are still not possible (e.g.
loading NNet3 models and having access to the parameters of the model). Modifying Pykaldi requires us to have
custom versions of Kaldi and CLIF. Moreover, if one simply wanted to converting Tensors in GPU to Kaldi CuMatrix
was not efficient (the general route afaik would be Tensor GPU -> Tensor CPU -> Kaldi Matrix CPU -> Kaldi Matrix GPU).

**Pykaldi2** provides a version of LF-MMI training, which uses Pykaldi functions.

## INSTALLATION

**ATTENTION!!!**: Make sure that your pytorch environment is loaded before following the rest of the instructions.

1. Activate your pytorch environment
2. Compile Kaldi with `CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"`. This code has been tested to work with Kaldi ``6f329a62``.
3. Set ``KALDI_ROOT`` and optionally ``MKL_ROOT`` in the environment. Note: in the future this will be made easier with autoconf.
4. Run ``make``

### KNOWN ISSUES/COMMON PITFALLS

- the g++ version of pytorch, kaldi and pkwrap should match!

## USAGE


Before importing do check if Kaldi libraries used to compile the package are accessible by python.
Otherwise, it should be added to ``$LD_LIBRARY_PATH`` as follows

```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KALDI_ROOT/src/lib
```

Currently there is one functional recipe for the conventional LF-MMI training with 
pkwrap

- [Minilibrespeech recipe](egs/mini_librespeech/s5/README.md)

## REFERENCES & CITATION

Technical report coming soon!
