# README

**UPDATE (2021-02-03)**: See [changelog](./CHANGELOG.md) for the latest updates to the code

This is a (yet another!) python wrapper for Kaldi. The main goal is to be able to train acoustic models in Pytorch
so that we can

1. use MMI cost function during training
2. use NG-SGD for affine transformations, which enables multi-GPU training with SGE

------------------------------------
Table of Contents
------------------------------------

<!--ts-->
   * [Table of contents](#table-of-contents)
   * [Motivation](#motivation)
   * [Installation](#installation)
   * [Usage](#usage)
   * [Works based on Pkwrap](#works-based-on-pkwrap)
   * [References and citation](#references-and-citation)
<!--te-->

------------------------------------
Motivation
------------------------------------
The main motivation of this project is to run MMI training in Pytorch. The idea is to use existing
functionality in Kaldi so that we don't have to re-implement anything.

### Why not use the existing wrappers?

**Pykaldi** is good for exposing high-level functionalties. However, many things are still not possible (e.g.
loading NNet3 models and having access to the parameters of the model). Modifying Pykaldi requires us to have
custom versions of Kaldi and CLIF. Moreover, if one simply wanted to converting Tensors in GPU to Kaldi CuMatrix
was not efficient (the general route afaik would be Tensor GPU -> Tensor CPU -> Kaldi Matrix CPU -> Kaldi Matrix GPU).

**Pykaldi2** provides a version of LF-MMI training, which uses Pykaldi functions.

------------------------------------
Installation
------------------------------------
**ATTENTION!!!**: Make sure that your pytorch environment is loaded before following the rest of the instructions.

1. Activate your pytorch environment
2. Compile Kaldi with `CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"`.
3. Set ``KALDI_ROOT`` and optionally ``MKL_ROOT`` in the environment. Note: in the future this will be made easier with autoconf.
4. Run ``make``

### Known Issues / Common Pitfalls

- the g++ version of pytorch, kaldi and pkwrap should match!

------------------------------------
Usage
------------------------------------
Before importing do check if Kaldi libraries used to compile the package are accessible by python.
Otherwise, it should be added to ``$LD_LIBRARY_PATH`` as follows

```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KALDI_ROOT/src/lib
```

Currently there are recipes for conventional LF-MMI training with
pkwrap

- [Minilibrespeech recipe](egs/mini_librespeech/s5/README.md)
- [Switchboard](egs/switchboard/v1/README.md)
- [Librispeech](egs/librispeech/v1/README.md)

For flatstart LF-MMI training there is a recipe in [Librispeech](egs/librispeech/v1/README.md).

For experiments related to quantization of acoustic models trained in Kaldi see ``egs/librispeech/quant`` in ``load_kaldi_models`` branch.

------------------------------------
Works based on Pkwrap
------------------------------------
Following list of works are based on this repository and might be of interest:

1.Lattice-Free MMI Adaptation Of Self-Supervised Pretrained Acoustic Models
  - Paper: [https://arxiv.org/abs/2012.14252](https://arxiv.org/abs/2012.14252)
  - Code: [Adaptation of Pretrained Acoustic Models (APAM)](https://github.com/idiap/apam)

------------------------------------
References and Citation
------------------------------------
The technical report is now available [here](https://arxiv.org/abs/2010.03466). The report can
be cited as in the following bibtex example:

```
@article{madikeri2020pkwrap,
  title={Pkwrap: a PyTorch Package for LF-MMI Training of Acoustic Models},
  author={Madikeri, Srikanth and Tong, Sibo and Zuluaga-Gomez, Juan and Vyas, Apoorv and Motlicek, Petr and Bourlard, Herv{\'e}},
  journal={arXiv preprint arXiv:2010.03466},
  year={2020}
}
```
