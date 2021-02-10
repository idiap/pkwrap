# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

# Welcome to pkwrap 
# This tool allows to train acoustic models with Pytorch using Kaldi's LF-MMI cost
# function
# Run python setup.py install to build and install the library.

import os
import sys
from setuptools import setup, find_packages
from torch.utils import cpp_extension

import torch
pytorch_version = torch.__version__
pytorch_major_ver, pytorch_min_ver = list(map(int, pytorch_version.split('.')[:2]))
if pytorch_major_ver != 1 or pytorch_min_ver > 7:
    sys.stderr.write("We support pytorch version until 1.7 only\n")
    quit(1)

KALDI_ROOT = os.getenv('KALDI_ROOT')
if not KALDI_ROOT:
    sys.stderr.write('ERROR: KALDI_ROOT variable is not defined or empty')
    quit(1)
KALDI_LIB_DIR = os.path.join(KALDI_ROOT, 'src', 'lib')

PACKAGE_NAME = 'pkwrap'
EXTENSION_NAME = '_pkwrap'
SRC_FILES = ['src/pkwrap-main.cc',
             'src/matrix.cc',
             'src/chain.cc',
             'src/nnet3.cc',
            ]
EXTRA_COMPILE_ARGS = {
    'cxx':[ '-I{}/src'.format(KALDI_ROOT),
            '-I{}/tools/openfst/include'.format(KALDI_ROOT),
            '-m64',
            '-msse',
            '-msse2',
            '-DHAVE_CUDA=1',
            '-Wno-sign-compare', # this helped clear openfst related compilation errors
            # additional flags used by Kaldi, but not sure we need this
            '-Wno-deprecated-declarations',
            '-Winit-self',
            '-DKALDI_DOUBLEPRECISION=0',
            '-DHAVE_EXECINFO_H=1',
            '-w', # potentially dangerous, but less annoying
          ]
    }
LIBRARIES = [
    "kaldi-base", "kaldi-matrix", "kaldi-util", "kaldi-cudamatrix",
    "kaldi-decoder", "kaldi-lat", "kaldi-gmm", "kaldi-hmm", "kaldi-tree",
    "kaldi-transform", "kaldi-chain", "kaldi-fstext", "kaldi-nnet3"
]
LIBRARY_DIRS = [KALDI_LIB_DIR]
MKL_ROOT = os.getenv('MKL_ROOT')
MKL_LIB_DIR = ''
if MKL_ROOT:
    MKL_LIB_DIR = os.path.join(MKL_ROOT, 'lib')
    LIBRARY_DIRS.append(MKL_LIB_DIR)
    EXTRA_COMPILE_ARGS['cxx'] += ['-I{}/include'.format(MKL_ROOT)]
    LIBRARIES += ["mkl_intel_lp64", "mkl_core", "mkl_sequential"]

with open('./AUTHORS') as ipf:
    AUTHORS = [ln.strip() for ln in ipf]
    AUTHOR_STR = ','.join(AUTHORS)

LICENSE = 'Apache 2.0'
VERSION = '0.2.27.2'

setup(name=PACKAGE_NAME,
      version=VERSION,
      author=AUTHOR_STR,
      license=LICENSE,
      packages = find_packages(),
      ext_modules=[cpp_extension.CppExtension(EXTENSION_NAME, SRC_FILES,
                                              language="c++",
                                              extra_compile_args=EXTRA_COMPILE_ARGS,
                                              libraries=LIBRARIES,
                                              library_dirs=LIBRARY_DIRS,
                                             )
                  ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
