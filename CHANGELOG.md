# CHANGELOG

## [0.2.27.1] - 2021-02-11

- Patch to enable compatibility with all recent Kaldi versions (since Apr 2020)

## [0.2.27] - 2021-02-03

- Fixes to minilibrispeech recipe
    - typos in tdnnf.py fixed
    - missing parameters in config file added
- train.py: regression fix to handle e2e option correctly

## [0.2.26] - 2021-02-02

- Librispeech 100h e2e recipe added
    - ``train.py`` supports normal and flatstart training now
    - ``ChainE2EModel`` class added to ``chain.py``
- ``xent_regularize`` is now actually a tunable parameter. Earlier it was fixed to be 0.1
- ``train_lfmmi_one_iter`` can now take an optimizer
- TDNN-F's ``constrain_orthonormal`` now uses ``add_`` instead of ``addmm_``
    - ``addmm_`` was not providing accurate results
- missing copyrights added
