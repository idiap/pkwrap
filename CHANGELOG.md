# CHANGELOG

## [0.2.26] - 2021-02-02

- Librispeech 100h e2e recipe added
    - ``train.py`` supports normal and flatstart training now
    - ``ChainE2EModel`` class added to ``chain.py``
- ``xent_regularize`` is now actually a tunable parameter. Earlier it was fixed to be 0.1
- ``train_lfmmi_one_iter`` can now take an optimizer
- TDNN-F's ``constrain_orthonormal`` now uses ``add_`` instead of ``addmm_``
    - ``addmm_`` was not providing accurate results
- missing copyrights added
