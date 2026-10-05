.. _difumo_atlas:

DiFuMo atlas
============

Access
------
See :func:`nilearn.datasets.fetch_atlas_difumo`.

Notes
-----
We provide Dictionaries of Functional Modes “DiFuMo” (:footcite:t:`Dadi2020`)
that can serve as atlases to extract functional signals,
e.g to serve as IDPs, with different dimensionalities (64, 128, 256, 512, and 1024).
These modes are optimized to represent well raw :term:`BOLD` timeseries,
over a with range of experimental conditions.

* All atlases are available in .nii.gz format and sampled to :term:`MNI` space

Additionally, we provide meaningful names for these modes,
based on their anatomical location, to facilitate reporting of results.

* Anatomical names are available for each resolution in .csv

Direct download links from OSF:

    - 64: https://osf.io/pqu9r/download
    - 128: https://osf.io/wjvd5/download
    - 256: https://osf.io/3vrct/download
    - 512: https://osf.io/9b76y/download
    - 1024: https://osf.io/34792/download

Content
-------
    :'maps': Nifti images with the (probabilistic) region definitions
    :'labels': CSV file specifying the label information

References
----------
For more information about this dataset's structure:
https://inria.hal.science/hal-02904869

.. footbibliography::

Mensch, A., Mairal, J., Thirion, B., Varoquaux, G., 2018.
Stochastic Subsampling for Factorizing Huge Matrices.
IEEE Transactions on Signal Processing 66, 113–128.

Poldrack, R.A., Barch, D.M., Mitchell, J.P., et al., 2013.
Toward open sharing of task-based fMRI data:
the OpenfMRI project. Frontiers in neuroinformatics 7.

License
-------
usage is unrestricted for non-commercial research purposes.
