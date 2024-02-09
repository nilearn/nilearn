.. _spm_multimodal_dataset:

SPM multimodal dataset
======================

Access
------
See :func:`nilearn.datasets.fetch_spm_multimodal_fmri`.

Notes
-----
The example shows the analysis of an :term:`SPM` dataset studying face perception.
The analysis is performed in native space.
Realignment parameters are provided with the input images,
but those have not been resampled to a common space.

The experimental paradigm is simple, with two conditions:
viewing a face image or a scrambled face image,
supposedly with the same low-level statistical properties,
to find face-specific responses.

See :footcite:t:`spm_multiface`.

Content
-------
    :'func1': Paths to functional images for run 1
    :'func2': Paths to functional images for run 2
    :'trials_ses1': Path to onsets file for run 1
    :'trials_ses2': Path to onsets file for run 2
    :'anat': Path to anat file

References
----------

.. footbibliography::

For details on the data, please see :footcite:t:`Henson2003`.

License
-------
unknown
