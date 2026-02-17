.. _spm_multimodal_dataset:

SPM multimodal dataset
======================

Access
------
See :func:`nilearn.datasets.fetch_spm_multimodal_fmri`.

Notes
-----
The example shows the analysis of an :term:`SPM` dataset studying face perception.
The images are in native space: they have not been resampled to a common space.

The experimental paradigm is simple, with two conditions:
viewing a face image or a scrambled face image,
supposedly with the same low-level statistical properties,
to find face-specific responses.

Images were acquired with a repetition time of 2 seconds.

The full dataset as well as its fmriprep derivatives are available
on `openneuro <https://openneuro.org/datasets/ds000117>`_.

See :footcite:t:`spm_multiface`.

Content
-------
:'func1': Paths to functional images for run 1 (list of 3D images)
:'func2': Paths to functional images for run 2 (list of 3D images)
:'events1': Path to onsets TSV file for run 1
:'trials_ses1': Path to .mat file containing onsets for run 1
:'events2': Path to onsets TSV file for run 2
:'trials_ses1': Path to .mat file containing onsets for run 2
:'anat': Path to anat file

References
----------

.. footbibliography::

License
-------
unknown
