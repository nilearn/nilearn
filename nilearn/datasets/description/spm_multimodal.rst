.. _spm_multimodal:

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
.. nilearn_dataset_content:: spm_multimodal

References
----------

.. footbibliography::

License
-------
.. nilearn_dataset_license:: spm_multimodal
