.. _spm_auditory:

SPM auditory dataset
====================

Access
------
See :func:`nilearn.datasets.fetch_spm_auditory`.

Notes
-----
These whole brain BOLD/EPI images were acquired on a modified 2T Siemens MAGNETOM Vision system.
Each acquisition consisted of 64 contiguous slices (64x64x64 3mm x 3mm x 3mm voxels).
Acquisition took 6.05 seconds, with the scan to scan repeat time (RT) set arbitrarily to 7 seconds.

96 acquisitions were made (RT= 7 seconds), in blocks of 6, giving 16 blocks of 42 seconds.
The condition for successive blocks alternated between rest and auditory stimulation,
starting with rest.

Auditory stimulation was bi-syllabic words presented binaurally at a rate of 60 per minute.

A structural image was also acquired.

.. warning::

    This dataset is a raw BIDS dataset.
    The data are in the native space
    and no spatial or temporal preprocessing has been performed.

This experiment was conducted by Geriant Rees
under the direction of Karl Friston and the FIL methods group.

See :footcite:t:`spm_auditory`.

Content
-------
.. nilearn_dataset_content:: spm_auditory

References
----------
.. footbibliography::

License
-------
.. nilearn_dataset_license:: spm_auditory
