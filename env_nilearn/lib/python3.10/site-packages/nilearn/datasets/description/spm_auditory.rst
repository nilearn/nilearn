.. _spm_auditory_dataset:

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
    :'func': Paths to functional images
    :'anat': Path to anat image
    :'events': Path to events.tsv
    :'description': Data description

References
----------

.. footbibliography::

License
-------
The purpose was to explore new equipment and techniques.
As such it has not been formally written up,
and is freely available for personal education and evaluation purposes.
Those wishing to use these data for other purposes,
including published evaluation of methods,
should contact the methods group at the Wellcome Department of Cognitive Neurology.
