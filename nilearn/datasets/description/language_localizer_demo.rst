.. _language_localizer_dataset:

language localizer demo dataset
===============================

Access
------
See :func:`nilearn.datasets.fetch_language_localizer_demo_dataset`.

Notes
-----
10 subjects were scanned with fMRI during a "language localizer"
where they (covertly) read meaningful sentences (trial_type='language')
or strings of consonants (trial_type='string'),
presented one word at a time at the center of the screen (rapid serial visual presentation).

The functional images files (in derivatives/)
have been preprocessed (spatially realigned and normalized into the :term:`MNI` space).
Initially acquired with a :term:`voxel` size of 1.5x1.5x1.5mm,
they have been resampled to 4.5x4.5x4.5mm to save disk space.

https://osf.io/k4jp8/

Content
-------
    :'data_dir': Path to downloaded dataset.
    :'downloaded_files': Absolute paths of downloaded files on disk


References
----------


License
-------
ODC-BY-SA
