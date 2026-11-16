.. _language_localizer_demo:

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

Direct download link from OSF:  https://osf.io/k4jp8/

Content
-------
.. nilearn_dataset_content:: language_localizer_demo

.. code-block::

    ├── access_data.py
    ├── CHANGES
    ├── dataset_description.json
    ├── derivatives
    │   ├── dataset_description.json
    │   ├── sub-01
    │   │   └── func
    │   │       ├── sub-01_task-languagelocalizer_desc-confounds_regressors.tsv
    │   │       ├── sub-01_task-languagelocalizer_desc-preproc_bold.json
    │   │       └── sub-01_task-languagelocalizer_desc-preproc_bold.nii.gz
    │   ├── ...
    │   └── sub-10
    │       └── func
    │           ├── sub-10_task-languagelocalizer_desc-confounds_regressors.tsv
    │           ├── sub-10_task-languagelocalizer_desc-preproc_bold.json
    │           └── sub-10_task-languagelocalizer_desc-preproc_bold.nii.gz
    ├── participants.tsv
    ├── README
    ├── sub-01
    │   └── func
    │       └── sub-01_task-languagelocalizer_events.tsv
    ├── ...
    └── sub-10
        └── func
            └── sub-10_task-languagelocalizer_events.tsv

License
-------
.. nilearn_dataset_license:: language_localizer_demo
