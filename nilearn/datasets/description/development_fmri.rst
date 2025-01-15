.. _development_dataset:

development fMRI dataset
========================

Access
------
See :func:`nilearn.datasets.fetch_development_fmri`.

Notes
-----
This movie-watching based functional MRI dataset is used for teaching how to use
machine learning to predict age from naturalistic stimuli (movie)
watching with Nilearn.

The dataset consists of 50 children (ages 3-13) and 33 young adults (ages
18-39). This dataset can be used to try to predict who are adults and
who are children.

The data is downsampled to 4mm resolution for convenience. The original
data is downloaded from OpenNeuro.

For full information about pre-processing steps on raw-fMRI data, have a look
at README at https://osf.io/wjtyq/

Full pre-processed data: https://osf.io/5hju4/files/

Raw data can be accessed from : https://openneuro.org/datasets/ds000228/versions/1.0.0

See :footcite:t:`Richardson2018`.

Content
-------
    :'func': functional MRI Nifti images (4D) per subject
    :'confounds': TSV file contain nuisance information per subject
    :'phenotypic': Phenotypic information for each subject such as age,
                   age group, gender, handedness.

References
----------

.. footbibliography::

License
-------
usage is unrestricted for non-commercial research purposes.
