The movie watching based brain development dataset (fMRI)


Notes
-----
This functional MRI dataset is used for teaching how to use
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

Content
-------
    :'func': functional MRI Nifti images (4D) per subject
    :'confounds': TSV file contain nuisance information per subject
    :'phenotypic': Phenotypic informaton for each subject such as age,
                   age group, gender, handedness.


References
----------
Please cite this paper if you are using this dataset:
Richardson, H., Lisandrelli, G., Riobueno-Naylor, A., & Saxe, R. (2018).
Development of the social brain from age three to twelve years.
Nature communications, 9(1), 1027.
https://www.nature.com/articles/s41467-018-03399-2

Licence: usage is unrestricted for non-commercial research purposes.
