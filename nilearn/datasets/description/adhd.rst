.. _adhd_dataset:

ADHD dataset
============

Access
------
See :func:`nilearn.datasets.fetch_adhd`.

Notes
-----
Part of the 1000 Functional Connectome Project. Phenotypic
information includes: diagnostic status, dimensional ADHD symptom measures,
age, sex, intelligence quotient (IQ) and lifetime medication status.
Preliminary quality control assessments (usable vs. questionable) based upon
visual timeseries inspection are included for all :term:`resting-state` :term:`fMRI` scans.

Includes preprocessed data from 40 participants.

Project was coordinated by Michael P. Milham.

See :footcite:t:`ADHDdataset`.

Content
-------
    :'func': Nifti images of the :term:`resting-state` data
    :'phenotypic': Explanations of preprocessing steps
    :'confounds': CSV files containing the nuisance variables

References
----------

.. footbibliography::

For more information about this dataset's structure:
http://fcon_1000.projects.nitrc.org/indi/adhd200/index.html

License
-------
usage is unrestricted for non-commercial research purposes.
