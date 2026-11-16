.. _fiac:

fiac first level dataset
========================

Access
------
See :func:`nilearn.datasets.fetch_fiac_first_level`.

Notes
-----
Analysis from the Functional Imaging Analysis Contest (FIAC).

This is a block design experiment with a 2 X 2 experimental design
with the following factors:

- ``Sentence`` with 2 levels: Same (``SSt``) vs Different (``DSt``)
- ``Speaker``  with 2 levels: Same (``SSp``) vs Different (``DSp``)

giving the 4 following conditions:

- Same Sentence-Same Speaker (``SStSSp``)
- Same Sentence-Different Speakers (``SStDSp``)
- Different Sentences-Same Speaker (``DStSSp``)
- Different Sentences-Different Speakers (``DStDSp``)

The design also included a 5th condition
containing the first sentence pooled across all conditions.

Direct download link: https://nipy.org/data-packages/nipy-data-0.2.tar.gz.

For more details on the data, please see experiment 2 :footcite:t:`Dehaene2006`.

Content
-------
.. nilearn_dataset_content:: fiac

References
----------
.. footbibliography::

License
-------
.. nilearn_dataset_license:: fiac
