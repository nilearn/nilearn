.. _brainomics_maps:

Brainomics Localizer
====================

Access
------
See :func:`nilearn.datasets.fetch_localizer_contrasts`.

Notes
-----
A protocol that captures the cerebral bases of auditory and
visual perception, motor actions, reading, language comprehension
and mental calculation at an individual level. Individual functional
maps are reliable and quite precise.

You may cite :footcite:t:`Papadopoulos-Orfanos2017`
when using this dataset.

Scientific results obtained using this dataset are described
in :footcite:t:`Pinel2007`.

Content
-------
    :'func': Nifti images of the neural activity maps
    :'cmaps': Nifti images of contrast maps
    :'tmaps': Nifti images of corresponding t-maps
    :'masks': Structural images of the mask used for each subject.
    :'anats': Structural images of anatomy of each subject

References
----------

.. footbibliography::

For more information about this dataset's structure:
http://brainomics.cea.fr/localizer/

License
-------
usage is unrestricted for non-commercial research purposes.
