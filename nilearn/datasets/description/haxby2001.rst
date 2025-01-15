.. _haxby_dataset:

Haxby dataset
=============

Access
------
See :func:`nilearn.datasets.fetch_haxby`.

Notes
-----
Results from a classical :term:`fMRI` study that investigated the differences between
the neural correlates of face versus object processing in the ventral visual
stream. Face and object stimuli showed widely distributed and overlapping
response patterns.

See :footcite:t:`Haxby2001`.

Content
-------
The "simple" dataset includes:
    :'func': Nifti images with bold data
    :'session_target': Text file containing run data
    :'mask': Nifti images with employed mask
    :'session': Text file with condition labels

The full dataset additionally includes
    :'anat': Nifti images with anatomical image
    :'func': Nifti images with bold data
    :'mask_vt': Nifti images with mask for ventral visual/temporal cortex
    :'mask_face': Nifti images with face-reponsive brain regions
    :'mask_house': Nifti images with house-reponsive brain regions
    :'mask_face_little': Spatially more constrained version of the above
    :'mask_house_little': Spatially more constrained version of the above

References
----------

.. footbibliography::

For more information see:
PyMVPA provides a tutorial using this dataset :
http://www.pymvpa.org/tutorial.html

More information about its structure :
http://dev.pymvpa.org/datadb/haxby2001.html


License
-------
unknown
