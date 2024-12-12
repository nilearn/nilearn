.. _msdl_atlas:

MSDL atlas
==========

Access
------
See :func:`nilearn.datasets.fetch_atlas_msdl`.

Notes
-----
Multi-subject Dictionary learning atlas.

Result maps of sparse :term:`Dictionary learning` based on :term:`resting-state` data.

This can be understand as a variant of :term:`ICA` based on the assumption
of sparsity rather than independence.

It can be downloaded at :footcite:t:`atlas_msdl`,
and cited using :footcite:t:`Varoquaux2011`.

See also :footcite:t:`Varoquaux2013` for more information.

Content
-------
    :'maps': Nifti images with the (probabilistic) region definitions
    :'labels': CSV file specifying the label information

References
----------

.. footbibliography::

For more information about this dataset's structure:
https://team.inria.fr/parietal/research/spatial_patterns/spatial-patterns-in-resting-state/

License
-------
usage is unrestricted for non-commercial research purposes.
