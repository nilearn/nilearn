.. _miyawaki_dataset:

Miyawaki 2008 dataset
=====================

Access
------
See :func:`nilearn.datasets.fetch_miyawaki2008`.

Notes
-----
Collection of result images from MVPA investigation of the human visual systems.

This :term:`fMRI` study reconstructed visual images by combining local
image bases of multiple scales. Their :term:`contrasts<contrast>` were independently
decoded from :term:`fMRI` activity by selecting important :term:`voxels<voxel>` and
capitalizing on their correlation structure.

See :footcite:t:`Miyawaki2008`.

Content
-------

    :'label': Paths to text files containing run and target data
    :'func': Paths to nifti files with :term:`BOLD` data
    :'mask': Path to general mask nifti that defines target volume in visual cortex
    :'mask_roi': List of paths to images with specific data
                 ('RH' for right hemisphere, 'LH' for left hemisphere, 'Vxxx' denote visual areas)

References
----------

.. footbibliography::

For more information on this dataset's structure, see
https://bicr.atr.jp/dni/en/downloads/fmri-data-set-for-visual-image-reconstruction/


License
-------
unknown
