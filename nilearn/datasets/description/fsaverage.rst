.. _fsaverage:

fsaverage template
==================

Access
------
See :func:`nilearn.datasets.fetch_surf_fsaverage`.

Content
-------
.. nilearn_dataset_content:: fsaverage

See Also
--------
See :func:`~nilearn.datasets.load_fsaverage` and
:func:`~nilearn.datasets.load_fsaverage_data`
to access fsaverage data as :class:`~nilearn.surface.SurfaceImage`.

Notes
-----
Fsaverage standard surface as distributed with Freesurfer (:footcite:t:`Fischl1999`).

============ ============================== ====================
Density      Number vertices per hemisphere Direct download link
============ ============================== ====================
fsaverage    163842                         https://osf.io/svf8k
fsaverage3   642                            https://osf.io/28uma
fsaverage4   2562                           https://osf.io/329dq
fsaverage5   10242
fsaverage6   40962                          https://osf.io/jzxyr
============ ============================== ====================

For fsaverage3 and fsaverage4,
the order of vertices was modified to match that of fsaverage5-7.

.. the figures are copied in the folder during doc build
   see doc/Makefile sym_links_datasets
.. figure:: test_surface_fs_data_left-fsaverage3.png
   :align: center

   data fsaverage3 left hemisphere

.. figure:: test_surface_fs_data_right-fsaverage3.png
   :align: center

   data fsaverage3 right hemisphere

.. figure:: test_surface_fs_data_left-fsaverage4.png
   :align: center

   data fsaverage4 left hemisphere

.. figure:: test_surface_fs_data_right-fsaverage4.png
   :align: center

   data fsaverage4 right hemisphere

.. figure:: test_surface_fs_data_left-fsaverage5.png
   :align: center

   data fsaverage5 left hemisphere

.. figure:: test_surface_fs_data_right-fsaverage5.png
   :align: center

   data fsaverage5 right hemisphere

References
----------
.. footbibliography::

License
-------
.. nilearn_dataset_license:: fsaverage
