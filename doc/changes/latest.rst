.. currentmodule:: nilearn

.. include:: names.rst

0.11.1.dev
==========

Fixes
-----

- :bdg-dark:`Code` Allow using ``confounds`` and ``sample_mask`` via ``.fit_transform`` in :obj:`~nilearn.maskers.SurfaceLabelsMasker` (:gh:`4926` by `Himanshu Aggarwal`_).

Enhancements
------------

- :bdg-dark:`Code` Allow pathlike type for ``design matrix`` in :obj:`~nilearn.plotting.plot_design_matrix` and :obj:`~nilearn.plotting.plot_design_matrix_correlation`, as well as for ``model_event`` in :obj:`~nilearn.plotting.plot_event` (:gh:`4807` by `Rémi Gau`_).

- :bdg-dark:`Code` Ensure that ``design matrix`` and / or ``events`` can be pathlike objects in :meth:`nilearn.glm.first_level.FirstLevelModel.fit`, :meth:`nilearn.glm.second_level.SecondLevelModel.fit`, :meth:`~nilearn.glm.first_level.make_first_level_design_matrix`  (:gh:`4807` by `Rémi Gau`_).

- :bdg-dark:`Code` Implement :obj:`~nilearn.maskers.SurfaceMapsMasker` class to extract signals from surface maps (:gh:`4830` by `Himanshu Aggarwal`_).

- :bdg-primary:`Doc` Add a :ref:`page <meaning_difference>` in the user guide to explain GLM terminology across software (Nilearn, SPM, FSL) regarding the meaning of 'levels'  (:gh:`4287` by `Thiti Premrudeepreechacharn`_).

Changes
-------
