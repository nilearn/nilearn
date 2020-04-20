.. _nistats_migration:

===================================================
A Quick Guide to Migrating Nistats Code to Nilearn
===================================================

A quick guide to changing your Nistats imports to work in Nilearn


Datasets
========
Imports from
`nistats.datasets <https://nistats.github.io/modules/reference.html#module-nistats.datasets>`_
are in :mod:`nilearn.datasets`

Hemodynamic Models
==================
Imports from
`nilearn.stats.hemodynamic_models <https://nistats.github.io/modules/reference.html#module-nistats.hemodynamic_models>`_
are in :mod:`nilearn.stats.first_level_model`

Design matrix
=============
Most imports from
`nistats.design_matrix <https://nistats.github.io/modules/reference.html#module-nistats.design_matrix>`_
are in :mod:`nilearn.stats.first_level_model`

`nistats.design_matrix.make_second_level_design_matrix <https://nistats.github.io/modules/generated/nistats.design_matrix.make_second_level_design_matrix.html#nistats.design_matrix.make_second_level_design_matrix>`_
is in :func:`nilearn.stats.second_level_model.make_second_level_design_matrix`

Experimental Paradigm
=====================
Imports from
`nistats.experimental_paradigm <https://nistats.github.io/modules/reference.html#module-nistats.experimental_paradigm>`_ are in :mod:`nilearn.stats.first_level_model`


Statistical Models
==================
Imports from
`nistats.model <https://nistats.github.io/modules/reference.html#module-nistats.model>`_
are now in :mod:`nilearn.stats`


Regression Models
=================
Imports from
`nistats.regression <https://nistats.github.io/modules/reference.html#module-nistats.regression>`_
are in :mod:`nilearn.stats`


First Level Model
=================
Imports from
`nistats.first_level_model <https://nistats.github.io/modules/reference.html#module-nistats.first_level_model>`_
are in :mod:`nilearn.stats.first_level_model`

Second Level Model
==================
Imports from
`nistats.second_level_model <https://nistats.github.io/modules/reference.html#module-nistats.second_level_model>`_
are in :mod:`nilearn.stats.second_level_model`

Contrasts
=========
imports from
`nistats.contrasts <https://nistats.github.io/modules/reference.html#module-nistats.contrasts>`_
are in :mod:`nilearn.stats`

Thresholding Maps
=================
Imports from
`nistats.thresholding <https://nistats.github.io/modules/reference.html#module-nistats.thresholding>`_
are in :mod:`nilearn.stats`

Report plotting functions
==========================
Imports from
`nistats.reporting <https://nistats.github.io/modules/reference.html#module-nistats.reporting>`_
are in :mod:`nilearn.reporting`

Utility functions
=================
Imports from
`nistats.utils <https://nistats.github.io/modules/reference.html#module-nistats.utils>`_
are in `nilearn._utils` and are usually meant for developer's use.
