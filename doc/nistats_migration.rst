.. _nistats_migration:

======================
Migrating from Nistats
======================

A quick guide to changing your Nistats imports to work in Nilearn

Datasets
========
Imports from
`nistats.datasets <https://nistats.github.io/modules/reference.html#module-nistats.datasets>`_
are in :mod:`nilearn.datasets`

Hemodynamic Models
==================
Imports from
`nistats.hemodynamic_models <https://nistats.github.io/modules/reference.html#module-nistats.hemodynamic_models>`_
are in :mod:`nilearn.glm.first_level`

Design matrix
=============
Most imports from
`nistats.design_matrix <https://nistats.github.io/modules/reference.html#module-nistats.design_matrix>`_
are in :mod:`nilearn.glm.first_level`

`nistats.design_matrix.make_second_level_design_matrix <https://nistats.github.io/modules/generated/nistats.design_matrix.make_second_level_design_matrix.html#nistats.design_matrix.make_second_level_design_matrix>`_
is in :mod:`nilearn.glm.second_level`

Experimental Paradigm
=====================
Imports from
`nistats.experimental_paradigm <https://nistats.github.io/modules/reference.html#module-nistats.experimental_paradigm>`_ are in :mod:`nilearn.glm.first_level`


Statistical Models
==================
Imports from
`nistats.model <https://nistats.github.io/modules/reference.html#module-nistats.model>`_
are now in :mod:`nilearn.glm`


Regression Models
=================
Imports from
`nistats.regression <https://nistats.github.io/modules/reference.html#module-nistats.regression>`_
are in :mod:`nilearn.glm`


First Level Model
=================
Imports from
`nistats.first_level <https://nistats.github.io/modules/reference.html#module-nistats.first_level>`_
are in :mod:`nilearn.glm.first_level`

Second Level Model
==================
Imports from
`nistats.second_level <https://nistats.github.io/modules/reference.html#module-nistats.second_level>`_
are in :mod:`nilearn.glm.second_level`

Contrasts
=========
imports from
`nistats.contrasts <https://nistats.github.io/modules/reference.html#module-nistats.contrasts>`_
are in :mod:`nilearn.glm`

Thresholding Maps
=================
Imports from
`nistats.thresholding <https://nistats.github.io/modules/reference.html#module-nistats.thresholding>`_
are in :mod:`nilearn.glm`

Report plotting functions
==========================
Imports from
`nistats.reporting <https://nistats.github.io/modules/reference.html#module-nistats.reporting>`_
are in :mod:`nilearn.reporting` or :mod:`nilearn.plotting`

Utility functions
=================
Imports from
`nistats.utils <https://nistats.github.io/modules/reference.html#module-nistats.utils>`_
are in `nilearn._utils` and are usually meant for development purposes.
