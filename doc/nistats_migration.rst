.. _nistats_migration:

===================================================
A Quick Guide to Migrating Nistats Code to Nilearn
===================================================

A quick guide to changing your Nistats imports to work in Nilearn


Datasets
========

Imports from :doc:`nistats.datasets` are in :mod:`nilearn.datasets`

Hemodynamic Models
==================
Imports from :doc:`nilearn.stats.hemodynamic_models` are in :mod:`nilearn.stats.first_level_model`

Design matrix
=============
Most imports from :doc:`nistats.design_matrix` are in :mod:`nilearn.stats.first_level_model`
:doc:`nistats.design_matrix.make_second_level_design_matrix` is in :func:`nilearn.stats.second_level_model.make_second_level_design_matrix`

Experimental Paradigm
=====================
Imports from :doc:`nistats.experimental_paradigm` are in :mod:`nilearn.stats.first_level_model`


Statistical Models
==================
Imports from :doc:`nistats.model` are now in :mod:`nilearn.stats`


Regression Models
=================
Imports from :doc:`nistats.regression` are in :mod:`nilearn.stats`


First Level Model
=================
Imports from :doc:`nistats.first_level_model` are in :mod:`nilearn.stats.first_level_model`

Second Level Model
==================
Imports from :doc:`nistats.second_level_model` are in :mod:`nilearn.stats.second_level_model`

Contrasts
=========
imports from :doc:`nistats.contrasts` are in :mod:`nilearn.stats`

Thresholding Maps
=================
Imports from :doc:`nistats.thresholding` are in :mod:`nilearn.stats`

Report plotting functions
==========================
Imports from :doc:`nistats.reporting` are in :mod:`nilearn.reporting`

Utility functions
=================
Imports from :doc:`nistats.utils` are in :mod:`nilearn._utils`
