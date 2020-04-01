.. _nistats_migration:

===================================================
A Quick Guide to Migrating Nistats Code to Nilearn
===================================================

A quick guide to changing your Nistats imports to work in Nilearn


Datasets
========

Imports from :ref:`nistats.datasets` are in :mod:`nilearn.datasets`

Hemodynamic Models
==================
Imports from :ref:`nilearn.stats.hemodynamic_models` are in :mod:`nilearn.stats.first_level_model`

Design matrix
=============
Most imports from :ref:`nistats.design_matrix` are in :mod:`nilearn.stats.first_level_model`
:ref:`nistats.design_matrix.make_second_level_design_matrix` is in :func:`nilearn.stats.second_level_model.make_second_level_design_matrix`

Experimental Paradigm
=====================
Imports from :ref:`nistats.experimental_paradigm` are in :mod:`nilearn.stats.first_level_model`


Statistical Models
==================
Imports from :ref:`nistats.model` are now in :mod:`nilearn.stats`


Regression Models
=================
Imports from :ref:`nistats.regression` are in :mod:`nilearn.stats`


First Level Model
=================
Imports from :ref:`nistats.first_level_model` are in :mod:`nilearn.stats.first_level_model`

Second Level Model
==================
Imports from :ref:`nistats.second_level_model` are in :mod:`nilearn.stats.second_level_model`

Contrasts
=========
imports from :ref:`nistats.contrasts` are in :mod:`nilearn.stats`

Thresholding Maps
=================
Imports from :ref:`nistats.thresholding` are in :mod:`nilearn.stats`

Report plotting functions
==========================
Imports from :ref:`nistats.reporting` are in :mod:`nilearn.reporting`

Utility functions
=================
Imports from :ref:`nistats.utils` are in :mod:`nilearn._utils`
