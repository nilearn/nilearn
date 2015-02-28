.. _functional_connectomes:

===============================================================
Extracting times series to build a functional connectome
===============================================================

.. topic:: **Page summary**

   A *functional connectome* is a set of connections representing brain
   interactions between regions. Here we show how to extract activation
   time-series to compute functional connectomes.

.. contents:: **Contents**
    :local:
    :depth: 1


.. topic:: **References**

   * `Varoquaux and Craddock, Learning and comparing functional
     connectomes across subjects, NeuroImage 2013
     <http://www.sciencedirect.com/science/article/pii/S1053811913003340>`_ 


Time-series from a brain parcellation or "MaxProb" atlas
===========================================================

Brain parcellations
--------------------

.. currentmodule:: nilearn.datasets

Regions used to extract the signal can be defined by a "hard"
parcellation. For instance, the :mod:`nilearn.datasets` has functions to
download atlases forming reference parcellation, eg
:func:`fetch_craddock_2011_atlas`, :func:`fetch_harvard_oxford`,
:func:`fetch_yeo_2011_atlas`.

For instance to retrieve the Yeo 2011 cortical parcelation into 17
regions, with a thick cortical model::

  from nilearn import datasets
  atlas_filename, labels = datasets.fetch_harvard_oxford('cort-maxprob-thr25-2mm')


Plotting can then be done as::

    from nilearn import plotting
    plotting.plot_roi(atlas_filename)

.. image:: ../auto_examples/manipulating_visualizing/images/plot_atlas_1.png
   :target: ../auto_examples/manipulating_visualizing/plot_atlas.html
   :scale: 60

.. seealso::

   * The :ref:`plotting documentation <plotting>`

   * The :ref:`dataset downloaders <datasets_ref>`

Extracting signals on a parcellation
----------------------------------------

.. currentmodule:: nilearn.input_data

To extract signal on the parcellation, the easiest option is to use the
:class:`nilearn.input_data.NiftiLabelsMasker`. As any ''maskers'' in
nilearn, it is an processing object that is created by specicifying all
the important parameters, but not the data::

    from nilearn.input_data import NiftiLabelsMasker
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

The Nifti data can then be turned to time-series by calling the
:class:`NiftiLabelsMasker` `fit_transform` method, that takes either
filenames or `NiftiImage objects
<http://nipy.org/nibabel/nibabel_images.html>`_::

    time_series = masker.fit_transform(frmi_files, confounds=csv_file)

|

Note that confound signals can be specified in the call. Indeed, to
obtain time series that capture well the functional interactions between
regions, regressing out noise sources is indeed very important
`[Varoquaux & Craddock 2013] <https://hal.inria.fr/hal-00812911/>`_. 

.. image:: ../auto_examples/connectivity/images/plot_signal_extraction_1.png
   :target: ../auto_examples/connectivity/plot_signal_extraction.html
   :scale: 40
.. image:: ../auto_examples/connectivity/images/plot_signal_extraction_2.png
   :target: ../auto_examples/connectivity/plot_signal_extraction.html
   :scale: 40

.. topic:: **Full example**

    See the following example for a full file running the analysis:
    :ref:`example_connectivity_plot_signal_extraction.py`


Time-series from a probabilistic atlas
========================================

* The MSDL atlas

Highlighting functional interaction: the connectome
====================================================

* Show the impact of confounds
