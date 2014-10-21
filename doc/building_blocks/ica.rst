.. _ica_rest:

=========================================
ICA on resting-state data
=========================================

.. topic:: **Page summary**
   
   Here we show how to apply Independent Component Analysis (ICA) to fMRI
   data to extract brain networks. This page is a low-level technical
   example that shows the simplest way to apply an unsupervised method
   for spatial analysis of fMRI.
   
   For a high-performance extraction of resting-state networks, multi-subject
   models and proper data preparation is necessary. Please refer to
   :ref:`extracting_rsn`.

.. topic:: **References**

   * `Kiviniemi et al, *Independent component analysis of nondeterministic
     fMRI signal sources*, Neuroimage 2001 <http://dx.doi.org/10.1016/S1053-8119(03)00097-1>`_
      
   * `Beckmann et al, *Investigations into resting-state connectivity using
     independent component analysis*, Philos Trans R Soc Lond B 2005
     <http://dx.doi.org/10.1098/rstb.2005.1634>`_

Data preparation
==================

Retrieving the example data
----------------------------

As seen in :ref:`loading_data`, we fetch data from Internet and get
the filenames with a function provided by nilearn:


.. literalinclude:: ../../plot_ica_resting_state.py
    :start-after: ### Load nyu_rest dataset #####################################################
    :end-before: ### Preprocess ################################################################

Concatenating, smoothing, and masking
--------------------------------------

.. literalinclude:: ../../plot_ica_resting_state.py
    :start-after: ### Preprocess ################################################################
    :end-before: ### Apply ICA #################################################################

Applying ICA
==============

.. literalinclude:: ../../plot_ica_resting_state.py
    :start-after: ### Apply ICA #################################################################
    :end-before: ### Visualize the results #####################################################

Visualizing the results
========================

Visualization follows similarly as in other examples (for details, see
:ref:`plotting`).

.. literalinclude:: ../../plot_ica_resting_state.py
    :start-after: ### Visualize the results #####################################################

.. |left_img| image:: ../auto_examples/images/plot_ica_resting_state_1.png
   :target: ../auto_examples/plot_ica_resting_state.html
   :width: 48%

.. |right_img| image:: ../auto_examples/images/plot_ica_resting_state_2.png
   :target: ../auto_examples/plot_ica_resting_state.html
   :width: 48%

|left_img| |right_img|

