.. _extracting_rsn:

===========================================
Extracting resting-state networks with ICA
===========================================

.. topic:: **Page summary**

   This page demonstrates the use of multi-subject Independent Component
   Analysis (ICA) of resting-state fMRI data to extract brain networks in
   an data-driven way. Here we use the 'CanICA' approach, that implements
   a multivariate random effects model across subjects.

.. topic:: **References**

    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", `NeuroImage Vol 51 (2010) <http://www.sciencedirect.com/science/article/pii/S1053811910001618>`_, p. 288-299


.. currentmodule:: nilearn.decomposition

Data preparation: retrieving example data
==========================================

We will use sample data from the `ADHD 200 resting-state dataset
<http://fcon_1000.projects.nitrc.org/indi/adhd200/>`_ has been
preprocessed using `CPAC <http://fcp-indi.github.io/>`_. We use nilearn
functions to fetch data from Internet and get the filenames (:ref:`more
on data loading <loading_data>`):


.. literalinclude:: ../../examples/connectivity/plot_canica_resting_state.py
    :start-after: ### Load ADHD rest dataset ####################################################
    :end-before: ### Apply CanICA ##############################################################


Applying CanICA
================

:class:`CanICA` is a ready-to-use object that can be applied to
multi-subject Nifti data, for instance presented as filenames, and will
perform a multi-subject ICA decomposition following the CanICA model.
As with every object in nilearn, we give its parameters at construction,
and then fit it on the data.

.. literalinclude:: ../../examples/connectivity/plot_canica_resting_state.py
    :start-after: ### Apply CanICA ##############################################################
    :end-before: ### Visualize the results #####################################################

The components estimated are found as the `components_` attribute of the
object.

Visualizing the results
========================

We can visualize the components as in the previous examples.

.. literalinclude:: ../../examples/connectivity/plot_canica_resting_state.py
    :start-after: ### Visualize the results #####################################################

.. |left_img| image:: ../auto_examples/connectivity/images/plot_canica_resting_state_002.png
   :target: ../auto_examples/connectivity/plot_canica_resting_state.html
   :width: 48%

.. |right_img| image:: ../auto_examples/connectivity/images/plot_canica_resting_state_003.png
   :target: ../auto_examples/connectivity/plot_canica_resting_state.html
   :width: 48%

|left_img| |right_img|

.. seealso::

   The full code can be found as an example:
   :ref:`example_connectivity_plot_canica_resting_state.py`

.. note::

   Note that as the ICA components are not ordered, the two components
   displayed on your computer might not match those of the documentation. For
   a fair representation, you should display all components and
   investigate which one resemble those displayed above.
