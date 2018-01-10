.. _extracting_rsn:

==================================================
Extracting resting-state networks: ICA and related
==================================================

.. topic:: **Page summary**

   This page demonstrates the use of multi-subject Independent Component
   Analysis (ICA) of resting-state fMRI data to extract brain networks in
   an data-driven way. Here we use the 'CanICA' approach, that implements
   a multivariate random effects model across subjects. A newer technique,
   based on dictionary learning, is then described.


.. currentmodule:: nilearn.decomposition

Multi-subject ICA: CanICA
=========================

.. topic:: **References**

    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", `NeuroImage Vol 51 (2010)
      <http://www.sciencedirect.com/science/article/pii/S1053811910001618>`_, p. 288-299

Data preparation: retrieving example data
-----------------------------------------

We will use sample data from the `ADHD 200 resting-state dataset
<http://fcon_1000.projects.nitrc.org/indi/adhd200/>`_ has been
preprocessed using `CPAC <http://fcp-indi.github.io/>`_. We use nilearn
functions to fetch data from Internet and get the filenames (:ref:`more
on data loading <loading_data>`):


.. literalinclude:: ../../examples/03_connectivity/plot_canica_resting_state.py
    :start-after: # First we load the ADHD200 data
    :end-before:  ####################################################################

Applying CanICA
---------------

:class:`CanICA` is a ready-to-use object that can be applied to
multi-subject Nifti data, for instance presented as filenames, and will
perform a multi-subject ICA decomposition following the CanICA model.
As with every object in nilearn, we give its parameters at construction,
and then fit it on the data.

.. literalinclude:: ../../examples/03_connectivity/plot_canica_resting_state.py
    :start-after: # Here we apply CanICA on the data
    :end-before: ####################################################################

The components estimated are found as the `components_img_` attribute
of the object. A 4D Nifti image.

.. note::
    The `components_img_` attribute is implemented from version 0.4.1 which
    is easy for visualization without any additional step to unmask to image.
    For users who have older versions, components image can be done by
    unmasking attribute `components_`. See :ref:`section Inverse transform:
    unmasking data <unmasking_step>`.

Visualizing the results
-----------------------

We can visualize the components as in the previous examples. The first plot
shows a map generated from all the components. Then we plot an axial cut for
each component separately.

.. literalinclude:: ../../examples/03_connectivity/plot_canica_resting_state.py
    :start-after: # To visualize we plot the outline of all components on one figure
    :end-before: ####################################################################

.. figure:: ../auto_examples/03_connectivity/images/sphx_glr_plot_canica_resting_state_001.png
   :align: center
   :target: ../auto_examples/03_connectivity/plot_canica_resting_state.html

Finally, we can plot the map for different ICA components separately:

.. literalinclude:: ../../examples/03_connectivity/plot_canica_resting_state.py
    :start-after: # Finally, we plot the map for each ICA component separately

.. |left_img| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_canica_resting_state_003.png
   :width: 23%

.. |right_img| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_canica_resting_state_004.png
   :width: 23%

.. centered:: |left_img| |right_img|

.. seealso::

   The full code can be found as an example:
   :ref:`sphx_glr_auto_examples_03_connectivity_plot_canica_resting_state.py`

.. note::

   Note that as the ICA components are not ordered, the two components
   displayed on your computer might not match those of the documentation. For
   a fair representation, you should display all components and
   investigate which one resemble those displayed above.

Beyond ICA : Dictionary learning
================================

Recent work has shown that dictionary learning based techniques outperform
ICA in term of stability and constitutes a better first step in a statistical
analysis pipeline.
Dictionary learning in neuro-imaging seeks to extract a few representative
temporal elements along with their sparse spatial loadings, which constitute
good extracted maps.

.. topic:: **References**

   * Arthur Mensch et al. `Compressed online dictionary learning for fast resting-state fMRI decomposition <https://hal.archives-ouvertes.fr/hal-01271033/>`_,
     ISBI 2016, Lecture Notes in Computer Science

Applying DictLearning
---------------------

:class:`DictLearning` is a ready-to-use class with the same interface as CanICA.
Sparsity of output map is controlled by a parameter alpha: using a
larger alpha yields sparser maps.

.. literalinclude:: ../../examples/03_connectivity/plot_compare_resting_state_decomposition.py
    :start-after: # Dictionary learning
    :end-before: ###############################################################################

We can fit both estimators to compare them

.. literalinclude:: ../../examples/03_connectivity/plot_compare_resting_state_decomposition.py
    :start-after: # Fit both estimators
    :end-before: ###############################################################################

Visualizing the results
-----------------------

4D plotting offers an efficient way to compare both resulting outputs

.. literalinclude:: ../../examples/03_connectivity/plot_compare_resting_state_decomposition.py
    :start-after: # Visualize the results

.. |left_img_decomp| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_resting_state_decomposition_001.png
   :target: ../auto_examples/03_connectivity/plot_compare_resting_state_decomposition.html
   :width: 50%
.. |right_img_decomp| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_resting_state_decomposition_003.png
   :target: ../auto_examples/03_connectivity/plot_compare_resting_state_decomposition.html
   :width: 50%

.. |left_img_decomp_single| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_resting_state_decomposition_002.png
   :target: ../auto_examples/03_connectivity/plot_compare_resting_state_decomposition.html
   :width: 50%
.. |right_img_decomp_single| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_resting_state_decomposition_004.png
   :target: ../auto_examples/03_connectivity/plot_compare_resting_state_decomposition.html
   :width: 50%


.. centered:: |left_img_decomp| |right_img_decomp|
.. centered:: |left_img_decomp_single| |right_img_decomp_single|

Maps obtained with dictionary leaning are often easier to exploit as they are
less noisy than ICA maps, with blobs usually better defined. Typically,
*smoothing can be lower than when doing ICA*.
While dictionary learning computation time is comparable to CanICA, obtained
atlases have been shown to outperform ICA in a variety of
classification tasks.

.. seealso::

   The full code can be found as an example:
   :ref:`sphx_glr_auto_examples_03_connectivity_plot_compare_resting_state_decomposition.py`
