.. _extracting_rsn:

=====================================================
Extracting functional brain networks: ICA and related
=====================================================

.. topic:: **Page summary**

   This page demonstrates the use of multi-subject decompositions models
   to extract brain-networks from fMRI data in a data-driven way.
   Specifically, we will apply Independent Component Analysis (ICA), which
   implements a multivariate random effects model across subjects. We will
   then compare ICA to a newer technique, based on dictionary learning.


.. currentmodule:: nilearn.decomposition

Multi-subject ICA: CanICA
=========================

.. topic:: **References**

    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", `NeuroImage Vol 51 (2010)
      <http://www.sciencedirect.com/science/article/pii/S1053811910001618>`_, p. 288-299

Objective
----------
ICA is a useful approach for finding independent sources from fMRI
images. ICA and similar techniques can be therefore used to define
regions or networks that share similar BOLD signal across time. The
CanICA incorporates information both within-subjects and across subjects
to arrive at consensus components.

.. topic:: **Nilearn data for examples**

   Nilearn provides easy-to-analyze data to explore functional connectivity and resting: the
   `brain development dataset <https://osf.io/5hju4/files/>`_, which
   has been preprocessed using `FMRIPrep and Nilearn <https://osf.io/wjtyq/>`_
   We use nilearn functions to fetch data from Internet and get the
   filenames (:ref:`more on data loading <loading_data>`).


Fitting CanICA model with nilearn
---------------------------------
:class:`CanICA` is a ready-to-use object that can be applied to
multi-subject Nifti data, for instance presented as filenames, and will
perform a multi-subject ICA decomposition following the CanICA model.
As with every object in nilearn, we give its parameters at construction,
and then fit it on the data.

.. literalinclude:: ../../examples/03_connectivity/plot_canica_analysis.py
    :start-after: # Here we apply CanICA on the data
    :end-before: # Calculate explained variance score per component

The components estimated are found as the `components_img_` attribute
of the object. A 4D Nifti image.

.. note::
    The `components_img_` attribute is implemented from version 0.4.1 which
    is easy for visualization without any additional step to unmask to image.
    For users who have older versions, components image can be done by
    unmasking attribute `components_`. See :ref:`section Inverse transform:
    unmasking data <unmasking_step>`.

Calculate explained variance on CanICA components
-------------------------------------------------
We can also use `score` method from CanICA object to calculate explained
variance per component after applying CanICA.

.. literalinclude:: ../../examples/03_connectivity/plot_canica_resting_state.py
    :start-after: # Calculate explained variance score per component
    :end-before:  ##############################################################

Visualizing the results
-----------------------

Visualizing results
--------------------
We can visualize each component outlined over the brain:

.. figure:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_001.png
   :align: center
   :target: ../auto_examples/03_connectivity/plot_compare_decomposition.html

We can plot the map for different ICA components separately:

.. literalinclude:: ../../examples/03_connectivity/plot_canica_resting_state.py
    :start-after: # We can plot the map for different ICA components separately
    :end-before: ####################################################################

.. |ic1| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_002.png
   :width: 23%

.. |ic2| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_003.png
   :width: 23%

.. |ic3| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_004.png
   :width: 23%

.. |ic4| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_005.png
   :width: 23%

.. centered:: |ic1| |ic2| |ic3| |ic4|

Finally, we plot the explained variance score for each ICA component using matplotlib

.. literalinclude:: ../../examples/03_connectivity/plot_canica_resting_state.py
    :start-after: # Finally, we plot the `score` for each ICA component using matplotlib

.. figure:: ../auto_examples/03_connectivity/images/sphx_glr_plot_canica_resting_state_022.png
   :align: center
   :target: ../auto_examples/03_connectivity/plot_canica_resting_state.html

.. seealso::

   The full code can be found as an example:
   :ref:`sphx_glr_auto_examples_03_connectivity_plot_compare_decomposition.py`

.. note::

   Note that as the ICA components are not ordered, the two components
   displayed on your computer might not match those of the documentation. For
   a fair representation, you should display all components and
   investigate which one resemble those displayed above.

Interpreting such components
-----------------------------

ICA, and related algorithms, extract patterns that coactivate in the
signal. As a result, it finds functional networks, but also patterns of
non neural activity, ie confounding signals. Both are visible in the
plots of the components.

An alternative to ICA: Dictionary learning
===========================================
Recent work has shown that dictionary learning based techniques outperform
ICA in term of stability and constitutes a better first step in a statistical
analysis pipeline.
Dictionary learning in neuro-imaging seeks to extract a few representative
temporal elements along with their sparse spatial loadings, which constitute
good extracted maps.

.. topic:: **References**

    * Arthur Mensch et al. `Compressed online dictionary learning for fast resting-state fMRI decomposition <https://hal.archives-ouvertes.fr/hal-01271033/>`_,
      ISBI 2016, Lecture Notes in Computer Science

:class:`DictLearning` is a ready-to-use class with the same interface as CanICA.
Sparsity of output map is controlled by a parameter alpha: using a
larger alpha yields sparser maps.

We can fit both estimators to compare them. 4D plotting (using
:func:`nilearn.plotting.plot_prob_atlas`) offers an efficient way to
compare both resulting outputs.

.. figure:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_022.png
   :target: ../auto_examples/03_connectivity/plot_compare_decomposition.html
   :align: center

.. figure:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_001.png
   :target: ../auto_examples/03_connectivity/plot_compare_decomposition.html
   :align: center


Maps obtained with dictionary learning are often easier to exploit as they are
more contrasted than ICA maps, with blobs usually better defined. Typically,
*smoothing can be lower than when doing ICA*.

.. |dl1| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_023.png
   :width: 23%

.. |dl2| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_024.png
   :width: 23%

.. |dl3| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_025.png
   :width: 23%

.. |dl4| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_026.png
   :width: 23%

.. centered:: |dl1| |dl2| |dl3| |dl4|

While dictionary learning computation time is comparable to CanICA, obtained
atlases have been shown to outperform ICA in a variety of
classification tasks.

.. seealso::

   The full code can be found as an example:
   :ref:`sphx_glr_auto_examples_03_connectivity_plot_compare_decomposition.py`

.. seealso::

   Learn how to extract fMRI data from regions created with
   dictionary learning with this example:
   :ref:`sphx_glr_auto_examples_03_connectivity_plot_extract_regions_dictlearning_maps.py`


