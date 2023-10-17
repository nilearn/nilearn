.. _extracting_rsn:

=====================================================
Extracting functional brain networks: ICA and related
=====================================================

.. topic:: **Page summary**

   This page demonstrates the use of multi-subject decompositions models
   to extract brain-networks from :term:`fMRI` data in a data-driven way.
   Specifically, we will apply Independent Component Analysis (:term:`ICA`), which
   implements a multivariate random effects model across subjects. We will
   then compare :term:`ICA` to a newer technique, based on dictionary learning.


.. currentmodule:: nilearn.decomposition

Multi-subject ICA: CanICA
=========================

.. topic:: **References**

    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", `NeuroImage Vol 51 (2010)
      <https://www.sciencedirect.com/science/article/pii/S1053811910001618>`_, p. 288-299

Objective
----------
:term:`ICA` is a useful approach for finding independent sources from :term:`fMRI`
images. :term:`ICA` and similar techniques can be therefore used to define
regions or networks that share similar :term:`BOLD` signal across time. The
:term:`CanICA` incorporates information both within-subjects and across subjects
to arrive at consensus components.

.. topic:: **Nilearn data for examples**

   Nilearn provides easy-to-analyze data to explore :term:`functional connectivity`
   and resting: the `brain development dataset <https://osf.io/5hju4/files/>`_,
   which has been preprocessed using `FMRIPrep and Nilearn <https://osf.io/wjtyq/>`_
   We use nilearn functions to fetch data from Internet and get the
   filenames (:ref:`more on data loading <loading_data>`).


Fitting CanICA model with nilearn
---------------------------------
:class:`CanICA` is a ready-to-use object that can be applied to
multi-subject Nifti data, for instance presented as filenames, and will
perform a multi-subject :term:`ICA` decomposition following the :term:`CanICA` model.
As with every object in nilearn, we give its parameters at construction,
and then fit it on the data. For examples of this process, see
here: :ref:`sphx_glr_auto_examples_03_connectivity_plot_compare_decomposition.py`

Once an :term:`ICA` object has been fit to an :term:`fMRI` dataset, the individual
components can be accessed as a 4D Nifti object using the
``components_img_`` attribute.

Visualizing results
--------------------
We can visualize each component outlined over the brain:

.. figure:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_001.png
   :align: center
   :target: ../auto_examples/03_connectivity/plot_compare_decomposition.html

We can also plot the map for different components separately:

.. |ic1| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_002.png
   :width: 23%

.. |ic2| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_003.png
   :width: 23%

.. |ic3| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_004.png
   :width: 23%

.. |ic4| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_005.png
   :width: 23%

.. centered:: |ic1| |ic2| |ic3| |ic4|

.. seealso::

   The full code can be found as an example:
   :ref:`sphx_glr_auto_examples_03_connectivity_plot_compare_decomposition.py`

.. note::

   Note that as the :term:`ICA` components are not ordered, the two components
   displayed on your computer might not match those of the documentation. For
   a fair representation, you should display all components and
   investigate which one resemble those displayed above.

Interpreting such components
-----------------------------

:term:`ICA`, and related algorithms, extract patterns that coactivate in the
signal. As a result, it finds functional networks, but also patterns of
non neural activity, ie confounding signals. Both are visible in the
plots of the components.

An alternative to :term:`ICA`: :term:`Dictionary learning`
==========================================================
Recent work has shown that :term:`Dictionary learning` based techniques
outperform :term:`ICA` in term of stability and constitutes a better first
step in a statistical analysis pipeline.
:term:`Dictionary learning` in neuro-imaging seeks to extract a few
representative temporal elements along with their sparse spatial loadings,
which constitute good extracted maps.

.. topic:: **References**

    * Arthur Mensch et al. `Compressed online dictionary learning for fast resting-state fMRI decomposition <https://hal.archives-ouvertes.fr/hal-01271033/>`_,
      ISBI 2016, Lecture Notes in Computer Science

:class:`DictLearning` is a ready-to-use class with the same interface as
:class:`CanICA`. Sparsity of output map is controlled by a parameter alpha: using
a larger alpha yields sparser maps.

We can fit both estimators to compare them. 4D plotting (using
:func:`nilearn.plotting.plot_prob_atlas`) offers an efficient way to
compare both resulting outputs.

.. figure:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_022.png
   :target: ../auto_examples/03_connectivity/plot_compare_decomposition.html
   :align: center

.. figure:: ../auto_examples/03_connectivity/images/sphx_glr_plot_compare_decomposition_001.png
   :target: ../auto_examples/03_connectivity/plot_compare_decomposition.html
   :align: center


Maps obtained with :term:`Dictionary learning` are often easier to exploit as they are
more contrasted than :term:`ICA` maps, with blobs usually better defined. Typically,
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

While :term:`Dictionary learning` computation time is comparable to
:term:`CanICA`, obtained atlases have been shown to outperform :term:`ICA`
in a variety of classification tasks.

.. seealso::

   The full code can be found as an example:
   :ref:`sphx_glr_auto_examples_03_connectivity_plot_compare_decomposition.py`

.. seealso::

   Learn how to extract :term:`fMRI` data from regions created with
   :term:`Dictionary learning` with this example:
   :ref:`sphx_glr_auto_examples_03_connectivity_plot_extract_regions_dictlearning_maps.py`
