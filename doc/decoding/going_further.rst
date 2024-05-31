.. _going_further:

==========================================================================
Running scikit-learn functions for more control on the analysis
==========================================================================

This section gives pointers to design your own decoding pipelines with
scikit-learn. This builds on the :ref:`didactic introduction to decoding <decoding_intro>`.

.. note::

   This documentation gives links and additional definitions needed to work
   correctly with scikit-learn. For a full code example, please check out: :ref:`sphx_glr_auto_examples_07_advanced_plot_advanced_decoding_scikit.py`


Performing decoding with scikit-learn
=======================================

Using scikit-learn estimators
--------------------------------

You can easily import estimators from the `scikit-learn <https://scikit-learn.org>`_ machine-learning library,
those available in the ``Decoder`` object and many others.
They all have the ``fit`` and ``predict`` functions.
For example you can directly import the versatile `Support Vector Classifier <https://scikit-learn.org/stable/modules/svm.html>`_ (or SVC).

To learn more about the variety of classifiers available in scikit-learn, see the `scikit-learn documentation on supervised learning <https://scikit-learn.org/stable/supervised_learning.html>`_.


Cross-validation with scikit-learn
-----------------------------------

To perform cross-validation using a scikit-learn estimator, you should first
mask the data using a :class:`nilearn.maskers.NiftiMasker`: to extract
only the :term:`voxels<voxel>` inside the mask of interest, and transform 4D input :term:`fMRI`
data to 2D arrays (shape (n_timepoints, n_voxels)) that estimators can work on.

.. note::

   This example shows how to use masking:
   :ref:`sphx_glr_auto_examples_06_manipulating_images_plot_nifti_simple.py`

Then use a specific function :func:`sklearn.model_selection.cross_val_score`
that computes for you the score of your model for the different folds
of cross-validation.

You can change many parameters of the cross_validation here, for example:

* use a different cross-validation scheme, for example :class:`sklearn.model_selection.LeaveOneGroupOut`.

* speed up the computation by using ``n_jobs=-1``, which will spread the computation equally across all processors.

* use a different scoring function, as a keyword or imported from scikit-learn such as ``scoring="roc_auc"``.

.. seealso::

   * If you need more than only than cross-validation scores (i.e the predictions
     or models for each fold) or if you want to learn more on various cross-validation schemes,
     see `here <https://scikit-learn.org/stable/modules/cross_validation.html>`_.

   * `How to evaluate a model using scikit-learn
     <https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values>`_.


Measuring the chance level
---------------------------

**Dummy estimators**: The simplest way to measure prediction performance at chance is to use a *"dummy"* classifier: :class:`sklearn.dummy.DummyClassifier`.

**Permutation testing**: A more controlled way, but slower, is to do permutation testing on the labels, with :func:`sklearn.model_selection.permutation_test_score`.

.. topic:: **Decoding on simulated data**

   Simple simulations may be useful to understand the behavior of a given
   decoder on data. In particular, simulations enable us to set the true
   weight maps and compare them to the ones retrieved by decoders. A full
   example running simulations and discussing them can be found in
   :ref:`sphx_glr_auto_examples_02_decoding_plot_simulated_data.py`
   Simulated data cannot easily mimic all properties of brain data. An
   important aspect, however, is its spatial structure, that we create in
   the simulations.


Going further with scikit-learn
================================

We have seen a very simple analysis with scikit-learn, but your can easily add
intermediate processing steps if your analysis requires it. Some common
examples are :

* adding a feature selection step using scikit-learn pipelines
* use any model available in scikit-learn (or compatible with) at any step
* add more intermediate steps such as clustering

Decoding without a mask: Anova-SVM using scikit-learn
------------------------------------------------------

We can also implement feature selection before decoding as a scikit-learn pipeline (:class:`sklearn.pipeline.Pipeline`).
For this, we need to import the :mod:`sklearn.feature_selection` module and use :func:`sklearn.feature_selection.f_classif`, a simple F-score based feature selection (a.k.a. `Anova <https://en.wikipedia.org/wiki/Analysis_of_variance#The_F-test>`_),

Using any other model in the pipeline
-------------------------------------

:term:`Anova<ANOVA>` - :term:`SVM` is a good baseline that will give reasonable results
in common settings. However it may be interesting for you to explore the
`wide variety of supervised learning algorithms in the scikit-learn
<https://scikit-learn.org/stable/supervised_learning.html>`_. These can readily
replace the :term:`SVM` in your pipeline and might be better fitted
to some usecases as discussed in the previous section.

The feature selection step can also be tuned. For example we could use a more
sophisticated scheme, such as `Recursive Feature Elimination (RFE)
<https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination>`_
or add some `a clustering step <https://scikit-learn.org/stable/modules/clustering.html>`_
before feature selection. This always amount to creating
`a pipeline <https://scikit-learn.org/stable/modules/compose.html>`_ that will
link those steps together and apply a sensible cross-validation scheme to it.
Scikit-learn usually takes care of the rest for us.

.. seealso::

  * The corresponding full code example to practice with pipelines :ref:`sphx_glr_auto_examples_07_advanced_plot_advanced_decoding_scikit.py`

  * The `scikit-learn documentation <https://scikit-learn.org>`_ with detailed
    explanations on a large variety of estimators and machine learning techniques.
    To become better at decoding, you need to study it.


Setting estimator parameters
============================

Most estimators have parameters that can be set to optimize their
performance. Importantly, this must be done via **nested**
cross-validation.

Indeed, there is noise in the cross-validation score, and when we vary
the parameter, the curve showing the score as a function of the parameter
will have bumps and peaks due to this noise. These will not generalize to
new data and chances are that the corresponding choice of parameter will
not perform as well on new data.

With scikit-learn nested cross-validation is done via
:class:`sklearn.model_selection.GridSearchCV`. It is unfortunately time
consuming, but the ``n_jobs`` argument can spread the load on multiple
CPUs.

.. seealso::

   `The scikit-learn documentation on choosing estimators and their parameters
   selection <https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html>`_
