.. for doctests to run, we need to define variables that are define in
   the literal includes
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> fmri_masked  = iris.data
    >>> target = iris.target
    >>> session = np.ones_like(target)
    >>> n_samples = len(target)

.. Remove doctest: +SKIP at LDA while dropping support for sklearn older than
    versions 0.17

.. _going_further:

==========================================================================
Running scikit-learn low-level functions for more control on the analysis
==========================================================================

This section gives pointers to design your own decoding pipelines with
scikit-learn. This builds on the :ref:`didactic introduction to decoding <decoding_intro>`.
Here also perform decoding of the visual category of a stimuli on Haxby
2001 dataset.

.. contents:: **Contents**
    :local:
    :depth: 1


Loading and preparing the data
===============================

Loading the data into nilearn
-----------------------------

* **Retrieving the data**: use :func:`nilearn.datasets.fetch_haxby`.

    >>> from nilearn import datasets  # doctest: +SKIP
    >>> haxby_dataset = datasets.fetch_haxby()  # doctest: +SKIP


* **Masking fMRI data**: To perform the analysis some voxels only, we use spatial mask provided with the dataset.

    >>> mask_filename = haxby_dataset.mask_vt[0]  # doctest: +SKIP

* **Loading the behavioral labels**:

    >>> import pandas as pd  # doctest: +SKIP
    >>> behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=' ')  # doctest: +SKIP

* **Sample mask**: Masking some of the time points
  may be useful to
  restrict to a specific pair of conditions (*eg* cats versus faces).

    >>> conditions = behavioral['labels']  # doctest: +SKIP
    >>> condition_mask = conditions.isin(['face', 'cat'])  # doctest: +SKIP
    >>> fmri_niimgs = index_img(fmri_filename, condition_mask)  # doctest: +SKIP
    >>> conditions = conditions[condition_mask]  # doctest: +SKIP
    >>> # Convert to numpy array
    >>> conditions = conditions.values  # doctest: +SKIP
    >>> session_label = behavioral['chunks'][condition_mask] # doctest: +SKIP

.. note::
  For the sake of brevity, we didn't repeat here basic informations and warnings
  about data preprocessing and manipulation. They are available in :ref:decoding_intro


Performing decoding with scikit-learn
=======================================

Using scikit-learn estimators
--------------------------------


You can easily import estimators from the `scikit-learn <http://scikit-learn.org>`
machine-learning library. Those available in the `Decoder` object and others
all have the `fit` and `predict` functions. For example for `Support Vector Classifier
  <http://scikit-learn.org/stable/modules/svm.html>`_ (or SVC):

    >>> from sklearn.svm import SVC
    >>> svc = SVC()

To learn more about the variety of classifiers available in scikit-learn,
see the `scikit-learn documentation on supervised learning
<http://scikit-learn.org/stable/supervised_learning.html>`_).


Cross-validation with scikit-learn
----------------------------------

To perform cross-validation using a scikit-learn estimator, you should first
mask the data using a :class:`nilearn.input_data.NiftiMasker`: to extract
only the voxels inside the mask of interest, and transform 4D input fMRI
data to 2D arrays (shape (n_timepoints, n_voxels)) that estimators can work on.


    >>> masker = NiftiMasker(mask_img=mask_filename, sessions=session_label,
                           smoothing_fwhm=4, standardize=True,
                           memory="nilearn_cache", memory_level=1)
    >>> fmri_masked = masker.fit_transform(fmri_niimgs)

Then use a specific function :func:`sklearn.model_selection.cross_val_score`
that computes for you the score for the different folds of cross-validation::

    >>> from sklearn.model_selection import cross_val_score  # doctest: +SKIP
    >>> cv_scores = cross_val_score(svc, fmri_masked, conditions, cv=5)  # doctest: +SKIP
    >>> # Here `cv=5` stipulates a 5-fold cross-validation

You can change many parameters of the cross_validation here, for example:
* use a different cross-validation scheme, for example LeaveOneGroupOut()
* speed up the computation by using n_jobs=-1, which will spread the
  computation equally across all processors.
* use a different scoring function, as a keyword or imported from scikit-learn
scoring='roc_auc'

    >>> cv = LeaveOneGroupOut() # doctest: +SKIP
    >>> cv_scores = cross_val_score(svc, fmri_masked, conditions,
                                    cv=cv,scoring='roc_auc',
                                    groups=session_label, n_jobs=-1, ) #doctest: +SKIP

.. seealso::

  * If you need more than only than cross-validation scores (i.e the predictions
    or models for each fold) or if you want to learn more on various
    cross-validation schemes, see:
    <https://scikit-learn.org/stable/modules/cross_validation.html>`_
  * `how to evaluate a model using scikit-learn
    <http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values>`_


Measuring the chance level
----------------------------------

**Dummy estimators**: The simplest way to measure prediction performance
at chance, is to use a *"dummy"* classifier,
:class:`sklearn.dummy.DummyClassifier` (purely random)::

    >>> from sklearn.dummy import DummyClassifier
    >>> null_cv_scores = cross_val_score(DummyClassifier(), fmri_masked, conditions, cv=cv)  # doctest: +SKIP

**Permutation testing**: A more controlled way, but slower, is to do
permutation testing on the labels, with
:func:`sklearn.model_selection.permutation_test_score`::

    >>> from sklearn.model_selection import permutation_test_score
    >>> null_cv_scores = permutation_test_score(svc, fmri_masked, conditions, cv=cv)  # doctest: +SKIP

.. topic:: **Decoding on simulated data**

   Simple simulations may be useful to understand the behavior of a given
   decoder on data. In particular, simulations enable us to set the true
   weight maps and compare them to the ones retrieved by decoders. A full
   example running simulations and discussing them can be found in
   :ref:`sphx_glr_auto_examples_02_decoding_plot_simulated_data.py`
   Simulated data cannot easily mimic all properties of brain data. An
   important aspect, however, is its spatial structure, that we create in
   the simulations.


Decoding without a mask: Anova-SVM in scikit-lean
==================================================

We can also implement feature selection before decoding as a scikit-learn
`pipeline`(:class:`sklearn.pipeline.Pipeline`). For this, we need to import
the :mod:`sklearn.feature_selection` module and use
:func:`sklearn.feature_selection.f_classif`, a simple F-score
based feature selection (a.k.a. `Anova <https://en.wikipedia.org/wiki/Analysis_of_variance#The_F-test>`_),

    >>> from sklearn.feature_selection import SelectPercentile, f_classif
    >>> feature_selection = SelectPercentile(f_classif, percentile=5)
    >>> from sklearn.pipeline import Pipeline
    >>> anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])
    >>> # We can use our ``anova_svc`` object exactly as we were using our ``svc``
    >>> # object previously.
    >>> cv_scores = cross_val_score(anova_svc, fmri_masked, conditions,
                                    cv=cv, groups=session_label) # doctest: +SKIP
    >>> print(cv_scores.mean()) # doctest: +SKIP
    >>> # Visualize the SVC's discriminating weights
    >>> coef = svc.coef_ # doctest: +SKIP
    >>> coef = feature_selection.inverse_transform(coef) # doctest: +SKIP
    >>> weight_img = masker.inverse_transform(coef) # doctest: +SKIP
    >>> plot_stat_map(weight_img, title='Anova+SVC weights') # doctest: +SKIP

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

  * `The scikit-learn documentation on choosing estimators and their parameters
    selection <https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html>`_


Going further with scikit-learn
===============================

We have seen a very simple analysis with scikit-learn, but it may be
interesting to explore the `wide variety of supervised learning
algorithms in the scikit-learn
<http://scikit-learn.org/stable/supervised_learning.html>`_.

Changing the prediction engine
------------------------------

.. for doctest:
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> from sklearn.svm import LinearSVC
    >>> feature_selection = SelectKBest(f_classif, k=4) # doctest: +SKIP


We now see how one can easily change the prediction engine, if needed.
We can try Fisher's `Linear Discriminant Analysis (LDA)
<http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html>`_

Import the module::

    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # doctest: +SKIP

Construct the new estimator object and use it in a pipeline::

    >>> from sklearn.pipeline import Pipeline
    >>> lda = LinearDiscriminantAnalysis()  # doctest: +SKIP
    >>> anova_lda = Pipeline([('anova', feature_selection), ('LDA', lda)])  # doctest: +SKIP

.. note::
  Import Linear Discriminant Analysis method in "sklearn.lda.LDA" if you are using
  scikit-learn older than version 0.17.

and recompute the cross-validation score::

    >>> cv_scores = cross_val_score(anova_lda, fmri_masked, target, cv=cv, verbose=1)  # doctest: +SKIP
    >>> classification_accuracy = np.mean(cv_scores)  # doctest: +SKIP
    >>> n_conditions = len(set(target))  # number of target classes
    >>> print("Classification accuracy: %.4f / Chance Level: %.4f" % \
    ...    (classification_accuracy, 1. / n_conditions))  # doctest: +SKIP
    Classification accuracy: 0.7846 / Chance level: 0.5000


Changing the feature selection
------------------------------
Let's start by defining a linear SVM as a first classifier::

    >>> clf = LinearSVC()


Let's say that you want a more sophisticated feature selection, for example a
`Recursive Feature Elimination (RFE)
<http://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination>`_

Import the module::

    >>> from sklearn.feature_selection import RFE

Construct your new fancy selection::

    >>> rfe = RFE(SVC(kernel='linear', C=1.), 50, step=0.25)

and create a new pipeline, composing the two classifiers `rfe` and `clf`::

    >>> rfe_svc = Pipeline([('rfe', rfe), ('svc', clf)])

and recompute the cross-validation score::

    >>> cv_scores = cross_val_score(rfe_svc, fmri_masked, target, cv=cv,
    ...     n_jobs=-1, verbose=1)  # doctest: +SKIP

But, be aware that this can take *A WHILE*...

|

.. seealso::

  * The `scikit-learn documentation <http://scikit-learn.org>`_
    has very detailed explanations on a large variety of estimators and
    machine learning techniques. To become better at decoding, you need
    to study it.
