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

.. _decoding_intro:

=============================
An introduction to decoding
=============================

This section gives an introduction to the main concept of decoding:
predicting from brain images.

The discussion and examples are articulated on the analysis of the Haxby
2001 dataset, showing how to predict from fMRI images the stimuli that
the subject is viewing. However the process is the same in other settings
predicting from other brain imaging modalities, for instance predicting
phenotype or diagnostic status from VBM (Voxel Based Morphometry) maps
(as illustrated in :ref:`a more complex example
<sphx_glr_auto_examples_02_decoding_plot_oasis_vbm.py>`), or from FA maps
to capture diffusion mapping.


.. contents:: **Contents**
    :local:
    :depth: 1


Loading and preparing the data
===============================

The Haxby 2001 experiment
-------------------------

In the Haxby experiment, 
subjects were presented visual stimuli from different categories. We are
going to predict which category the subject is seeing from the fMRI
activity recorded in masks of the ventral stream. Significant prediction
shows that the signal in the region contains information on the
corresponding category.

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_stimuli_004.png
   :target: ../auto_examples/02_decoding/plot_haxby_stimuli.html
   :scale: 30
   :align: left

   Face stimuli

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_stimuli_002.png
   :target: ../auto_examples/02_decoding/plot_haxby_stimuli.html
   :scale: 30
   :align: left

   Cat stimuli

.. figure:: ../auto_examples/01_plotting/images/sphx_glr_plot_haxby_masks_001.png
   :target: ../auto_examples/01_plotting/plot_haxby_masks.html
   :scale: 30
   :align: left

   Masks

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_full_analysis_001.png
   :target: ../auto_examples/02_decoding/plot_haxby_full_analysis.html
   :scale: 35
   :align: left

   Decoding scores per mask

_____

.. topic:: **fMRI: using beta maps of a first-level analysis**

   The Haxby experiment is unusual because the experimental paradigm is
   made of many blocks of continuous stimulation. Most cognitive
   experiments have a more complex temporal structure with rich sequences
   of events.

   The standard approach to decoding consists in fitting a first-level
   GLM to retrieve one response map (a beta map) per trial. This is
   sometimes known as "beta-series regressions" (see Mumford et al,
   *Deconvolving bold activation in event-related designs for multivoxel
   pattern classification analyses*, NeuroImage 2012). These maps can
   then be input to the decoder as below, predicting the conditions
   associated to trial.

   For simplicity, we will work on the raw time-series of the data.
   However, **it is strongly recomended that you fit a first level to
   include an HRF model and isolate the responses from various
   confounds**.


Loading the data into nilearn
-----------------------------

.. topic:: **Full code example**

   The documentation here just gives the big idea. A full code example,
   with explanation, can be found on
   :ref:`sphx_glr_auto_examples_plot_decoding_tutorial.py`

* **Starting an environment**: Launch IPython via "ipython --matplotlib"
  in a terminal, or use the Jupyter notebook.

* **Retrieving the data**: In the tutorial, we load the data using nilearn
  data downloading function, :func:`nilearn.datasets.fetch_haxby`.
  However, all this function does is to download the data and return 
  paths to the files downloaded on the disk. To input your own data to
  nilearn, you can pass in the path to your own files 
  (:ref:`more on data input <loading_data>`).

* **Loading the behavioral labels**: Behavioral information is often stored
  in a text file such as a CSV, and must be load with
  **numpy.recfromcsv** or `pandas <http://pandas.pydata.org/>`_

* **Extracting the fMRI data**: we then use the
  :class:`nilearn.input_data.NiftiMasker`: we extract only the voxels on
  the mask of the ventral temporal cortex that comes with the data,
  applying the `mask_vt` mask to
  the 4D fMRI data. The resulting data is then a matrix with a shape that is
  (n_timepoints, n_voxels)
  (see :ref:`mask_4d_2_3d` for a discussion on using masks).

* **Sample mask**: Masking some of the time points may be useful to
  restrict to a specific pair of conditions (*eg* cats versus faces).

.. note::

   Seemingly minor data preparation can matter a lot on the final score,
   for instance standardizing the data.


.. seealso::

   * :ref:`loading_data`
   * :ref:`masking`



Performing a simple decoding analysis
=======================================

The prediction engine
---------------------

An estimator object
...................

To perform decoding we need to use an estimator from the `scikit-learn
<http://scikit-learn.org>` machine-learning library. This object can
predict a condition label **y** given a set **X** of imaging data.

A simple and yet performant choice is the `Support Vector Classifier
<http://scikit-learn.org/stable/modules/svm.html>`_ (or SVC) with a
linear kernel. The corresponding class, :class:`sklearn.svm.SVC`, needs
to be imported from the scikit-learn.

Note that the documentation of the object details all parameters. In
IPython, it can be displayed as follows::

    In [10]: svc?
    Type:             SVC
    Base Class:       <class 'sklearn.svm.libsvm.SVC'>
    String Form:
    SVC(kernel=linear, C=1.0, probability=False, degree=3, coef0=0.0, tol=0.001,
    cache_size=200, shrinking=True, gamma=0.0)
    Namespace:        Interactive
    Docstring:
        C-Support Vector Classification.
        Parameters
        ----------
        C : float, optional (default=1.0)
            penalty parameter C of the error term.
    ...

.. seealso::

   the `scikit-learn documentation on SVMs
   <http://scikit-learn.org/stable/modules/svm.html>`_


Applying it to data: fit (train) and predict (test)
...................................................

The prediction objects have two important methods:

- a `fit` function that "learns" the parameters of the model from the data.
  Thus, we need to give some training data to `fit`.
- a `predict` function that "predicts" a target from new data.
  Here, we just have to give the new set of images (as the target should be
  unknown):

.. warning::

    **Do not predict on data used by the fit: this would yield misleadingly optimistic scores.**

.. for doctests (smoke testing):
    >>> from sklearn.svm import SVC
    >>> svc = SVC()

Measuring prediction performance
--------------------------------

Cross-validation
................

We cannot measure a prediction error on the same set of data that we have
used to fit the estimator: it would be much easier than on new data, and
the result would be meaningless. We need to use a technique called
*cross-validation* to split the data into different sets, called "folds",
in a `K-Fold strategy
<https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`_.

.. for doctests:
   >>> cv = 2

There is a specific function,
:func:`sklearn.cross_validation.cross_val_score` that computes for you
the score for the different folds of cross-validation::

  >>> from sklearn.cross_validation import cross_val_score  # doctest: +SKIP
  >>> cv_scores = cross_val_score(svc, fmri_masked, target, cv=5)  # doctest: +SKIP

`cv=5` stipulates a 5-fold cross-validation. Note that this function is located
in `sklearn.model_selection.cross_val_score` in the newest version of
scikit-learn.

You can speed up the computation by using n_jobs=-1, which will spread
the computation equally across all processors (but might not work under
Windows)::

 >>> cv_scores = cross_val_score(svc, fmri_masked, target, cv=5, n_jobs=-1, verbose=10) #doctest: +SKIP

**Prediction accuracy**: We can take a look at the results of the
`cross_val_score` function::

  >>> print(cv_scores)  # doctest: +SKIP
  [0.72727272727272729, 0.46511627906976744, 0.72093023255813948, 0.58139534883720934, 0.7441860465116279]

This is simply the prediction score for each fold, i.e. the fraction of
correct predictions on the left-out data.

Choosing a good cross-validation strategy
.........................................

There are many cross-validation strategies possible, including K-Fold or
leave-one-out. When choosing a strategy, keep in mind that:

* The test set should be as litte correlated as possible with the train
  set
* The test set needs to have enough samples to enable a good measure of
  the prediction error (a rule of thumb is to use 10 to 20% of the data).

In these regards, leave one out is often one of the worst options (see
Varoquaux et al, *Assessing and tuning brain decoders: cross-validation,
caveats, and guidelines*, Neuroimage 2017).

Here, in the Haxby example, we are going to leave a session out, in order
to have a test set independent from the train set. For this, we are going
to use the session label, present in the behavioral data file, and
:class:`sklearn.cross_validation.LeaveOneLabelOut`.

.. note::

   Full code for the above can be found on
   :ref:`sphx_glr_auto_examples_plot_decoding_tutorial.py`

|

.. topic:: **Exercise**
   :class: green

   Compute the mean prediction accuracy using `cv_scores`.

.. topic:: Solution

    >>> classification_accuracy = np.mean(cv_scores)  # doctest: +SKIP
    >>> classification_accuracy  # doctest: +SKIP
    0.76851...

For discriminating human faces from cats, we measure a total prediction
accuracy of *77%* across the different sessions.

Choice of the prediction accuracy measure
.........................................

The default metric used for measuring errors is the accuracy score, i.e.
the number of total errors. It is not always a sensible metric,
especially in the case of very imbalanced classes, as in such situations
choosing the dominant class can achieve a low number of errors.

Other metrics, such as the AUC (Area Under the Curve, for the ROC: the
Receiver Operating Characteristic), can be used::

    >>> cv_scores = cross_val_score(svc, fmri_masked, target, cv=cv,  scoring='roc_auc')  # doctest: +SKIP

.. seealso::

   the `list of scoring options
   <http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values>`_

Measuring the chance level
..........................

**Dummy estimators**: The simplest way to measure prediction performance
at chance, is to use a *"dummy"* classifier,
:class:`sklearn.dummy.DummyClassifier` (purely random)::

    >>> from sklearn.dummy import DummyClassifier
    >>> null_cv_scores = cross_val_score(DummyClassifier(), fmri_masked, target, cv=cv)  # doctest: +SKIP

**Permutation testing**: A more controlled way, but slower, is to do
permutation testing on the labels, with
:func:`sklearn.cross_validation.permutation_test_score`::

  >>> from sklearn.cross_validation import permutation_test_score
  >>> null_cv_scores = permutation_test_score(svc, fmri_masked, target, cv=cv)  # doctest: +SKIP

|

.. topic:: **Putting it all together**

    The :ref:`ROI-based decoding example
    <sphx_glr_auto_examples_02_decoding_plot_haxby_full_analysis.py>` does a decoding analysis per
    mask, giving the f1-score of the prediction for each object.

    It uses all the notions presented above, with ``for`` loop to iterate
    over masks and categories and Python dictionaries to store the
    scores.


.. figure:: ../auto_examples/01_plotting/images/sphx_glr_plot_haxby_masks_001.png
   :target: ../auto_examples/01_plotting/plot_haxby_masks.html
   :scale: 55
   :align: left

   Masks


.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_full_analysis_001.png
   :target: ../auto_examples/02_decoding/plot_haxby_full_analysis.html
   :scale: 70
   :align: left



Visualizing the decoder's weights
---------------------------------

We can visualize the weights of the decoder:

- we first inverse the masking operation, to retrieve a 3D brain volume
  of the SVC's weights.
- we then create a figure and plot as a background the first EPI image
- finally we plot the SVC's weights after masking the zero values


.. figure:: ../auto_examples/images/sphx_glr_plot_decoding_tutorial_002.png
   :target: ../auto_examples/plot_decoding_tutorial.html
   :scale: 65

.. note::

   Full code for the above can be found on
   :ref:`sphx_glr_auto_examples_plot_decoding_tutorial.py`


.. seealso::

   * :ref:`plotting`


Decoding without a mask: Anova-SVM
==================================

Dimension reduction with feature selection
------------------------------------------

If we do not start from a mask of the relevant regions, there is a very
large number of voxels and not all are useful for
face vs cat prediction. We thus add a `feature selection
<http://scikit-learn.org/stable/modules/feature_selection.html>`_
procedure. The idea is to select the `k` voxels most correlated to the
task.

For this, we need to import the :mod:`sklearn.feature_selection` module and use
:func:`sklearn.feature_selection.f_classif`, a simple F-score
based feature selection (a.k.a.
`Anova <https://en.wikipedia.org/wiki/Analysis_of_variance#The_F-test>`_),
that we will put before the SVC in a `pipeline`
(:class:`sklearn.pipeline.Pipeline`):

.. literalinclude:: ../../examples/02_decoding/plot_haxby_anova_svm.py
    :start-after: # Build the decoder
    :end-before: # Visualize the results



We can use our ``anova_svc`` object exactly as we were using our ``svc``
object previously.

Visualizing the results
-----------------------

To visualize the results, we need to:

- first get the support vectors of the SVC and inverse the feature
  selection mechanism
- then, as before, inverse the masking process to retrieve the weights
  and plot them.

.. literalinclude:: ../../examples/02_decoding/plot_haxby_anova_svm.py
    :start-after: # Visualize the results
    :end-before: # Saving the results as a Nifti file may also be important

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_anova_svm_001.png
   :target: ../auto_examples/02_decoding/plot_haxby_anova_svm.html
   :scale: 65

.. seealso::

   * :ref:`plotting`


.. topic:: **Final script**

    The complete script to do an SVM-Anova analysis can be found as
    :ref:`an example <sphx_glr_auto_examples_02_decoding_plot_haxby_anova_svm.py>`.


.. seealso::

   * :ref:`space_net`
   * :ref:`searchlight`


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
    >>> feature_selection = SelectKBest(f_classif, k=4)


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
