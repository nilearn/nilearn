.. for doctests to run, we need to define variables that are define in
   the literal includes
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> fmri_masked  = iris.data
    >>> target = iris.target
    >>> session = np.ones_like(target)
    >>> n_samples = len(target)

.. _fmri_decoding:

================================================================================
Decoding: predicting behavior or phenotype from brain images
================================================================================

Decoding consists in predicting external variables such as behavioral or
phenotypic variables from brain image. This page is a tutorial articulated on
the analysis of the Haxby 2001 dataset. It shows how to:

1. Load fMRI volumes in Python.
2. Perform a state-of-the-art decoding analysis of fMRI data.
3. Perform even more sophisticated analyses of fMRI data.

.. contents:: **Chapters contents**
    :local:
    :depth: 1


Data loading and preparation
================================

The Haxby 2001 experiment
-------------------------

Subjects are presented visual stimuli from different categories. We are
going to predict which category the subject is seeing from the fMRI
activity recorded in masks of the ventral stream. Significant prediction
shows that the signal in the region contains information on the
corresponding category.

.. figure:: ../auto_examples/images/plot_haxby_stimuli_4.png
   :target: ../auto_examples/plot_haxby_stimuli.html
   :scale: 30
   :align: left

   Face stimuli

.. figure:: ../auto_examples/images/plot_haxby_stimuli_2.png
   :target: ../auto_examples/plot_haxby_stimuli.html
   :scale: 30
   :align: left

   Cat stimuli

.. figure:: ../auto_examples/images/plot_haxby_masks_1.png
   :target: ../auto_examples/plot_haxby_masks.html
   :scale: 30
   :align: left

   Masks

.. figure:: ../auto_examples/images/plot_haxby_full_analysis_1.png
   :target: ../auto_examples/plot_haxby_full_analysis.html
   :scale: 35
   :align: left

   Decoding scores per mask


Loading the data into Python
-----------------------------

Launch ipython::

  $ ipython -pylab

First, load the data using nilearn's data downloading function,
:func:`nilearn.datasets.fetch_haxby_simple`:

.. literalinclude:: ../../plot_haxby_simple.py
    :start-after: ### Load haxby dataset ########################################################
    :end-before: ### Load Target labels ########################################################

The ``data`` object has several entries that contain paths to the files
downloaded on the disk::

    >>> print data # doctest: +SKIP
    {'anat': ['/home/varoquau/dev/nilearn/nilearn_data/haxby2001/subj1/anat.nii.gz'],
    'func': ['/home/varoquau/dev/nilearn/nilearn_data/haxby2001/subj1/bold.nii.gz'],
    'mask_face': ['/home/varoquau/dev/nilearn/nilearn_data/haxby2001/subj1/mask8b_face_vt.nii.gz'],
    'mask_face_little': ['/home/varoquau/dev/nilearn/nilearn_data/haxby2001/subj1/mask8_face_vt.nii.gz'],
    'mask_house': ['/home/varoquau/dev/nilearn/nilearn_data/haxby2001/subj1/mask8b_house_vt.nii.gz'],
    'mask_house_little': ['/home/varoquau/dev/nilearn/nilearn_data/haxby2001/subj1/mask8_house_vt.nii.gz'],
    'mask_vt': ['/home/varoquau/dev/nilearn/nilearn_data/haxby2001/subj1/mask4_vt.nii.gz'],
    'session_target': ['/home/varoquau/dev/nilearn/nilearn_data/haxby2001/subj1/labels.txt']}


We load the behavioral labels from the corresponding text file and limit
our analysis to the `face` and `cat` conditions:

.. literalinclude:: ../../plot_haxby_simple.py
    :start-after: ### Load Target labels ########################################################
    :end-before: ### Prepare the data: apply the mask ##########################################

.. currentmodule:: nilearn.input_data

Then we prepare the fMRI data: we use the :class:`NiftiMasker` to apply the
`mask_vt` mask to the 4D fMRI data, so that its shape becomes (n_samples,
n_features) (see :ref:`mask_4d_2_3d` for a discussion on using masks).


.. note::

   seemingly minor data preparation can matter a lot on the final score,
   for instance standardizing the data.


.. literalinclude:: ../../plot_haxby_simple.py
    :start-after: ### Prepare the data: apply the mask ##########################################
    :end-before: ### Prediction ################################################################

.. seealso::

   * :ref:`loading_data`
   * :ref:`masking`



Performing the decoding analysis
====================================

The prediction engine
-----------------------

An estimator object
....................

To perform decoding we construct an estimator, predicting a condition
label **y** given a set **X** of images.

We use here a simple `Support Vector Classification
<http://scikit-learn.org/stable/modules/svm.html>`_ (or SVC) with a
linear kernel. We first import the correct module from scikit-learn and we
define the classifier, :class:`sklearn.svm.SVC`:

.. literalinclude:: ../../plot_haxby_simple.py
    :start-after: # Here we use a Support Vector Classification, with a linear kernel
    :end-before: # And we run it


The documentation of the object details all parameters. In IPython, it
can be displayed as follows::

    In [10]: svc?
    Type:             SVC
    Base Class:       <class 'sklearn.svm.libsvm.SVC'>
    String Form:
    SVC(kernel=linear, C=1.0, probability=False, degree=3, coef0=0.0, eps=0.001,
    cache_size=100.0, shrinking=True, gamma=0.0)
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
   <http://scikit-learn.org/modules/svm.html>`_


Applying it to data: fit (train) and predict (test)
....................................................

In scikit-learn, the prediction objects have two important methods:

- a *fit* function that "learns" the parameters of the model from the data.
  Thus, we need to give some training data to *fit*.
- a *predict* function that "predicts" a target from new data.
  Here, we just have to give the new set of images (as the target should be
  unknown):

.. literalinclude:: ../../plot_haxby_simple.py
    :start-after: # And we run it
    :end-before: ### Cross-validation ##########################################################

.. warning::

    **Do not predict on data used by the fit:** the prediction that we obtain here
    is to good to be true (see next paragraph). Here we are just doing a sanity
    check.

.. for doctests (smoke testing):
    >>> from sklearn.svm import SVC
    >>> svc = SVC()

Measuring prediction performance
---------------------------------

Cross-validation
.................

However, the last analysis is *wrong*, as we have learned and tested on
the same set of data. We need to use a cross-validation to split the data
into different sets, called "folds", in a `K-Fold strategy
<http://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`_.

We use a cross-validation object,
:class:`sklearn.cross_validation.KFold`, that simply generates the
indices of the folds within a loop.

.. literalinclude:: ../../plot_haxby_simple.py
    :start-after: ### Cross-validation ##########################################################
    :end-before: ### Unmasking #################################################################


.. for doctests:
   >>> cv = 2

There is a specific function,
:func:`sklearn.cross_validation.cross_val_score` that computes for you
the score for the different folds of cross-validation::

  >>> from sklearn.cross_validation import cross_val_score
  >>> cv_scores = cross_val_score(svc, fmri_masked, target, cv=cv)

You can speed up the computation by using n_jobs=-1, which will spread
the computation equally across all processors (but will probably not work
under Windows)::

 >>> cv_scores = cross_val_score(svc, fmri_masked, target, cv=cv, n_jobs=-1, verbose=10) #doctest: +SKIP

**Prediction accuracy**: We can take a look at the results of the
*cross_val_score* function::

  >>> print cv_scores # doctest: +SKIP
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

In these regards, leave one out is often one of the worst options.

Here, in the Haxby example, we are going to leave a session out, in order
to have a test set independent from the train set. For this, we are going
to use the session label, present in the behavioral data file, and
:class:`sklearn.cross_validation.LeaveOneLabelOut`::

    >>> from sklearn.cross_validation import LeaveOneLabelOut
    >>> session_label = labels['chunks'] # doctest: +SKIP
    >>> # We need to remember to remove the rest conditions
    >>> session_label = session_label[condition_mask] # doctest: +SKIP
    >>> cv = LeaveOneLabelOut(labels=session_label) # doctest: +SKIP
    >>> cv_scores = cross_val_score(svc, fmri_masked, target, cv=cv)
    >>> print cv_scores
    [ 1.          0.61111111  0.94444444  0.88888889  0.88888889  0.94444444
      0.72222222  0.94444444  0.5         0.72222222  0.5         0.55555556]

.. topic:: **Exercise**
   :class: green

   Compute the mean prediction accuracy using *cv_scores*

.. topic:: Solution

    >>> classification_accuracy = np.mean(cv_scores)
    >>> classification_accuracy # doctest: +SKIP
    0.76851851851851849

We have a total prediction accuracy of 77% across the different sessions.

Choice of the prediction accuracy measure
..........................................

The default metric used for measuring errors is the accuracy score, i.e.
the number of total errors. It is not always a sensible metric,
especially in the case of very imbalanced classes, as in such situations
choosing the dominant class can achieve a low number of errors.

Other metrics, such as the f1-score, can be used::

    >>> cv_scores = cross_val_score(svc, fmri_masked, target, cv=cv,  scoring='f1')

.. seealso::

   the `list of scoring options
   <http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values>`_

Measuring the chance level
...........................

**Dummy estimators**: The simplest way to measure prediction performance
at chance, is to use a dummy classifier,
:class:`sklearn.dummy.DummyClassifier`::

    >>> from sklearn.dummy import DummyClassifier
    >>> null_cv_scores = cross_val_score(DummyClassifier(), fmri_masked, target, cv=cv)

**Permutation testing**: A more controlled way, but slower, is to do
permutation testing on the labels, with
:func:`sklearn.cross_validation.permutation_test_score`::

  >>> from sklearn.cross_validation import permutation_test_score
  >>> null_cv_scores = permutation_test_score(svc, fmri_masked, target, cv=cv)

|

.. topic:: **Putting it all together**

    The :ref:`ROI-based decoding example
    <example_plot_haxby_full_analysis.py>` does a decoding analysis per
    mask, giving the f1-score of the prediction for each object.

    It uses all the notions presented above, with ``for`` loop to iterate
    over masks and categories and Python dictionnaries to store the
    scores.


.. figure:: ../auto_examples/images/plot_haxby_masks_1.png
   :target: ../auto_examples/plot_haxby_masks.html
   :scale: 55
   :align: left

   Masks


.. figure:: ../auto_examples/images/plot_haxby_full_analysis_1.png
   :target: ../auto_examples/plot_haxby_full_analysis.html
   :scale: 70
   :align: left



Visualizing the decoder's weights
---------------------------------

We can visualize the weights of the decoder:

- we first inverse the masking operation, to retrieve a 3D brain volume
  of the SVC's weights.
- we then create a figure and plot as a background the first EPI image
- finally we plot the SVC's weights after masking the zero values

.. figure:: ../auto_examples/images/plot_haxby_simple_1.png
   :target: ../auto_examples/plot_haxby_simple.html
   :align: right
   :scale: 65

.. literalinclude:: ../../plot_haxby_simple.py
    :start-after: ### Unmasking #################################################################
    :end-before: ### Visualize the mask ########################################################


.. seealso::

   * :ref:`visualizing`


Decoding without a mask: Anova-SVM
===================================

Dimension reduction with feature selection
-------------------------------------------

If we do not start from a mask of the relevant regions, there is a very
large number of voxels and not all are useful for
face vs cat prediction. We thus add a `feature selection
<http://scikit-learn.org/stable/modules/feature_selection.html>`_
procedure. The idea is to select the `k` voxels most correlated to the
task.

For this, we need to import the :mod:`sklearn.feature_selection` module and use
:func:`sklearn.feature_selection.f_classif`, a simple F-score
based feature selection (a.k.a.
`Anova <http://en.wikipedia.org/wiki/Analysis_of_variance#The_F-test>`_),
that we will put before the SVC in a `pipeline`
(:class:`sklearn.pipeline.Pipeline`):

.. literalinclude:: ../../plot_haxby_anova_svm.py
    :start-after: ### Dimension reduction #######################################################
    :end-before: ### Visualization #############################################################
We can use our ``anova_svc`` object exactly as we were using our ``svc``
object previously.

Visualizing the results
-------------------------

To visualize the results, we need to:

- first get the support vectors of the SVC and inverse the feature
  selection mechanism
- then, as before, inverse the masking process to retrieve the weights
  and plot them.

.. figure:: ../auto_examples/images/plot_haxby_anova_svm_1.png
   :target: ../auto_examples/plot_haxby_anova_svm.html
   :align: right
   :scale: 65

.. literalinclude:: ../../plot_haxby_anova_svm.py
    :start-after: ### Visualization #############################################################
    :end-before: ### Cross validation ##########################################################

.. seealso::

   * :ref:`visualizing`


.. topic:: **Final script**

    The complete script to do an SVM-Anova analysis can be found as
    :ref:`an example <example_plot_haxby_anova_svm.py>`.


.. seealso::

   * :ref:`searchlight`
   * :ref:`decoding_simulated`

Going further with scikit-learn
===================================

We have seen a very simple analysis with scikit-learn, but it may be
interesting to explore the `wide variety of supervised learning
algorithms in the scikit-learn
<http://scikit-learn.org/stable/supervised_learning.html>`_.

Changing the prediction engine
--------------------------------

.. for doctest:
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> feature_selection = SelectKBest(f_classif, k=4)
    >>> clf = LinearSVC()

We now see how one can easily change the prediction engine, if needed.
We can try Fisher's `Linear Discriminant Analysis (LDA)
<http://scikit-learn.org/auto_examples/plot_lda_qda.html>`_

Import the module::

    >>> from sklearn.lda import LDA

Construct the new estimator object and use it in a pipeline::

    >>> from sklearn.pipeline import Pipeline
    >>> lda = LDA()
    >>> anova_lda = Pipeline([('anova', feature_selection), ('LDA', lda)])

and recompute the cross-validation score::

    >>> cv_scores = cross_val_score(anova_lda, X, y, cv=cv, verbose=1)
    >>> classification_accuracy = np.mean(cv_scores)
    >>> print "Classification accuracy: %f" % classification_accuracy, \
    ...     " / Chance level: %f" % (1. / n_conditions) # doctest: +SKIP
    Classification accuracy: 1.000000   / Chance level: 0.500000


Changing the feature selection
------------------------------

Let's say that you want a more sophisticated feature selection, for example a
`Recursive Feature Elimination (RFE)
<http://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination>`_

Import the module::

    >>> from sklearn.feature_selection import RFE

Construct your new fancy selection::

    >>> rfe = RFE(SVC(kernel='linear', C=1.), 50, step=0.25)

and create a new pipeline::

    >>> rfe_svc = Pipeline([('rfe', rfe), ('svc', clf)])

and recompute the cross-validation score::

    >>> cv_scores = cross_val_score(rfe_svc, X, y, cv=cv, n_jobs=-1,
    ...     verbose=True) # doctest: +SKIP

But, be aware that this can take A WHILE...

|

.. seealso::

  * The :ref:`supervised_learning` section of the nilearn documentation.

  * The `scikit-learn documentation <http://scikit-learn.org>`_
    has very detailed explanations on a large variety of estimators and
    machine learning techniques. To become better at decoding, you need
    to study it.
