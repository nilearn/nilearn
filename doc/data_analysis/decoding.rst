.. for doctests to run, we need to define variables that are define in
   the literal includes
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> y += 1
    >>> session = np.ones_like(y)
    >>> n_samples = len(y)

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

.. figure:: ../auto_examples/images/plot_haxby_stimuli_5.png
   :target: ../auto_examples/plot_haxby_stimuli.html
   :scale: 30
   :align: left

   House stimuli

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
our analysis to the `face` and `house` conditions:

.. literalinclude:: ../../plot_haxby_simple.py
    :start-after: ### Load Target labels ########################################################
    :end-before: ### Prepare the data: apply the mask ##########################################

.. currentmodule:: nilearn.input_data

Then we prepare the fMRI data: we use the :class:`NiftiMasker` to apply the
`mask_vt` mask to the 4D fMRI data, so that its shape becomes (n_samples,
n_features) (see :ref:`mask_4d_2_3d` for a discussion on using masks)


.. literalinclude:: ../../plot_haxby_simple.py
    :start-after: ### Prepare the data: apply the mask ##########################################
    :end-before: ### Prediction function #######################################################

Performing the decoding analysis
====================================

Prediction function: the estimator
-----------------------------------

To perform decoding we construct an estimator, predicting a condition
label **y** given a set **X** of images.

We define here a simple `Support Vector Classification
<http://scikit-learn.org/stable/modules/svm.html>`_ (or SVC) with C=1, and a
linear kernel. We first import the correct module from scikit-learn and we
define the classifier:

.. literalinclude:: ../../plot_haxby_simple.py
    :start-after: ### Prediction function #######################################################
    :end-before: ### Unmasking #################################################################


Need some doc ?

    >>> svc ? # doctest: +SKIP
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

Or go to the `scikit-learn
documentation <http://scikit-learn.org/modules/svm.html>`_
We use a SVC here, but we can use
`many other
classifiers <http://scikit-learn.org/stable/supervised_learning.html>`_


Launching it on real data: fit (train) and predict (test)
----------------------------------------------------------

In scikit-learn, the prediction objects (classifiers, regression) have a very simple API:

- a *fit* function that "learns" the parameters of the model from the data.
  Thus, we need to give some training data to *fit*.
- a *predict* function that "predicts" a target from new data.
  Here, we just have to give the new set of images (as the target should be
  unknown):

.. literalinclude:: ../../plot_haxby_anova_svm.py
        :start-after: ### Fit and predict ###########################################################
        :end-before: ### Visualisation #############################################################

**Warning ! Do not do this at home:** the prediction that we obtain here
is to good to be true (see next paragraph). Here we are just doing a
sanity check.

.. for doctests (smoke testing):
    >>> from sklearn.svm import LinearSVC, SVC
    >>> anova_svc = LinearSVC()

Cross-validation: measuring prediction performance
---------------------------------------------------

However, the last analysis is *wrong*, as we have learned and tested on
the same set of data. We need to use a cross-validation to split the data
into different sets.

In scikit-learn, a cross-validation is simply a function that generates
the indices of the folds within a loop.
Now, we can apply the previously defined *pipeline* with the
cross-validation:

.. literalinclude:: ../../plot_haxby_anova_svm.py
        :start-after: ### Cross validation ########################################################## 
        :end-before: ### Print results #############################################################

.. for doctests:
   >>> cv = 2

But we are lazy people, so there is a specific
function, *cross_val_score* that computes for you the results for the
different folds of cross-validation::

  >>> from sklearn.cross_validation import cross_val_score
  >>> cv_scores = cross_val_score(anova_svc, X, y, cv=cv)

If you are the happy owner of a multiple-processor computer you can
speed up the computation by using n_jobs=-1, which will spread the
computation equally across all processors (but will probably not work
under Windows)::

 >>> cv_scores = cross_val_score(anova_svc, X, y, cv=cv, n_jobs=-1, verbose=10) #doctest: +SKIP

**Prediction accuracy**: We can take a look at the results of the
*cross_val_score* function::

  >>> cv_scores # doctest: +SKIP
  array([ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
          1.        ,  1.        ,  0.94444444,  1.        ,  1.        ,
          1.        ,  1.        ])

This is simply the prediction score for each fold, i.e. the fraction of
correct predictions on the left-out data.


.. topic:: **Exercise**
   :class: green

   Compute the mean prediction accuracy using *cv_scores*

.. topic:: Solution

    >>> classification_accuracy = np.mean(cv_scores)
    >>> classification_accuracy # doctest: +SKIP
    0.99537037037037035

We have a total prediction accuracy of 99% across the different folds.

Visualizing the decoder's weights
---------------------------------

We can visualize the weights of the decoder:

- we first inverse the masking operation, to retrieve a 3D brain volume
  of the SVC's weights.
- we then create a figure and plot as a background the first EPI image
- we plot the SVC's weights after masking the zero values

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

Dimension reduction
-------------------

If we do not start from a mask of the relevant regions, there is a very
large number of voxels and not all are useful for
face vs house prediction. We thus add a `feature selection
<http://scikit-learn.org/stable/modules/feature_selection.html>`_
procedure. The idea is to select the `k` voxels most correlated to the
task.

For this, we need to import the `feature_selection` module and define a
simple F-score
based feature selection (a.k.a. 
`Anova <http://en.wikipedia.org/wiki/Analysis_of_variance#The_F-test>`_),
that we will put before the SVC in a `pipeline`:

.. literalinclude:: ../../plot_haxby_anova_svm.py
        :start-after: ### Dimension reduction #######################################################
        :end-before: ### Fit and predict ###########################################################


Visualizing the results
-------------------------

We can visualize the result of our algorithm:

- we first get the support vectors of the SVC and inverse the feature
  selection mechanism
- we remove the mask
- then we overlay our previously-computed, mean image with our support vectors

.. figure:: ../auto_examples/images/plot_haxby_anova_svm_1.png
   :target: ../auto_examples/plot_haxby_anova_svm.html
   :align: right
   :scale: 65

.. literalinclude:: ../../plot_haxby_anova_svm.py
    :start-after: ### Visualisation #############################################################
    :end-before: ### Cross validation ########################################################## 

.. seealso::

   * :ref:`visualizing`



We can add a line to print the results:

.. literalinclude:: ../../plot_haxby_anova_svm.py
        :start-after: ### Print results #############################################################


.. topic:: **Final script**

    The complete script can be found as 
    :ref:`an example <example_plot_haxby_anova_svm.py>`.
    Now, all you have to do is publish the results :)


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

