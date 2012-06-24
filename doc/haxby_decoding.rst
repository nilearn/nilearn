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


================================================================================
fMRI decoding: predicting which objects a subject is viewing 
================================================================================

.. topic:: Objectives

   At the end of this tutorial you will be able to:

    1. Load fMRI volumes in Python.
    2. Perform a state-of-the-art decoding analysis of fMRI data.
    3. Perform even more sophisticated analyzes of fMRI data.

.. role:: input(strong)

The haxby dataset
===================

We use here the *Haxby 2001* dataset used in `Distributed and Overlapping Representations 
of Faces and Objects in Ventral Temporal Cortex
<http://www.sciencemag.org/content/293/5539/2425.full>`_, Haxby et al. (2001),
that has been reanalyzed in `Combinatorial codes in ventral temporal lobe for object
recognition: Haxby (2001) revisited: is there a “face” area?
<http://www.sciencedirect.com/science/article/pii/S105381190400299X>`_, Hanson et al (2004)
and `Partially Distributed Representations of Objects and Faces in Ventral Temporal Cortex
<http://www.mitpressjournals.org/doi/abs/10.1162/0898929053467550?journalCode=jocn>`_, O'Toole et al. (2005).

In short, we have:

  - 8 objects presented once in 12 sessions
  - 864 volumes containing 39912 voxels

Additional information : http://www.sciencemag.org/content/293/5539/2425

First step: looking at the data
================================


Now, launch ipython::

  $ ipython -pylab

First, we load the data. Nisl provides an easy way to download and preprocess
data. Simply call:

.. literalinclude:: ../plot_haxby_decoding.py
        :start-after: ### Load Haxby dataset ########################################################
        :end-before: ### Preprocess data ########################################################### 

Then we preprocess the data to get make it handier:

- compute the mean of the image to replace anatomic data
- check its dimension (1452 trials of 40x64x64 voxels)
- mask the data X and transpose the matrix, so that its shape becomes
  (n_samples, n_features)
- finally detrend the data for each session

.. literalinclude:: ../plot_haxby_decoding.py
        :start-after: ### Preprocess data ########################################################### 
        :end-before: ### Restrict to faces and houses ##############################################

.. topic:: **Exercise**
   :class: green

   1. Extract the period of activity from the data (i.e. remove the remainder).

.. topic:: Solution

    As 'y == 0' in rest, we want to keep only time points for which 
    `y != 0`::

     >>> X, y, session = X[y!=0], y[y!=0], session[y!=0]

Here, we limit to face and house conditions:

.. literalinclude:: ../plot_haxby_decoding.py
        :start-after: ### Restrict to faces and houses ##############################################
        :end-before: ### Prediction function #######################################################

Second step: decoding analysis
================================

In a decoding analysis we construct a model, so that one can predict
a value of y given a set X of images.

Prediction function
-------------------

We define here a simple Support Vector Classification (or SVC) with C=1,
and a linear kernel. We first import the correct module from
scikit-learn and we define the classifier:

.. literalinclude:: ../plot_haxby_decoding.py
        :start-after: ### Prediction function #######################################################
        :end-before: ### Dimension reduction #######################################################

Need some doc ?

    >>> clf ? # doctest: +SKIP
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
classifiers <http://scikit-learn.org/supervised_learning.html>`_


Dimension reduction
-------------------

But a classification with few samples and many features is plagued by the
*curse of dimensionality*. Let us add a feature selection procedure.

For this, we need to import the correct module and define a simple F-score
based feature selection (a.k.a. Anova):

.. literalinclude:: ../plot_haxby_decoding.py
        :start-after: ### Dimension reduction #######################################################
        :end-before: ### Fit and predict ###########################################################

We use a univariate feature selection, but we can use other dimension
reduction such as `RFE
<http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.rfe.RFE.html>`_

Third step: launch it on real data
==================================

Fit (train) and predict (test)
-------------------------------

In scikit-learn, prediction function have a very simple API:

  - a *fit* function that "learn" the parameters of the model from the data.
    Thus, we need to give some training data to *fit*
  - a *predict* function that "predict" a target from new data.
    Here, we just have to give the new set of images (as the target should be
    unknown):

.. literalinclude:: ../plot_haxby_decoding.py
        :start-after: ### Fit and predict ###########################################################
        :end-before: ### Visualisation #############################################################

**Warning ! Do not do this at home !** the score that we obtain here is 
heavily biased (see next paragraph). This is used here to check that 
we have one predicted value per image.

.. for doctests (smoke testing):
    >>> from sklearn.svm import LinearSVC, SVC
    >>> anova_svc = LinearSVC()

Note that you could have done this in only 1 line::

    >>> y_pred = anova_svc.fit(X, y).predict(X)

Visualisation
-------------

We can visualize the result of our algorithm:

  - we first get the support vectors of the SVC and revert the feature
    selection mechanism
  - we remove the mask
  - then we overlay our mean image computed before with our support vectors

.. figure:: auto_examples/images/plot_haxby_decoding_1.png
   :target: auto_examples/plot_haxby_decoding.html
   :align: center
   :scale: 60

.. literalinclude:: ../plot_haxby_decoding.py
        :start-after: ### Visualisation #############################################################
        :end-before: ### Cross validation ########################################################## 


Cross-validation
----------------

However, the last analysis is *wrong*, as we have learned and tested on
the same set of data. We need to use a cross-validation to split the data
into different sets.

In scikit-learn, a cross-validation is simply a function that generates
the index of the folds within a loop.
So, now, we can apply the previously defined *pipeline* with the
cross-validation:

.. literalinclude:: ../plot_haxby_decoding.py
        :start-after: ### Cross validation ########################################################## 
        :end-before: ### Print results #############################################################

.. for doctests:
   >>> cv = 2

But we are lazy people, so there is a specific
function, *cross_val_score* that computes for you the results for the
different folds of cross-validation::

  >>> from sklearn.cross_validation import cross_val_score
  >>> cv_scores = cross_val_score(anova_svc, X, y, cv=cv, verbose=10)

If you are the happy owner of a multiple processors computer you can
speed up the computation by using n_jobs=-1, which will spread the
computation equally across all processors (this will probably not work
under Windows)::

 >>> cv_scores = cross_val_score(anova_svc, X, y, cv=cv, n_jobs=-1, verbose=10)


Prediction accuracy
-------------------

We can take a look to the results of the *cross_val_score* function::

  >>> cv_scores # doctest: +SKIP
  array([ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
          1.        ,  1.        ,  0.83333333,  1.        ,  1.        ,
          1.        ,  1.        ])

This is simply the prediction score for each fold, i.e. the fraction of
correct predictions on the left-out data.


.. topic:: **Exercise**
   :class: green

   1. Compute the mean prediction accuracy using *cv_scores*

.. topic:: Solution

    >>> classification_accuracy = np.mean(cv_scores)
    >>> classification_accuracy # doctest: +SKIP
    0.74421296296296291

We have a total prediction accuracy of 74% across the different folds.


We can add a line to print the results:

.. literalinclude:: ../plot_haxby_decoding.py
        :start-after: ### Print results #############################################################


.. topic:: Final script

    The complete script can be found as 
    :ref:`an example <example_tutorial_plot_haxby_decoding.py>`
    Now, you just have to publish the results :)

Going further with scikit-learn
===================================

We have seen a very simple analysis with scikit-learn.


`Other prediction functions for supervised learning: Linear models,
Support vector machines, Nearest Neighbor, etc.
<http://scikit-learn.org/supervised_learning.html>`_

`Unsupervised learning (e.g. clustering, PCA, ICA) with
Scikit-learn <http://scikit-learn.org/modules/clustering.html>`_

Example of the simplicity of scikit-learn
-----------------------------------------

One of the major assets of scikit-learn is the real simplicity of use.

Changing the prediction function
--------------------------------

.. for doctest:
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> feature_selection = SelectKBest(f_classif, k=4)
    >>> clf = LinearSVC()

We now see how one can easily change the prediction function, if needed.
We can try the Linear Discriminant Analysis
(LDA) `<http://scikit-learn.org/auto_examples/plot_lda_qda.html>`_

Import the module::

    >>> from sklearn.lda import LDA

Construct the new prediction function and use it in a pipeline::

    >>> from sklearn.pipeline import Pipeline
    >>> lda = LDA()
    >>> anova_lda = Pipeline([('anova', feature_selection), ('LDA', lda)])

and recompute the cross-validation score::

    >>> cv_scores = cross_val_score(anova_lda, X, y, cv=cv, verbose=1)
    >>> classification_accuracy = np.sum(cv_scores) / float(n_samples)
    >>> print "Classification accuracy: %f" % classification_accuracy, \
    ...     " / Chance level: %f" % (1. / n_conditions) # doctest: +SKIP
    Classification accuracy: 0.728009   / Chance level: 0.125000


Changing the feature selection
------------------------------

Let's say that you want a more sophisticated feature selection, for example a
`Recursive Feature Elimination
(RFE) <http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.rfe.RFE.html>`_

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

