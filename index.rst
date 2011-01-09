

****************************************
`Scikits-learn <http://scikit-learn.sourceforge.net/>`_ for fMRI data analysis
****************************************

INRIA Parietal Project Team and scikits-learn folks, \
among which **V. Michel, A. Gramfort, G. Varoquaux, \
F. Pedregosa and B. Thirion**

Thanks to M. Hanke and Y. Halchenko for data and packaging.

Objectives
==========

At the end of this session you will be able to:

  1. Install and use the required tools (Nibabel and scikits-learn).
  2. Load fMRI volumes in python.
  3. Perform a state-of-the-art decoding analysis of fMRI data.
  4. Perform even more sophisticated analyzes of fMRI data.

.. role:: input(strong)



What is Scikits-learn?
---------------------

Scikits-learn is a Python library for machine learning.

Principal features:

- Easy to use.
- Easy to install.
- Well documented.
- Provide standard machine learning methods for non-experts.

Technical choices:

- Python: general-purpose, high-level language.
- Simple data structures (numpy arrays).
- BSD license : reuse even in commercial settings



Installation of the required materials
--------------------------------------

The data
^^^^^^^^

We use here the *Haxby 2001* dataset  [Haxby et al. (2001)], that has been
reanalyzed in [Hanson et al (2004), O'Toole et al. (2005)].

In short, we have:

  - 8 objects presented once in 12 sessions
  - 864 volumes containing 39912 voxels

Additional information : http://www.sciencemag.org/content/293/5539/2425

Download the data::

  $ wget http://www.pymvpa.org/files/pymvpa_exampledata.tar.bz2

decompress them::

  $ tar xjfv pymvpa_exampledata.tar.bz2

and go to the data directory::

  $ cd pymvpa-exampledata


Nibabel
^^^^^^^^

Easy to use reader of ANALYZE (plain, SPM99, SPM2), GIFTI, NIfTI1, MINC
(former PyNIfTI)::

  $ easy_install nibabel

and if you can not be root::

  $ easy_install --prefix=~/usr nibabel


Scikits-learn
^^^^^^^^

(Quick) installation::

  $ easy_install scikits.learn













First step: looking at the data (always interesting...)
=======================================================


Now, launch ipython::

  $ ipython


First, we load the data. We have to import the nibabel module and the numpy
module (basic array manipulations):

    >>> import nibabel as ni
    >>> import numpy as np

... load the fMRI volumes (what we will call X)

    >>> X = ni.load("bold.nii.gz").get_data()

... the mask

    >>> mask = ni.load("mask.nii.gz").get_data()

... and the target (that we will call y), and the session index:

    >>> y, session = np.loadtxt("attributes.txt").astype("int").T


Check the dimensions of the data:

    >>> X.shape
    (40, 64, 64, 1452)
    >>> mask.shape
    (40, 64, 64)

Mask the data X and transpose the matrix, so that its shape becomes (n_samples,
n_features):

    >>> X = X[mask!=0].T
    >>> X.shape
    (1452, 39912)

and we (hopefully) retrieve the correct number of voxels (39912).

Finally, we can detrend the data (for each session separately):

    >>> from scipy import signal
    >>> for s in np.unique(session):
            X[session==s] = signal.detrend(X[session==s], axis=0)

Now, we take a look to the target y:

    >>> y.shape
    (1452,)
    >>> np.unique(y)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])

where 0 is rest period, and [1..8] is the label of each object.


Exercise
^^^^^^^^

  1. Extract the period of activity from the data (i.e. remove the remainder).


Solution
^^^^^^^^

    >>> X, y, session = X[y!=0], y[y!=0], session[y!=0]

We can check that:

    >>> n_samples, n_features = X.shape
    >>> n_samples
    864
    >>> n_features
    39912

and we have the 8 conditions:

    >>> n_conditions = np.size(np.unique(y))
    >>> n_conditions
    8




Second step: basic (but state of the art) decoding analysis
=============================================================

In a decoding analysis we construct a model, so that one can predict
a value of y given a set X of images.

Prediction function
-------------------

We define here a simple Support Vector Classification (or SVC) with C=1, and a
linear kernel. We first import the correct module from scikits-learn:

    >>> from scikits.learn.svm import SVC

and we define the classifier:

    >>> clf = SVC(kernel='linear', C=1.)
    >>> clf
    SVC(kernel='linear', C=1.0, probability=False, degree=3, coef0=0.0,
    eps=0.001, cache_size=100.0, shrinking=True, gamma=0.0)

Need some doc ?

    >>> clf ?
    Type:             SVC
    Base Class:       <class 'scikits.learn.svm.libsvm.SVC'>
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

Or go to the `scikits-learn
documentation <http://scikit-learn.sourceforge.net/modules/svm.html>`_

We use a SVC here, but we can use
`many other
classifiers <http://scikit-learn.sourceforge.net/supervised_learning.html>`_


Dimension reduction
-------------------

But a classification with few samples and many features is plagued by the
*curse of dimensionality*. Let us add a feature selection procedure.

For this, we need to import the correct module:

    >>> from scikits.learn.feature_selection import SelectKBest, f_classif

and define a simple F-score based feature selection (a.k.a. Anova):

    >>> feature_selection = SelectKBest(f_classif, k=500)
    >>> feature_selection
    SelectKBest(k=500, score_func=<function f_classif at 0x8c93684>)


We have our classifier (SVC), our feature selection (SelectKBest), and now, we
can plug them together in a *pipeline* that performs the two operations
successively:


    >>> from scikits.learn.pipeline import Pipeline
    >>> anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])
    >>> anova_svc
    Pipeline(steps=[('anova', SelectKBest(k=500, score_func=<function
    f_classif at 0x8c93684>)), ('svc', SVC(kernel='linear', C=1.0,
    probability=False, degree=3, coef0=0.0, eps=0.001,
    cache_size=100.0, shrinking=True, gamma=0.0))])

We use a univariate feature selection, but we can use other dimension
reduction such as
`RFE
<http://scikit-learn.sourceforge.net/modules/generated/scikits.learn.
feature_selection.rfe.RFE.html>`_




Third step: launch it on real data
==================================

Fit (train) and predict (test):
-------------------------------

In scikits-learn, prediction function have a very simple API:

    - a *fit* function that "learn" the parameters of the model from the data.
    Thus, we need to give some training data to *fit*

    >>> anova_svc.fit(X, y)
    Pipeline(steps=[('anova', SelectKBest(k=500, score_func=<function f_classif
    at 0x8c93684>)), ('svc', SVC(kernel='linear', C=1.0, probability=False,
    degree=3, coef0=0.0, eps=0.001,
    cache_size=100.0, shrinking=True, gamma=0.0))])

    - a *predict* function that "predict" a target from new data.
    Here, we just have to give the new set of images (as the target should be
    unknown).

    >>> y_pred = anova_svc.predict(X)
    >>> y_pred.shape
    (864,)
    >>> X.shape
    (864, 39912)

    **Warning ! Do not do this at home !** the score that we obtain here is 
    heavily biased (see next paragraph). This is used here to check that 
    we have one predicted value per image.

    Note that you could have done this in only 1 line:

    >>> y_pred = anova_svc.fit(X, y).predict(X)

Cross-validation
----------------

    However, the last analysis is *wrong*, as we have learned and testeddd 
    on the same set of data.
    We need to use a cross-validation to split the data into different sets.

    Let us define a Leave-one-session-out cross-validation:

    >>> from scikits.learn.cross_val import LeaveOneLabelOut
    >>> cv = LeaveOneLabelOut(session)

    In scikits-learn, a cross-validation is simply a function that generates
    the index of the folds within a loop.
    So, now, we can apply the previously defined *pipeline* with the
    cross-validation::

    >>> cv_scores = [] # will store the number of correct predictions in each
fold
    >>> for train, test in cv:
    >>>     y_pred = anova_svc.fit(X[train], y[train]).predict(X[test])
    >>>     cv_scores.append(np.sum(y_pred == y[test]))


    But we are lazy people, so there is a specific
    function, *cross_val_score* that computes for you the results for the
    different folds of cross-validation:

    >>> from scikits.learn.cross_val import cross_val_score
    >>> cv_scores = cross_val_score(anova_svc, X, y, cv=cv, n_jobs=1,
                            verbose=1, iid=True)

    n_jobs = 1 means that the computation is not parallel.
    But, if you are the happy owner of a multiple processors computer, you can
    even speed up the computation:

    >>> cv_scores = cross_val_score(anova_svc, X, y, cv=cv, n_jobs=4,
                            verbose=1, iid=True)


Prediction accuracy
-------------------

    We can take a look to the results of the *cross_val_score* function:

    >>> cv_scores
    array([ 60.,  59.,  65.,  49.,  57.,  56.,  52.,  44.,  54.,  47.,  49.,
        51.])

    This is simply the number of correct predictions for each fold.


Exercise
^^^^^^^^

  1. Compute the mean prediction accuracy using *cv_scores*


Solution
^^^^^^^^

    >>> classification_accuracy = np.sum(cv_scores) / float(n_samples)
    >>> classification_accuracy
    0.74421296296296291

We have a total prediction accuracy of 74% across the different folds.


We can add a line to print the results:

    >>> print "Classification accuracy: %f" % classification_accuracy, \
        " / Chance level: %f" % (1. / n_conditions)
    Classification accuracy: 0.744213  / Chance level: 0.125000






Final script
============

An thus, the global script is::

    ### All the imports
    import numpy as np
    from scipy import signal
    import nibabel as ni
    from scikits.learn.svm import SVC
    from scikits.learn.feature_selection import SelectKBest
    from scikits.learn.feature_selection import f_classif
    from scikits.learn.pipeline import Pipeline
    from scikits.learn.cross_val import LeaveOneLabelOut
    from scikits.learn.cross_val import cross_val_score

    ### Load data
    y, session = np.loadtxt("attributes.txt").astype("int").T
    X = ni.load("bold.nii.gz").get_data()
    mask = ni.load("mask.nii.gz").get_data()

    # Process the data in order to have a two-dimensional design matrix X of
    # shape (nb_samples, nb_features).
    X = X[mask!=0].T

    # Detrend data on each session independently
    for s in np.unique(session):
        X[session==s] = signal.detrend(X[session==s], axis=0)

    # Remove volumes corresponding to rest
    X, y, session = X[y!=0], y[y!=0], session[y!=0]
    n_samples, n_features = X.shape
    n_conditions = np.size(np.unique(y))

    ### Define the prediction function to be used.
    # Here we use a Support Vector Classification, with a linear kernel and C=1
    clf = SVC(kernel='linear', C=1.)

    ### Define the dimension reduction to be used.
    # Here we use a classical univariate feature selection based on F-test,
    # namely Anova. We set the number of features to be selected to 500
    feature_selection = SelectKBest(f_classif, k=500)

    ### We combine the dimension reduction and the prediction function
    anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])

    ### Define the cross-validation scheme used for validation.
    # Here we use a LeaveOneLabelOut cross-validation on the session, which
    # corresponds to a leave-one-session-out
    cv = LeaveOneLabelOut(session)

    ### Compute the prediction accuracy for the different folds (i.e. session)
    cv_scores = cross_val_score(anova_svc, X, y, cv=cv, n_jobs=-1,
                                verbose=1, iid=True)

    ### Return the corresponding mean prediction accuracy
    classification_accuracy = np.sum(cv_scores) / float(n_samples)
    print "Classification accuracy: %f" % classification_accuracy, \
        " / Chance level: %f" % (1. / n_conditions)


Now, you just have to publish the results :)







Going further with scikits-learn
===================================

We have seen a very simple analysis with scikits-learn.


`Other prediction functions with
Scikits-learn <http://scikit-learn.sourceforge.net/modules/glm.html>`_

`Unsupervised learning (e.g. clustering, PCA, ICA) with
Scikits-learn <http://scikit-learn.sourceforge.net/modules/clustering.html>`_




Example of the simplicity of scikits-learn
-----------------------------------------

One of the major assets of scikits-learn is the real simplicity of use.



Changing the prediction function
--------------------------------

We now see how one can easily change the prediction function, if needed.
We can try the Linear Discriminant Analysis
(LDA) `<http://scikit-learn.sourceforge.net/auto_examples/plot_lda_qda.html>`_

Import the module:

    >>> from scikits.learn.lda import LDA

Construct the new prediction function and use it in a pipeline:

    >>> lda = LDA()
    >>> anova_lda = Pipeline([('anova', feature_selection), ('LDA', lda)])

and recompute the cross-validation score:

    >>> cv_scores = cross_val_score(anova_lda, X, y, cv=cv, n_jobs=4,
                            verbose=1, iid=True)
    >>> classification_accuracy = np.sum(cv_scores) / float(n_samples)
    >>> print "Classification accuracy: %f" % classification_accuracy, \
        " / Chance level: %f" % (1. / n_conditions)
    Classification accuracy: 0.728009   / Chance level: 0.125000





Changing the feature selection
------------------------------
Let's say that you want a more sophisticated feature selection, for example a
`Recursive Feature Elimination
(RFE) <http://scikit-learn.sourceforge.net/modules/generated/scikits.learn.
feature_selection.rfe.RFE.html>`_

Import the module:

    >>> from scikits.learn.feature_selection import RFE

Construct your new fancy selection:

    >>> rfe = RFE(SVC(kernel='linear', C=1.), n_features=50, percentage=0.25)

and create a new pipeline:

    >>> rfe_svc = Pipeline([('rfe', rfe), ('svc', clf)])

and recompute the cross-validation score:

    >>> cv_scores = cross_val_score(rfe_svc, X, y, cv=cv, n_jobs=4,
                            verbose=1, iid=True)

But, be aware that this can take A WHILE...






Any questions ?
===============


 `http://scikit-learn.sourceforge.net/ <http://scikit-learn.sourceforge.net/>`_





