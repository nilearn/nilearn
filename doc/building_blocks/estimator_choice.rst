
.. _estimator_choice:

============================================
Considerations on the choice of an estimator
============================================

This page gives a few simple considerations on the choice of an estimator.
It is slightly oriented towards a *decoding* application, that is the
prediction of external variables such as behavior or clinical traits from
brain images. For a didactic introduction to decoding with nilearn, see
the :ref:`dedicated section of the nilearn documentation <fmri_decoding>`.

Predictions: regression, classification and multi-class
========================================================


Regression
-----------

A regression problem is a learning task in which the variable to predict
--that we often call ``y``-- is a continuous value, such as an age.
Encoding models [1]_ typically call for regressions.

.. [1]

   Naselaris et al, Encoding and decoding in fMRI, NeuroImage Encoding
   and decoding in fMRI.2011 http://www.ncbi.nlm.nih.gov/pubmed/20691790

Classification: two classes or multi-class
-------------------------------------------

A classification task consists in predicting a *class* label for each
observation. In other words, the variable to predict is categorical.

Often classification is performed between two classes, but it may well be
applied to multiple classes, in which case it is known as a multi-class
problem. It is important to keep in mind that the larger the number of
classes, the harder the prediction problem.

Some estimators support multi-class prediction out of the box, but many
work by dividing the multi-class problem in a set of two class problems.
There are two noteworthy strategies:

:One versus All:

    :class:`sklearn.multiclass.OneVsRestClassifier`
    An estimator is trained to distinguish each class from all the others,
    and during prediction, the final decision is taken by a vote across
    the different estimators.

:One versus One:

    :class:`sklearn.multiclass.OneVsOneClassifier`
    An estimator is trained to distinguish each pair of classes,
    and during prediction, the final decision is taken by a vote across
    the different estimators.

The "One vs One" strategy is more computationally costly than the "One
vs All". The former scales as the square of the number of classes,
whereas the former is linear with the number of classes.

.. seealso::

    `Multi-class prediction in scikit-learn's documentation
    <http://scikit-learn.org/stable/modules/multiclass.html>`_


**Confusion matrix** `The confusion matrix
<http://en.wikipedia.org/wiki/Confusion_matrix>`_,
:func:`sklearn.metrics.confusion_matrix` is a useful tool to 
understand the classifier's errors in a multiclass problem.

.. figure:: ../auto_examples/decoding/images/plot_haxby_multiclass_1.png
   :target: ../auto_examples/decoding/plot_haxby_multiclass.html
   :align: left
   :scale: 60

.. figure:: ../auto_examples/decoding/images/plot_haxby_multiclass_2.png
   :target: ../auto_examples/decoding/plot_haxby_multiclass.html
   :align: left
   :scale: 40

.. figure:: ../auto_examples/decoding/images/plot_haxby_multiclass_3.png
   :target: ../auto_examples/decoding/plot_haxby_multiclass.html
   :align: left
   :scale: 40

Setting estimator parameters
=============================

Most estimators have parameters that can be set to optimize their
performance. Importantly, this must be done via **nested**
cross-validation.

Indeed, there is noise in the cross-validation score, and when we vary
the parameter, the curve showing the score as a function of the parameter
will have bumps and peaks due to this noise. These will not generalize to
new data and chances are that the corresponding choice of parameter will
not perform as well on new data.

.. figure:: ../auto_examples/decoding/images/plot_haxby_grid_search_1.png
   :target: ../auto_examples/decoding/plot_haxby_grid_search.html
   :align: center
   :scale: 60

With scikit-learn nested cross-validation is done via
:class:`sklearn.grid_search.GridSearchCV`. It is unfortunately time
consuming, but the ``n_jobs`` argument can spread the load on multiple
CPUs.


.. seealso::

   * `The scikit-learn documentation on parameter selection
     <http://scikit-learn.org/stable/modules/grid_search.html>`_

   * The example :ref:`example_decoding_plot_haxby_grid_search.py`

Different linear models
========================

There is a wide variety of classifiers available in scikit-learn (see the
`scikit-learn documentation on supervised learning
<http://scikit-learn.org/stable/supervised_learning.html>`_).
Here we apply a few linear models to fMRI data:

* SVC: the support vector classifier
* SVC cv: the support vector classifier with its parameter C set by
  cross-validation
* log l2: the logistic regression with l2 penalty
* log l2 cv: the logistic regression with l2 penalty with its parameter
  set by cross-validation
* log l1: the logistic regression with l1 penalty: **sparse model**
* log l1 50: the logistic regression with l1 penalty and a high sparsity
  parameter
* log l1 cv: the logistic regression with l1 penalty with its parameter
  (controlling the sparsity) set by cross-validation
* ridge: the ridge classifier
* ridge cv: the ridge classifier with its parameter set by
  cross-validation

.. note::

   * The SVC is fairly insensitive to the choice of the regularization
     parameter
   * cross-validation (CV) takes time
   * The ridge and ridge CV are fast, but will not work well on
     ill-separated classes, and, most importantly give ugly weight maps
     (see below)
   * Parameter selection is difficult with sparse models
   * **There is no free lunch**: no estimator will work uniformely better
     in every situation.


.. figure:: ../auto_examples/decoding/images/plot_haxby_different_estimators_1.png
   :target: ../auto_examples/decoding/plot_haxby_different_estimators.html
   :align: center
   :scale: 80


Note that what is done to the data before applying the estimator is
often more important than the choice of estimator. Typically,
standardizing the data is important, smoothing can often be useful,
and confounding effects, such as session effect, must be removed.

____

The corresponding weight maps (below) differ widely from one estimator to
the other, although the prediction scores are fairly similar. In other
terms, a well-performing estimator in terms of prediction error gives us
little guarantee on the brain maps.

.. figure:: ../auto_examples/decoding/images/plot_haxby_different_estimators_7.png
   :target: ../auto_examples/decoding/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/decoding/images/plot_haxby_different_estimators_8.png
   :target: ../auto_examples/decoding/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/decoding/images/plot_haxby_different_estimators_5.png
   :target: ../auto_examples/decoding/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/decoding/images/plot_haxby_different_estimators_6.png
   :target: ../auto_examples/decoding/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/decoding/images/plot_haxby_different_estimators_4.png
   :target: ../auto_examples/decoding/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/decoding/images/plot_haxby_different_estimators_2.png
   :target: ../auto_examples/decoding/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/decoding/images/plot_haxby_different_estimators_3.png
   :target: ../auto_examples/decoding/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/decoding/images/plot_haxby_different_estimators_9.png
   :target: ../auto_examples/decoding/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/decoding/images/plot_haxby_different_estimators_10.png
   :target: ../auto_examples/decoding/plot_haxby_different_estimators.html
   :align: left
   :scale: 70


SpaceNet
========
SpaceNet implements a suite of multi-variate priors which for improved brain decoding. It uses priors like TV (Total Variation) [Michel et al. 2011], TV-L1 [Baldassarre et al. 2012], [Gramfort et al. 2013] (`penalty="tvl1"), and Smooth-Lasso [Hebiri et al. 2011] (known as GraphNet in neuroimaging [Grosenick 2013]) to regularize classification and regression problems in brain imaging. The result are brain maps which are both sparse (i.e regression coefficients are zero everywhere, except at predictive voxels) and structured (blobby). The superiority of TV-L1 over methods without structured priors like the Lasso, SVM, ANOVA, Ridge, etc. for yielding more interpretable maps and improved prediction scores is now well established [Baldassarre et al. 2012], [Gramfort et al. 2013], [Grosenick et al. 2013].

Note that TV-L1 prior leads to a hard optimization problem, and so can be slow to run.
The follow ing table summarizes the parameter(s) used to activate a given prior:

- TV-L1: `penalty="tv-l1"`
- Smooth-Lasso: `penalty="smooth-lasso"` (this is the default prior in SpaceNet)
- TV: `l1_ratio=0`
- Lasso: `l1_ratio=1`

Examples
........

* Mixed gambles

.. figure:: ../auto_examples/decoding/images/plot_poldrack_space_net_1.png
   :align: right
   :scale: 60

.. figure:: ../auto_examples/decoding/images/plot_poldrack_space_net_2.png
   :scale: 60

.. literalinclude:: ../../examples/decoding/plot_poldrack_space_net.py

* Haxby

.. figure:: ../auto_examples/decoding/images/plot_haxby_space_net_1.png
   :align: right
   :scale: 60

.. figure:: ../auto_examples/decoding/images/plot_haxby_space_net_2.png
   :scale: 60

See the script here:
:doc:`plot_haxby_space_net.py <../auto_examples/decoding/plot_haxby_space_net>`
