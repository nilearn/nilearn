
.. _estimator_choice:

=====================================================
Choosing the right predictive model for neuroimaging
=====================================================

This page gives a few simple considerations on the choice of an estimator to
tackle your *decoding* application, that is the prediction of external
variables such as behavior or clinical traits from brain images. It is
focusing on practical concepts to understand which prediction pipeline
is well suited to your problem and how to implement it easily with Nilearn.
This builds on concepts introduced in this :ref:`didactic
introduction to decoding with nilearn <decoding_intro>`.


Predictions: regression, classification and multi-class
=======================================================

As seen in the previous section, high-level objects in Nilearn help you decode
easily your dataset using a **mask** and/or **feature selection**. You can tune
the **cross-validation** and **scoring** schemes of your model. Those objects
come in two kinds, depending on your usecase : :term:`Regression<regression>` or :term:`Classification<classification>`.

Regression
----------

A :term:`regression` problem is a learning task in which the variable to predict
--that we often call **y** -- is a continuous value, such as an age.
Encoding models (:footcite:t:`Naselaris2011`) typically call for regressions.
:class:`nilearn.decoding.DecoderRegressor` implement easy and efficient
regression pipelines.

.. seealso::

   * :class:`nilearn.decoding.FREMRegressor`, a pipeline described in the
     :ref:`userguide <frem>`, which yields very good regression performance for
     neuroimaging at a reasonable computational cost.

Classification: two classes or multi-class
------------------------------------------

A :term:`classification` task consists in predicting a *class* label for each
observation. In other words, the variable to predict is categorical.

Often :term:`classification` is performed between two classes, but it may well be
applied to multiple classes, in which case it is known as a multi-class
problem. It is important to keep in mind that the larger the number of
classes, the harder the prediction problem.

:class:`nilearn.decoding.Decoder` implement easy and efficient
:term:`classification` pipelines.

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
whereas the latter is linear with the number of classes.

.. seealso::

  * `Multi-class prediction in scikit-learn's documentation <https://scikit-learn.org/stable/modules/multiclass.html>`_
  * :class:`nilearn.decoding.FREMClassifier`, a pipeline described in the
    :ref:`userguide <frem>`, yielding state-of-the art decoding performance.

**Confusion matrix** `The confusion matrix
<https://en.wikipedia.org/wiki/Confusion_matrix>`_,
:func:`sklearn.metrics.confusion_matrix` is a useful tool to
understand the classifier's errors in a multiclass problem.

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_multiclass_001.png
   :target: ../auto_examples/02_decoding/plot_haxby_multiclass.html
   :align: center
   :scale: 60

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_multiclass_002.png
   :target: ../auto_examples/02_decoding/plot_haxby_multiclass.html
   :align: center
   :scale: 40

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_multiclass_003.png
   :target: ../auto_examples/02_decoding/plot_haxby_multiclass.html
   :align: center
   :scale: 40


Different linear models
=======================

Using Nilearn high-level objects, several estimators are easily available
to model the relations between your images and the target to predict.
For :term:`classification`, :class:`nilearn.decoding.Decoder` let you choose them
through the ``estimator`` parameter:

* ``svc`` (same as ``svc_l2``) : The `support vector classifier <https://scikit-learn.org/stable/modules/svm.html>`_.

* ``svc_l1`` : SVC using `L1 penalization <https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity>`_ that yields a sparse solution : only a subset of feature weights is different from zero and contribute to prediction.

* ``logistic`` (or ``logistic_l2``) : The `logistic regression <https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression>`_ with `l2 penalty <https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html>`_.

* ``logistic_l1`` :  The `logistic regression <https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression>`_ with `l1 penalty <https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html>`_ (**sparse model**).

* ``ridge_classifier`` : A `Ridge Regression variant
  <https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification>`_.

* ``dummy classifier`` : A `dummy classifier <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html>`_ is a classifier that makes predictions using simple rules. It is useful as a simple baseline to compare with other classifiers.

In :class:`nilearn.decoding.DecoderRegressor` you can use some of these objects counterparts for regression :

* ``svr`` : `Support vector regression <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_.

* ``ridge_regressor`` (same as ``ridge``) : `Ridge regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html>`_.

* ``lasso_regressor`` (same as ``lasso``) : `Lasso regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html>`_.

* ``dummy_regressor`` : A `dummy regressor <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html>`_ is a regressor that makes predictions using simple rules. It is useful as a simple baseline to compare with other regressors.

.. note::

   * **There is no free lunch**: no estimator will work uniformly better
     in every situation.

   * The SVC-l2 is fairly insensitive to the choice of the regularization
     parameter which makes it a good and cheap first approach to most problems

   * The ridge is fast to fit and cross-validate, but it will not work well on
     ill-separated classes, and, most importantly give ugly weight maps

   * Whenever a model uses sparsity (have l1 in its name here) the parameter
     selection (amount of sparsity used) can change result a lot and is difficult
     to tune well.

   * What is done to the data  **before** applying the estimator is
     often  **more important** than the choice of estimator. Typically,
     standardizing the data is important, smoothing can often be useful,
     and nuisance effects, such as run effect, must be removed.

   * Many more estimators are available in scikit-learn (see the
     `scikit-learn documentation on supervised learning
     <https://scikit-learn.org/stable/supervised_learning.html>`_). To learn to
     do decoding with any of these, see : :ref:`going_further`

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_different_estimators_001.png
   :target: ../auto_examples/02_decoding/plot_haxby_different_estimators.html
   :align: center
   :scale: 80

____

The corresponding weight maps (below) differ widely from one estimator to
the other, although the prediction scores are fairly similar. In other
terms, a well-performing estimator in terms of prediction error gives us
little guarantee on the brain maps.

.. image:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_different_estimators_006.png
   :target: ../auto_examples/02_decoding/plot_haxby_different_estimators.html
   :scale: 70
.. image:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_different_estimators_005.png
   :target: ../auto_examples/02_decoding/plot_haxby_different_estimators.html
   :scale: 70
.. image:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_different_estimators_004.png
   :target: ../auto_examples/02_decoding/plot_haxby_different_estimators.html
   :scale: 70
.. image:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_different_estimators_002.png
   :target: ../auto_examples/02_decoding/plot_haxby_different_estimators.html
   :scale: 70
.. image:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_different_estimators_003.png
   :target: ../auto_examples/02_decoding/plot_haxby_different_estimators.html
   :scale: 70

Setting estimator parameters
============================

Most estimators have parameters (called "hyper-parameters") that can be set
to optimize their performance to a given problem. By default, the Decoder
objects in Nilearn already try several values to roughly adapt to your problem.

If you want to try more specific sets of parameters relevant to the model
your using, you can pass a dictionary to ``param_grid`` argument. It must contain
values for the suitable argument name. For example SVC has a parameter ``C``.
By default, the values tried for ``C`` are [1,10,100].

.. note::
  Full code example on parameter setting can be found at :
  :ref:`sphx_glr_auto_examples_02_decoding_plot_haxby_grid_search.py`

Be careful about **overfitting**. Giving a grid containing too many parameter
close to each other will be computationnaly costly to fit and may result in
choosing a parameter that works best on your training set, but does not give
as good performances on your data. You can see below an example in which the
curve showing the score as a function of the parameter has bumps and peaks
due to this noise.

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_grid_search_001.png
   :target: ../auto_examples/02_decoding/plot_haxby_grid_search.html
   :align: center
   :scale: 60

.. seealso::

   `The scikit-learn documentation on parameter selection
   <https://scikit-learn.org/stable/modules/grid_search.html>`_

Bagging several models
============================

`Bagging <https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator>`_
is a classical machine learning method to create ensemble of models that usually
generalize to new data better than single model. The easiest way is to average
the prediction of several models trained on slightly different part of a
dataset and thus should have different bias that may cancel out.

The :class:`nilearn.decoding.Decoder` and :class:`nilearn.decoding.DecoderRegressor`
implement a kind of bagging scheme under the hood in their ``fit`` method to
yield better and more stable decoders. For each cross-validation fold,
the best model coefficients are retained. The average of all those linear
models is then used to make predictions.

.. seealso::

  * The `scikit-learn documentation <https://scikit-learn.org>`_
    has very detailed explanations on a large variety of estimators and
    machine learning techniques. To become better at decoding, you need
    to study it.

  * :ref:`FREM <frem>`, a pipeline bagging many models that yields very
    good decoding performance at a reasonable computational cost.

  * :ref:`SpaceNet <space_net>`, a method promoting sparsity that can also
    give good brain decoding power and improved decoder maps when sparsity
    is important.

References
==========

.. footbibliography::
