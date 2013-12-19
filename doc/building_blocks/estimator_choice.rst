
.. _estimator_choice:

============================================
Considerations on the choice of an estimator
============================================

This page gives a few simple consideration on the choice of an estimator.
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
Encoding models [1] typically call for regressions.

.. _[1]:

   Naselaris et al, Encoding and decoding in fMRI, NeuroImage Encoding
   and decoding in fMRI.2011 http://www.ncbi.nlm.nih.gov/pubmed/20691790

Classification: two classes or multi-class
-------------------------------------------

A classification task consists in predicting a *class* label for each
observation. In other words, the variable to predict is categorical.

Often classification is performed between two classes, but is may well be 
applied to multiple classes, in which case it is known as a multi-class
problem. It is important to keep in mind that the large the number of
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

.. seealso::
   
    `Multi-class prediction in scikit-learn's documenation
    <http://scikit-learn.org/stable/modules/multiclass.html>`_


**Confusion matrix** `The confusion matrix
<http://en.wikipedia.org/wiki/Confusion_matrix>`_,
:func:`sklearn.metrics.confusion_matrix` is a useful tool to 
understand the classifier's errors in a multiclass problem.

.. figure:: ../auto_examples/images/plot_haxby_multiclass_1.png
   :target: ../auto_examples/plot_haxby_multiclass.html
   :align: left
   :scale: 60

.. figure:: ../auto_examples/images/plot_haxby_multiclass_2.png
   :target: ../auto_examples/plot_haxby_multiclass.html
   :align: left
   :scale: 40

.. figure:: ../auto_examples/images/plot_haxby_multiclass_3.png
   :target: ../auto_examples/plot_haxby_multiclass.html
   :align: left
   :scale: 40

Setting estimator parameters
=============================



Different linear models
========================

.. figure:: ../auto_examples/images/plot_haxby_different_estimators_1.png
   :target: ../auto_examples/plot_haxby_different_estimators.html
   :align: center
   :scale: 80

____

.. figure:: ../auto_examples/images/plot_haxby_different_estimators_7.png
   :target: ../auto_examples/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/images/plot_haxby_different_estimators_8.png
   :target: ../auto_examples/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/images/plot_haxby_different_estimators_5.png
   :target: ../auto_examples/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/images/plot_haxby_different_estimators_6.png
   :target: ../auto_examples/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/images/plot_haxby_different_estimators_4.png
   :target: ../auto_examples/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/images/plot_haxby_different_estimators_2.png
   :target: ../auto_examples/plot_haxby_different_estimators.html
   :align: left
   :scale: 70

.. figure:: ../auto_examples/images/plot_haxby_different_estimators_3.png
   :target: ../auto_examples/plot_haxby_different_estimators.html
   :align: left
   :scale: 70


