.. _frem:

================================================================
FREM: fast ensembling of regularized models for robust decoding
================================================================

:term:`FREM` uses an implicit spatial regularization through fast clustering and aggregates a high number of estimators trained on various splits of the training set, thus returning a very robust decoder at a lower computational cost than other spatially regularized methods. Its performance compared to usual classifiers was studied on several datasets in [:footcite:t:`Hoyos-Idrobo2018`].

FREM pipeline
=============

:term:`FREM` pipeline averages the coefficients of many models, each trained on a
different split of the training data. For each split:

  * aggregate similar :term:`voxels<voxel>` together to reduce the number of features (and the
    computational complexity of the decoding problem). :term:`ReNA` algorithm is used at this
    step, usually to reduce by a 10 factor the number of :term:`voxels<voxel>`.

  * optional : apply feature selection, an univariate statistical test on clusters
    to keep only the ones most informative to predict variable of interest and
    further lower the problem complexity.

  * find the best hyper-parameter and memorize the coefficients of this model

Then this ensemble model is used for prediction, usually yielding better and more stable predictions than a unique model at no extra-cost. Also, the resulting coefficient maps obtained tend to be more structured.

There are two object to apply :term:`FREM` in Nilearn:

  * :class:`nilearn.decoding.FREMClassifier` to predict categories

  * :class:`nilearn.decoding.FREMRegressor` to predict continuous values (age, gain / loss...)

They can use different type of models (l2-SVM, l1-SVM, Logistic, Ridge) through the parameter 'estimator'.


Empirical comparisons
=====================

Decoding performance increase on Haxby dataset
----------------------------------------------

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_frem_001.png

In this example we showcase the use of :term:`FREM` and the performance increase that
it brings on this problem.

.. topic:: **Code**

    The complete script can be found
    :ref:`here <sphx_glr_auto_examples_02_decoding_plot_haxby_frem.py>`.

Spatial regularization of decoding maps on mixed gambles study
---------------------------------------------------------------

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_mixed_gambles_frem_001.png


.. topic:: **Code**

    The complete script can be found
    :ref:`here <sphx_glr_auto_examples_02_decoding_plot_mixed_gambles_frem.py>`.

.. seealso::

    * The `scikit-learn documentation <http://scikit-learn.org>`_
      has very detailed explanations on a large variety of estimators and
      machine learning techniques. To become better at decoding, you need
      to study it.

    * :ref:`SpaceNet <space_net>`, a method promoting sparsity that can also
      give good brain decoding power and improved decoder maps when sparsity
      is important.

References
==========

.. footbibliography::
