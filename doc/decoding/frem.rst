.. for doctests to run, we need to define variables that are define in
   the literal includes
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> fmri_masked  = iris.data
    >>> target = iris.target
    >>> session = np.ones_like(target)
    >>> n_samples = len(target)

.. _frem:

================================================================
fREM: fast ensembling of regularized models for robust decoding
================================================================

The fREM decoder
=====================

fREM uses an implicit spatial regularization through fast clustering and
aggregates a high number of estimators trained on various splits of the
training set, thus returning a very robust decoder at a lower computational
cost than other spatially regularized methods.

Its performance compared to usual classifiers was studied on several datasets
in this article: `[Hoyos-Idrobo et al. 2017] <https:https://hal.archives-ouvertes.fr/hal-01615015>`_)

fREM pipeline averages the coefficients of many models, each trained on a
different split of the training data. For each split:

  * aggregate similar voxels together to reduce the number of features (and the
    computational complexity of the decoding problem). ReNA algorithm is used at this
    step, usually to reduce by a 10 factor the number of voxels.

  * optional : apply feature selection, an univariate statistical test on clusters
    to keep only the ones most informative to predict variable of interest and
    further lower the problem complexity.

  * find the best hyper-parameter and memorize the coefficients of this model

Then this ensemble model is used for prediction, usually yielding better and
more stable predictions than a unique model at no extra-cost. Also, the
resulting coefficient maps obtained tend to be more structured.

This pipeline is available through two objects :
* fREMRegressor to predict continuous values (such as age, gain / loss...)
* fREMClassifier to predict categories

As the Decoder and DecoderRegressor, these object can ensemble different type of
models through the parameter 'estimator' : SVM (l2 or l1), Logistic, Ridge

Empirical comparisons
=====================

Decoding performance increase on Haxby dataset
----------------------------------------------

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_frem_001.png

In this example we showcase the use of fREM and the performance increase that
it brings on this problem.

.. topic:: **Code**

    The complete script can be found
    :ref:`here <sphx_glr_auto_examples_02_decoding_plot_haxby_frem.py>`.

Comparison to SpaceNet on mixed gambles study
----------------------------------------------

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_mixed_gambles_space_net_001.png
   :align: right
   :scale: 40

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_mixed_gambles_space_net_002.png
   :align: center
   :scale: 40

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_mixed_gambles_space_net_003.png
   :align: left
   :scale: 40

.. topic:: **Code**

    The complete script can be found
    :ref:`here <sphx_glr_auto_examples_02_decoding_plot_mixed_gambles_space_net.py>`.
