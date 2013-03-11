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
Decoding on simulated data
================================================================================

.. topic:: Objectives

    1. Understand linear estimators (SVM, elastic net, ridge)
    2. Use the scikit-learn's linear models

Simple NeuroImaging-like simulations
=====================================

We simulate data as in
`Michel et al. 2012 <http://dx.doi.org/10.1109/TMI.2011.2113378>`_ :
a linear model with a random design matrix **X**:

.. math::

   \mathbf{y} = \mathbf{X} \mathbf{w} + \mathbf{e}

* **w**: the weights of the linear model correspond to the predictive 
  brain regions. Here, in the simulations, they form a 3D image with 4
  regions in opposite corners. 

.. figure:: auto_examples/images/plot_simulated_data_1.png
    :target: auto_examples/plot_simulated_data.html
    :align: center
    :scale: 90

* **X**: the design matrix corresponds to the observed fMRI data. Here
  we simulate random normal variables and smooth them as in Gaussian
  fields.

* **e** is random normal noise.

We provide a black-box function to create the data in the 
:ref:`example script <example_tutorial_plot_simulated_data.py>`:

.. literalinclude:: ../plot_simulated_data.py
    :start-after: # Create data
    :end-before: # Compute the results and estimated coef maps for different estimators


Running various estimators
===========================

We can now run different estimators and look at their prediction score,
as well as the feature maps that they recover. Namely, we will use 

* A support vector regression (`SVM
  <http://scikit-learn.org/stable/modules/svm.html>`_) 

* An `elastic-net
  <http://scikit-learn.org/stable/modules/linear_model.html>`_

* A *Bayesian* ridge estimator, i.e. a ridge estimator that sets its
  parameter according to a metaprior

* A ridge estimator that set its parameter by cross-validation

We can create a list with all the estimators readily created with the
parameters of our choice:

.. literalinclude:: ../plot_simulated_data.py
    :start-after: # Compute the results and estimated coef maps for different estimators
    :end-before: # Run the estimators

Note that the `RidgeCV` and the `ElasticNetCV` have names ending in `CV`
that stands for `cross-validation`: in the list of possible `alpha`
values that they are given, they choose the best by cross-validation.

As the estimators expose a fairly consistent API, we can all fit them in
a for loop: they all have a `fit` method for fitting the data, a `score`
method to retrieve the prediction score, and because they are all linear
models, a `coef_` attribute that stores the coefficients **w** estimated.

.. note:: All parameters estimated from the data end with an underscore

.. literalinclude:: ../plot_simulated_data.py
    :start-after: # Run the estimators
    :end-before:     print title

.. |estimator1| image:: auto_examples/images/plot_simulated_data_2.png
    :target: auto_examples/plot_simulated_data.html
    :scale: 60

.. |estimator2| image:: auto_examples/images/plot_simulated_data_3.png
    :target: auto_examples/plot_simulated_data.html
    :scale: 60

.. |estimator3| image:: auto_examples/images/plot_simulated_data_4.png
    :target: auto_examples/plot_simulated_data.html
    :scale: 60

.. |estimator4| image:: auto_examples/images/plot_simulated_data_5.png
    :target: auto_examples/plot_simulated_data.html
    :scale: 60

|estimator1| |estimator2| |estimator3| |estimator4|

.. topic:: **Exercise**
   :class: green

   Use recursive feature elimination (RFE) with the SVM::

    >>> from sklearn.feature_selection import RFE

   Read the object's documentation to find out how to use RFE.

   **Performance tip**: increase the `step` parameter, or it will be very
   slow.



