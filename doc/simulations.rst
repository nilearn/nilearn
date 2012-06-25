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

   At the end of this tutorial you will be able to:

    1. Load fMRI volumes in Python.
    2. Perform a state-of-the-art decoding analysis of fMRI data.
    3. Perform even more sophisticated analyzes of fMRI data.

.. role:: input(strong)

Simple NeuroImaging-like simulations
=====================================

.. figure:: auto_examples/images/plot_simulated_data_1.png
    :target: auto_examples/plot_simulated_data.html
    :align: center
    :scale: 90


First step: looking at the data
================================


.. literalinclude:: ../plot_haxby_decoding.py
        :start-after: ### Load Haxby dataset ########################################################
        :end-before: ### Preprocess data ########################################################### 

.. topic:: **Exercise**
   :class: green

   1. Extract the period of activity from the data (i.e. remove the remainder).

.. topic:: Solution

    As 'y == 0' in rest, we want to keep only time points for which 
    `y != 0`::

     >>> X, y, session = X[y!=0], y[y!=0], session[y!=0]


