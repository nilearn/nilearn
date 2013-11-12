.. _functional_connectomes:

================================================================
Learning functional connectomes
================================================================

.. topic:: **Page summary**

   A *functional connectome* is a set of connections representing brain
   interactions between regions. Here we show the use of sparse-inverse
   covariance estimators to extract functional connectomes.

.. topic:: **References**

   * `Smith et al, Network modelling methods for FMRI,
     NeuroImage 2011 <http://www.sciencedirect.com/science/article/pii/S1053811910011602>`_

   * `Varoquaux and Craddock, Learning and comparing functional
     connectomes across subjects, NeuroImage 2013
     <http://www.sciencedirect.com/science/article/pii/S1053811913003340>`_ 

Sparse inverse covariance for functional connectomes
=====================================================

Resting-state functional connectivity can be obtained by estimating a
covariance matrix **C** for signals from different brain regions. Each
element of **C** gives the covariance between two brain regions. The same
information can be represented as a weighted graph, vertices being brain
regions, weights on edges being covariances (gaussian graphical model).
In turn, this graph can be interpreted as a connection graph between
brain regions, with weights giving the strength of the connection.
However, coefficients in a covariance matrix reflects direct as well as
indirect connections. Covariance matrices tend to be dense, and it is
rather difficult to extract from them only the direct connections between
two regions.

This can be achieved using the inverse of the covariance matrix, ie the
*precision matrix*. It contains *partial covariances*, which are
covariances between two regions conditioned on all the others. It thus
gives only direct connections between regions.

.. |covariance| image:: ../auto_examples/images/plot_adhd_covariance_5.png
   :target: ../auto_examples/plot_adhd_covariance.html
   :scale: 39

.. |precision| image:: ../auto_examples/images/plot_adhd_covariance_6.png
   :target: ../auto_examples/plot_adhd_covariance.html
   :scale: 39

.. centered:: |covariance| |precision|

Sparsity in the inverse covariance matrix is important to reduce noise in
the estimated connectome by limiting the number of edges (technicaly,
this problem arises from multi-colinearity in time series, when the
number of time points is not very large compared to the number of
regions). Here we explore 2 different options to estimate sparse inverse
covariance estimates:

* The `graph lasso [Friedman et al, Biostatistics 2007] <http://biostatistics.oxfordjournals.org/content/9/3/432.short>`_ is useful to estimate one
  inverse covariance, ie to work on single-subject data or concatenate
  multi-subject data.

* The `group-sparse covariance [Varoquaux et al, NIPS 2010] <http://hal.inria.fr/inria-00512451>`_ estimates multiple connectomes from a multi-subject dataset, 
  with a similar structure, but differing connection values across
  subjects.


Testing the different approaches on simulated data
===================================================

Synthetic signals
-----------------

We simulate several sets of signals, one set representing one subject,
with different precision matrices, but sharing a common sparsity pattern:
10 brain regions, for 20 subjects:

.. literalinclude:: ../../plot_connect_comparison.py
   :start-after: # Generate synthetic data
   :end-before: fig = pl.figure(figsize=(10, 7))

`subjects` and `precisions` are lists containing respectively each
subject's signals and the corresponding true precision matrices used
in the generation (ground truth). `topology` is a single array with
only 0 and 1 giving the common sparsity pattern.

Estimation
----------

The actual estimation is performed using a `cross-validation
<http://scikit-learn.org/stable/modules/cross_validation.html>`_
scheme. This allows for selecting the regularization parameter value
for which the model generalizes best on unseen data. This is important
to get models that might be expected to be valid at the population
level.

A single-subject estimation can be performed using the Graph Lasso
estimator from the scikit-learn:

.. literalinclude:: ../../plot_connect_comparison.py
   :start-after: # Fit one graph lasso per subject
   :end-before:     pl.subplot(n_displayed, 4, 4 * n + 3)

After calling `fit`, the estimated precision matrix can be plotted
using:

.. literalinclude:: ../../plot_connect_comparison.py
   :start-after:     pl.subplot(n_displayed, 4, 4 * n + 3)
   :end-before:     if n == 0:

where `plot_matrix` is a convenience function to avoid repeating the
same code. It draws the matrix as an image, taking care of using a
symmetric range, so that zero values are just in the middle of the
colormap (white in that case):

.. literalinclude:: ../../plot_connect_comparison.py
   :start-after: import pylab as pl
   :end-before: # Generate synthetic data


It is also possible to fit a graph lasso on data from every subject at
once:

.. literalinclude:: ../../plot_connect_comparison.py
   :start-after: # Fit one graph lasso for all subjects at once
   :end-before: pl.subplot(n_displayed, 4, 4)

Running a group-sparse estimation is very similar, the estimator comes
from NiLearn this time:

..
   gsc

.. literalinclude:: ../../plot_connect_comparison.py
   :start-after: # Run group-sparse covariance on all subjects
   :end-before: for n in range(n_displayed):


The results are shown on the following figure:

.. image:: ../auto_examples/images/plot_connect_comparison_1.png
    :target: auto_examples/plot_connect_comparison.html
    :scale: 60

The group-sparse estimation outputs matrices with
the same sparsity pattern, but different values for the non-zero
coefficients. This is not the case for the graph lasso output, which
all have similar but different structures. Note that the graph lasso
applied to all subjects at once gives a sparsity pattern close to that
obtained with the group-sparse one, but cannot provide per-subject
information.

.. note::

   The complete source code for this example can be found here:
   :doc:`plot_connect_comparison.py <../auto_examples/plot_connect_comparison>`

A real-data example
====================

For a detailed example on real data:
:doc:`plot_adhd_covariance.py <../auto_examples/plot_adhd_covariance>`

____

A lot of technical details on the algorithm used for group-sparse
estimation and its implementation can be found in :doc:`../developers/group_sparse_covariance`.


