.. _func_connect:

================================================================
 Gaussian graphical models for functional connectivity analysis
================================================================

Resting-state functional connectivity can be obtained by estimating a
covariance matrix **C** for signals from different brain regions. Each
element of **C** gives the covariance between two brain regions. The
same information can be represented as a weighted graph, vertices
being brain regions, weights on edges being covariances (gaussian
graphical model). In turn, this graph can be interpreted as a
connection graph between brain regions, with weights giving the
strength of the connection. However, coefficients in a covariance
matrix reflects direct as well as indirect connections. As real brain
signals exhibit small-world properties (there always exists a rather
short path between any two regions), covariance matrices tend to be
dense, and it is rather difficult to extract from them only the direct
connections between two regions.

This can be achieved using a precision matrix, which is the inverse of
the covariance matrix. It contains *partial covariances*, which are
covariances between two regions conditioned on all the others. It thus
gives only direct connections between regions. In the case of fMRI
signals, estimating a covariance matrix using the empirical estimator
gives a very noisy result, mainly because the number of coefficients
to estimate is usually greater than the number of samples available.
That leads to an even worse precision matrix. Thus cleverer schemes
are required to get an usable result.

One way to reduce the number of coefficients to estimate is to impose
sparsity of the precision matrix. It is equivalent to limiting the
number of edges in the graph. Finding the sparsity pattern that gives
the maximum likelihood is a hard problem, since there is
asymptotically 2**(p*p) possible sparsity patterns (where p is the
number of brain regions), which is exponential with p. Thus,
sub-optimal algorithms are used. Two are presented here:
`graph lasso
<http://biostatistics.oxfordjournals.org/content/9/3/432.short>`_
and `group-sparse covariance <http://arxiv.org/abs/1207.4255>`_. Both
are based on maximizing the log-likelihood of the precision matrix,
penalized with a non-smooth regularizer. Both are convex functions,
for which efficient maximizing algorithms exist. The graph lasso
processes one covariance matrix at a time, whereas the group-sparse
covariance algorithm deals with several at the same time, imposing a
common sparsity pattern on all precision matrices.

For more details on these algorithms, please see

    Honorio, Jean, and Dimitris Samaras. '`Simultaneous and
    Group-Sparse Multi-Task Learning of Gaussian Graphical Models
    <http://arxiv.org/abs/1207.4255>`_' arXiv:1207.4255 (17 July 2012).

    Ledoit, Olivier, and Michael Wolf. '`A Well-conditioned Estimator
    for Large-dimensional Covariance Matrices
    <http://www.sciencedirect.com/science/article/pii/S0047259X03000964>`_'. Journal of
    Multivariate Analysis 88, no. 2 (february 2004): 365-411.


And for a general overview of functional connectivity estimation, see

    Gael Varoquaux, Alexandre Gramfort, Jean-Baptiste Poline, and
    Bertrand Thirion. '`Brain Covariance Selection: Better Individual
    Functional Connectivity Models Using Population Prior
    <http://arxiv.org/abs/1008.5071>`_'. arXiv:1008.5071 (30 August
    2010).



Synthetic signals
=================

NiLearn provides a function to generate random signals drawn from a
sparse gaussian graphical model. It can simulate several sets of
signals, one set representing one subject, with different precision
matrices, but sharing a common sparsity pattern. Here is how to use it
to generate signals for 10 brain regions, for 20 subjects:

.. literalinclude:: ../plot_connect_comparison.py
   :start-after: # Generate synthetic data
   :end-before: fig = pl.figure(figsize=(10, 7))

`subjects` and `precisions` are lists containing respectively each
subject's signals and the corresponding true precision matrices used
in the generation (ground truth). `topology` is a single array with
only 0 and 1 giving the common sparsity pattern.

Estimation
==========

The actual estimation is performed using a `cross-validation
<http://scikit-learn.org/stable/modules/cross_validation.html>`
scheme. This allows for selecting the regularization parameter value
for which the model generalizes best on unseen data. This is important
to get models that might be expected to be valid at the population
level.

A single-subject estimation can be performed using the Graph Lasso
estimator from the scikit-learn:

.. literalinclude:: ../plot_connect_comparison.py
   :start-after: # Fit one graph lasso per subject
   :end-before:     pl.subplot(n_displayed, 4, 4 * n + 3)

After calling `fit`, the estimated precision matrix can be plotted
using:

.. literalinclude:: ../plot_connect_comparison.py
   :start-after:     pl.subplot(n_displayed, 4, 4 * n + 3)
   :end-before:     if n == 0:

where `plot_matrix` is a convenience function to avoid repeating the
same code. It draws the matrix as an image, taking care of using a
symmetric range, so that zero values are just in the middle of the
colormap (white in that case):

.. literalinclude:: ../plot_connect_comparison.py
   :start-after: import pylab as pl
   :end-before: # Generate synthetic data


It is also possible to fit a graph lasso on data from every subject at
once:

.. literalinclude:: ../plot_connect_comparison.py
   :start-after: # Fit one graph lasso for all subjects at once
   :end-before: pl.subplot(n_displayed, 4, 4)

Running a group-sparse estimation is very similar, the estimator comes
from NiLearn this time:

..
   gsc

.. literalinclude:: ../plot_connect_comparison.py
   :start-after: # Run group-sparse covariance on all subjects
   :end-before: for n in range(n_displayed):


The results are shown on the following figure:

.. |results| image:: auto_examples/images/plot_connect_comparison_1.png
    :target: auto_examples/plot_connect_comparison.html
    :scale: 60

|results|

It is visible that the group-sparse estimation outputs matrices with
the same sparsity pattern, but different values for the non-zero
coefficients. This is not the case for the graph lasso output, which
all have similar but different structures. Note that the graph lasso
applied to all subjects at once gives a sparsity pattern close to that
obtained with the group-sparse one, but cannot provide per-subject
information.

.. topic:: Note

   The complete source code for this example can be found here:
   :doc:`plot_connect_comparison.py <auto_examples/plot_connect_comparison>`

.. seealso::
   For a detailed example on real data:
   :doc:`plot_adhd_covariance.py <auto_examples/plot_adhd_covariance>`

____

A lot of technical details on the algorithm used for group-sparse
estimation and its implementation can be found in :doc:`developers/group_sparse_covariance`.

