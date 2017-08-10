.. _connectome_extraction:

============================================================================
Connectome extraction: inverse covariance for direct connections
============================================================================

.. topic:: **Page summary**

   Given a set of time-series (eg as extracted in the previous section)
   A *functional connectome* is a set of connections representing brain
   interactions between regions. Here we show the use of sparse-inverse
   covariance to extract functional connectomes focussing only on direct
   interactions between regions.

.. contents:: **Contents**
    :local:
    :depth: 1

.. topic:: **References**

   * `Smith et al, Network modelling methods for FMRI,
     NeuroImage 2011 <http://www.sciencedirect.com/science/article/pii/S1053811910011602>`_

   * `Varoquaux and Craddock, Learning and comparing functional
     connectomes across subjects, NeuroImage 2013
     <http://www.sciencedirect.com/science/article/pii/S1053811913003340>`_

Sparse inverse covariance for functional connectomes
=====================================================

Resting-state functional connectivity can be obtained by estimating a
covariance (or correlation) matrix for signals from different brain
regions. The same information can be represented as a weighted graph,
vertices being brain regions, weights on edges being covariances
(gaussian graphical model). However, coefficients in a covariance matrix
reflect direct as well as indirect connections. Covariance matrices form
very dense brain connectomes, and it is rather difficult to extract from
them only the direct connections between two regions.


As shown in `[Smith 2011]
<http://www.sciencedirect.com/science/article/pii/S1053811910011602>`_,
`[Varoquaux 2010] <https://hal.inria.fr/inria-00512451>`_, it is more
interesting to use the inverse covariance matrix, ie the *precision
matrix*. It gives **only direct connections between regions**, as it
contains *partial covariances*, which are covariances between two regions
conditioned on all the others.


To recover well the interaction structure, a **sparse inverse covariance
estimator** is necessary. The GraphLasso, implemented in scikit-learn's
estimator :class:`sklearn.covariance.GraphLassoCV` is a good, simple
solution. To use it, you need to create an estimator object::

    >>> from sklearn.covariance import GraphLassoCV
    >>> estimator = GraphLassoCV()

And then you can fit it on the activation time series, for instance
extracted in :ref:`the previous section <functional_connectomes>`::

    >>> estimator.fit(time_series)  # doctest: +SKIP

The covariance matrix and inverse-covariance matrix (precision matrix)
can be found respectively in the `covariance_` and `precision_` attribute
of the estimator::

    >>> estimator.covariance_  # doctest: +SKIP
    >>> estimator.precision_  # doctest: +SKIP


.. |covariance| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_inverse_covariance_connectome_001.png
    :target: ../auto_examples/03_connectivity/plot_inverse_covariance_connectome.html
    :scale: 40
.. |precision| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_inverse_covariance_connectome_003.png
    :target: ../auto_examples/03_connectivity/plot_inverse_covariance_connectome.html
    :scale: 40

.. |covariance_graph| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_inverse_covariance_connectome_002.png
    :target: ../auto_examples/03_connectivity/plot_inverse_covariance_connectome.html
    :scale: 55

.. |precision_graph| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_inverse_covariance_connectome_004.png
    :target: ../auto_examples/03_connectivity/plot_inverse_covariance_connectome.html
    :scale: 55

.. centered:: |covariance| |precision|

.. centered:: |covariance_graph| |precision_graph|



.. topic:: **Parameter selection**

    The parameter controlling the sparsity is set by `cross-validation
    <http://scikit-learn.org/stable/modules/cross_validation.html>`_
    scheme. If you want to specify it manually, use the estimator
    :class:`sklearn.covariance.GraphLasso`.

.. topic:: **Full example**

    See the following example for a full file running the analysis:
    :ref:`sphx_glr_auto_examples_03_connectivity_plot_inverse_covariance_connectome.py`

.. topic:: **Exercise: computing sparse inverse covariance**
   :class: green

   Compute and visualize a connectome on the first subject of the ADHD
   dataset downloaded with :func:`nilearn.datasets.fetch_adhd`

   **Hints:** The example above has the solution

.. topic:: **Reference**

 * The `graph lasso [Friedman et al, Biostatistics 2007] <http://biostatistics.oxfordjournals.org/content/9/3/432.short>`_ is useful to estimate one
   inverse covariance, ie to work on single-subject data or concatenate
   multi-subject data.


Sparse inverse covariance on multiple subjects
================================================

To work at the level of a group of subject, it can be interesting to
estimate multiple connectomes for each, with a similar structure but
differing connection values across subjects.

For this, nilearn provides the
:class:`nilearn.connectome.GroupSparseCovarianceCV`
estimator. Its usage is similar to the GraphLassoCV object, but it takes
a list of time series::

    >>> estimator.fit([time_series_1, time_series_2, ...])  # doctest: +SKIP

And it provides one estimated covariance and inverse-covariance
(precision) matrix per time-series: for the first one::

    >>> estimator.covariances_[0]  # doctest: +SKIP
    >>> estimator.precisions_[0]  # doctest: +SKIP

|

.. currentmodule:: nilearn.connectome

One specific case where this may be interesting is for group analysis
across multiple subjects. Indeed, one challenge when doing statistics on
the coefficients of a connectivity matrix is that the number of
coefficients to compare grows quickly with the number of regions, and as
a result correcting for multiple comparisons takes a heavy toll on
statistical power.

In such a situation, you can use the :class:`GroupSparseCovariance` and
set an `alpha` value a bit higher than the alpha value selected by
cross-validation in the :class:`GroupSparseCovarianceCV`. Such a choice
will enforce a stronger sparsity on the precision matrices for each
subject. As the sparsity is common to each subject, you can then do the
group analysis only on the non zero coefficients.

.. topic:: **Full example**

    See the following example for a full file running the analysis:
    :ref:`sphx_glr_auto_examples_03_connectivity_plot_multi_subject_connectome.py`


.. topic:: **Exercise: computing the correlation matrix of rest fmri**
   :class: green

   Try using the information above to compute a connectome on the
   first 5 subjects of the ADHD dataset downloaded with
   :func:`nilearn.datasets.fetch_adhd`

   **Hint:** The example above has the solution


.. topic:: **Reference**

 * The `group-sparse covariance [Varoquaux et al, NIPS 2010] <https://hal.inria.fr/inria-00512451>`_

|

Comparing the different approaches on simulated data
====================================================

We simulate several sets of signals, one set representing one subject,
with different precision matrices, but sharing a common sparsity pattern:
10 brain regions, for 20 subjects.

A single-subject estimation can be performed using the
:class:`sklearn.covariance.GraphLassoCV` estimator from scikit-learn.

It is also possible to fit a graph lasso on data from every subject all
together.

Finally, we use the
:class:`nilearn.connectome.GroupSparseCovarianceCV` [#]_.



The results are the following:

.. image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_simulated_connectome_001.png
    :target: ../auto_examples/03_connectivity/plot_simulated_connectome.html
    :scale: 60

The group-sparse estimation outputs matrices with
the same sparsity pattern, but different values for the non-zero
coefficients. This is not the case for the graph lasso output, which
all have similar but different structures. Note that the graph lasso
applied to all subjects at once gives a sparsity pattern close to that
obtained with the group-sparse one, but cannot provide per-subject
information.

.. topic::  **Full Example**

   The complete source code for this example can be found here:
   :ref:`sphx_glr_auto_examples_03_connectivity_plot_simulated_connectome.py`


.. [#] A lot of technical details on the algorithm used for group-sparse
       estimation and its implementation can be found in
       :doc:`../developers/group_sparse_covariance`.

.. toctree::
   :hidden:

   ../developers/group_sparse_covariance

.. topic:: **Reference**

 * The `Brain covariance selection using population prior [Varoquaux et al, NIPS 2010] <http://papers.nips.cc/paper/4080-brain-covariance-selection-better-individual-functional-connectivity-models-using-population-prior>`_

Linking total and direct interactions at the group level
========================================================

Individual connectivity patterns reflect both on covariances and inverse covariances, but in different ways. For multiple subjects, mean covariance (or correlation) and group sparse inverse covariance provide different insights into the connectivity at the group level.

We can go one step further by coupling the information from total (pairwise) and direct interactions in a unique group connectome. This can be done through a geometrical framework allowing to measure interactions in a common space called **tangent space** `[Varoquaux et al, MICCAI 2010] <https://hal.inria.fr/inria-00512417/>`_.

In nilearn, this is implemented in
:class:`nilearn.connectome.ConnectivityMeasure`::

    >>> measure = ConnectivityMeasure(kind='tangent')  # doctest: +SKIP

The group connectivity is computed using all the subjects timeseries.::

    >>> connectivities = measure.fit([time_series_1, time_series_2, ...])  # doctest: +SKIP
    >>> group_connectivity = measure.mean_  # doctest: +SKIP

Deviations from this mean in the tangent space are provided in the connectivities array and can be used to compare different groups/sessions. In practice, the tangent measure can outperform the correlation and partial correlation measures, especially for noisy or heterogeneous data.


.. topic:: **Full example**

    See the following example for a full file running the analysis:
    :ref:`sphx_glr_auto_examples_03_connectivity_plot_group_level_connectivity.py`

.. topic:: **Exercise: computing connectivity in tangent space**
   :class: green

   Compute and visualize the tangent group connectome based on the NYU, OHSU and NeuroImage sites of the ADHD
   dataset downloaded with :func:`nilearn.datasets.fetch_adhd`

   **Hints:** The example above has the solution

.. topic:: **Reference**

 * The `tangent space for connectivity [Varoquaux et al, MICCAI 2010] <http://link.springer.com/chapter/10.1007%2F978-3-642-15705-9_25>`_
