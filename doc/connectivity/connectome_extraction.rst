.. _connectome_extraction:

============================================================================
Connectome extraction: inverse covariance for direct connections
============================================================================

.. topic:: **Page summary**

   Given a set of time-series (eg as extracted in the previous section)
   A *functional connectome* is a set of connections representing brain
   interactions between regions. Here we show the use of sparse-inverse
   covariance to extract functional connectomes focusing only on direct
   interactions between regions.

.. topic:: **References**

   * Network modeling methods for FMRI :footcite:p:`Smith2011`

   * Learning and comparing functional connectomes across subjects :footcite:p:`Varoquaux2013`

Sparse inverse covariance for functional connectomes
=====================================================

Functional connectivity can be obtained by estimating a covariance
(or correlation) matrix for signals from different brain
regions decomposed, for example on :term:`resting-state` or naturalistic-stimuli datasets.
The same information can be represented as a weighted graph,
:term:`vertices<vertex>` being brain regions, weights on edges being covariances
(gaussian graphical model). However, coefficients in a covariance matrix
reflect direct as well as indirect connections. Covariance matrices form
very dense brain connectomes, and it is rather difficult to extract from
them only the direct connections between two regions.


As shown in :footcite:t:`Smith2011`, :footcite:t:`Varoquaux2010a`,
it is more interesting to use the inverse covariance matrix,
ie the *precision matrix*.
It gives **only direct connections between regions**, as it
contains *partial covariances*, which are covariances between two regions
conditioned on all the others.


To recover well the interaction structure, a **sparse inverse covariance
estimator** is necessary. The GraphicalLasso, implemented in scikit-learn's
estimator :class:`sklearn.covariance.GraphicalLassoCV` is a good, simple
solution. To use it, you need to create an estimator object:

.. code-block:: python

     from sklearn.covariance import GraphicalLassoCV
     estimator = GraphicalLassoCV()

And then you can fit it on the activation time series, for instance
extracted in :ref:`the previous section <functional_connectomes>`:

.. code-block:: python

     estimator.fit(time_series)

The covariance matrix and inverse-covariance matrix (precision matrix)
can be found respectively in the ``covariance_`` and ``precision_`` attribute
of the estimator:

.. code-block:: python

     estimator.covariance_
     estimator.precision_


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

    The parameter controlling the sparsity is set by
    :sklearn:`cross-validation <modules/cross_validation.html>`
    scheme. If you want to specify it manually, use the estimator
    :class:`sklearn.covariance.GraphicalLasso`.

.. topic:: **Full example**

    See the following example for a full file running the analysis:
    :ref:`sphx_glr_auto_examples_03_connectivity_plot_inverse_covariance_connectome.py`

.. topic:: **Exercise: computing sparse inverse covariance**
   :class: green

   Compute and visualize a connectome on the first subject of the brain
   development dataset downloaded with :func:`nilearn.datasets.fetch_development_fmri`

   **Hints:** The example above has the solution

.. topic:: **Reference**

 * The graph lasso :footcite:p:`Friedman2008` is useful to estimate one
   inverse covariance, ie to work on single-subject data or concatenate
   multi-subject data.


Sparse inverse covariance on multiple subjects
================================================

To work at the level of a group of subject, it can be interesting to
estimate multiple connectomes for each, with a similar structure but
differing connection values across subjects.

For this, nilearn provides the
:class:`nilearn.connectome.GroupSparseCovarianceCV`
estimator. Its usage is similar to the GraphicalLassoCV object, but it takes
a list of time series:

.. code-block:: python

     estimator.fit([time_series_1, time_series_2, ...])

And it provides one estimated covariance and inverse-covariance
(precision) matrix per time-series: for the first one:

.. code-block:: python

     estimator.covariances_[0]
     estimator.precisions_[0]


|

.. currentmodule:: nilearn.connectome

One specific case where this may be interesting is for group analysis
across multiple subjects. Indeed, one challenge when doing statistics on
the coefficients of a connectivity matrix is that the number of
coefficients to compare grows quickly with the number of regions, and as
a result correcting for multiple comparisons takes a heavy toll on
statistical power.

In such a situation, you can use the :class:`GroupSparseCovariance` and
set an ``alpha`` value a bit higher than the alpha value selected by
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
   first 5 subjects of the brain development dataset downloaded with
   :func:`nilearn.datasets.fetch_development_fmri`

   **Hint:** The example above works through the solution for the ADHD dataset.
   adhd.


.. topic:: **Reference**

 * The Brain covariance selection: Better individual functional connectivity models
   using population prior :footcite:p:`Varoquaux2010a`

|

Comparing the different approaches on simulated data
====================================================

We simulate several sets of signals, one set representing one subject,
with different precision matrices, but sharing a common sparsity pattern:
10 brain regions, for 20 subjects.

A single-subject estimation can be performed using the
:class:`sklearn.covariance.GraphicalLassoCV` estimator from scikit-learn.

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

 * The Brain covariance selection: Better individual functional connectivity models
   using population prior :footcite:p:`Varoquaux2010a`

Linking total and direct interactions at the group level
========================================================

Individual connectivity patterns reflect both on covariances and inverse covariances, but in different ways.
For multiple subjects, mean covariance (or correlation)
and group sparse inverse covariance provide different insights into the connectivity at the group level.

We can go one step further by coupling the information from total (pairwise)
and direct interactions in a unique group connectome.
This can be done through a geometrical framework allowing to measure interactions
in a common space called **tangent space** `[Varoquaux et al, MICCAI 2010] <https://inria.hal.science/inria-00512417/>`_.

In nilearn, this is implemented in
:class:`nilearn.connectome.ConnectivityMeasure`:

.. code-block:: python

     measure = ConnectivityMeasure(kind='tangent')

The group connectivity is computed using all the subjects timeseries.:


.. code-block:: python

     connectivities = measure.fit([time_series_1, time_series_2, ...])
     group_connectivity = measure.mean_

Deviations from this mean in the tangent space are provided in the connectivities array
and can be used to compare different groups/runs.
In practice, the tangent measure can outperform the correlation
and partial correlation measures, especially for noisy or heterogeneous data.


.. topic:: **Full example**

    See the following example for a full file running the analysis:
    :ref:`sphx_glr_auto_examples_03_connectivity_plot_group_level_connectivity.py`

.. topic:: **Exercise: computing connectivity in tangent space**
   :class: green

   Compute and visualize the tangent group connectome based on the brain
   development
   dataset downloaded with :func:`nilearn.datasets.fetch_development_fmri`

   **Hints:** The example above has the solution

.. topic:: **Reference**

 * Detection of brain functional-connectivity difference in post-stroke patients using group-level covariance modeling} :footcite:p:`Varoquaux2010b`

References
----------

.. footbibliography::
