.. _searchlight:

===========================================================
Searchlight : finding voxels containing information
===========================================================

.. currentmodule:: nilearn.decoding

Principle of the Searchlight
============================

Searchlight was introduced in `Information-based functional brain mapping
<http://www.pnas.org/content/103/10/3863>`_, Nikolaus Kriegeskorte,
Rainer Goebel and Peter Bandettini (PNAS 2006) and consists in scanning the
images volume with a *searchlight*. Briefly, a ball of given radius is
scanned across the brain volume and the prediction accuracy of a
classifier trained on the corresponding voxels is measured.

Preparing the data
====================

Loading
-------

Fetching the data from internet and loading it can be done with the
provided functions (see :ref:`loading_data`):

.. literalinclude:: ../../plot_haxby_searchlight.py
    :start-after: ### Load Haxby dataset ########################################################
    :end-before: ### Restrict to faces and houses ##############################################

Reshaping the data
-------------------

For this example we need:

- to put X in the form *n_samples* x *n_features*
- compute a mean image for visualization background
- limit our analysis to the `face` and `house` conditions
  (like in the :ref:`decoding <fmri_decoding>` example)

.. literalinclude:: ../../plot_haxby_searchlight.py
    :start-after: ### Restrict to faces and houses ##############################################
    :end-before: ### Prepare masks #############################################################

Masking
-------

One of the main elements that distinguish Searchlight from other algorithms is
the notion of structuring element that scans the entire volume. If this seems
rather intuitive, it has in fact an impact on the masking procedure.

Most of the time, fMRI data is masked and then given to the algorithm. This is
not possible in the case of Searchlight because, to compute the score of
non-masked voxels, some masked voxels may be needed. This is why two masks will
be used here :

- *mask* is the anatomical mask
- *process_mask* is a subset of mask and contains voxels to be processed.

*process_mask* will then be used to restrain computation to one slice, in the
back of the brain. *mask* will ensure that no value outside the brain is
taken into account when iterating with the sphere.

.. literalinclude:: ../../plot_haxby_searchlight.py
        :start-after: #   brain to speed up computation)
        :end-before: ### Searchlight computation ###################################################

Third Step: Setting up the searchlight
=======================================

Classifier
----------

The classifier used by default by :class:`SearchLight` is LinearSVC with C=1 but
this can be customed easily by passing an estimator parameter to the
cross validation. See scikit-learn documentation for `other classifiers
<http://scikit-learn.org/stable/supervised_learning.html>`_.

Score function
--------------

Here we use precision as metrics to measure the proportion of true
positives among all positive results for one class. Others metrics can be
specified by the "scoring" argument to the :class:`SearchLight`, as
detailed in the `scikit-learn documentation
<http://scikit-learn.org/dev/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules>`_

Cross validation
----------------

:class:`SearchLight` will iterate on the volume and give a score to each voxel. This
score is computed by running a classifier on selected voxels. In order to make
this score as accurate as possible (and avoid overfitting), a cross validation
is made.

As :class:`SearchLight` is computationally costly, we have chosen a cross validation
method that does not take too much time. *K*-Fold along with *K* = 4 is a
good compromise between running time and quality.

.. literalinclude:: ../plot_haxby_searchlight.py
    :start-after: # set once and the others as learning sets
    :end-before: import nilearn.decoding

Running Searchlight
===================

Running :class:`SearchLight` is straightforward now that everything is set. The only
parameter left is the radius of the ball that will run through the data.
Kriegskorte et al. use a 4mm radius because it yielded the best detection
performance in their simulation.

.. literalinclude:: ../../plot_haxby_searchlight.py
    :start-after: # The radius is the one of the Searchlight sphere that will scan the volume
    :end-before: ### F-scores computation ######################################################
	
Visualization
=============

Searchlight
-----------

As the activation map is cropped, we use the mean image of all scans as a
background. We can see here that voxels in the visual cortex contains
information to distinguish pictures showed to the volunteers, which was the
expected result.

.. figure:: ../auto_examples/images/plot_haxby_searchlight_1.png
   :target: ../auto_examples/plot_haxby_searchlight.html
   :align: center
   :scale: 60

.. literalinclude:: ../../plot_haxby_searchlight.py
    :start-after: ### Visualization #############################################################
    :end-before: ### Show the F_score


.. seealso::

   * :ref:`visualizing`

Comparing to massively univariate analysis: F_score or SPM
----------------------------------------------------------

The standard approach to brain mapping is performed using *Statistical
Parametric Mapping* (SPM), using ANOVA (analysis of variance), and
F-tests. Here we compute the *p-values* of the voxels [1]_.
To display the results, we use the negative log of the p-value.

.. figure:: ../auto_examples/images/plot_haxby_searchlight_2.png
   :target: ../auto_examples/plot_haxby_searchlight.html
   :align: center
   :scale: 60

.. literalinclude:: ../../plot_haxby_searchlight.py
    :start-after: ### F_score results

F-scores can be converted into p-values using a reference theoretical
distribution, which is known under specific assumptions. In practice,
neuroimaging signal has a complex structure that might not match these
assumptions. An exact, non-parametric *permutation test* can be
performed as an alternative to the analytic F-test: the residuals of
the model are permuted so as to break any effect and the corresponding
decision statistic is recomputed. One thus builds the distribution of
the decision statistic under the hypothesis that there is no
relationship between the tested variates and the target variates. In neuroimaging, this is generally done by swapping the
signal values of all voxels while the tested variables remain
unchanged [2]_. A voxel-wise analysis is then performed on the permuted
data. The relationships
between the image descriptors and the tested variates are broken while
the value of the signal in each particular voxel can be observed with
the same probability than the original value associated to that
voxel. Note that it is hereby assumed that the signal distribution is
the same in every voxel. Several data permutations are performed
(typically 10,000) while the F-scores for every voxel and every data
permutation is stored. The empirical distribution of the F-scores is
thus constructed (under the hypothesis that there is no relationship
between the tested variates and the neuroimaging signal, the so-called
*null-hypothesis*) and we can compare the original F-scores to that
distribution: The higher the rank of the original F-score, the smaller is its associated p-value. The
:func:`nilearn.mass_univariate.permuted_ols` function returns the
p-values computed with a permutation test.

.. literalinclude:: ../../plot_haxby_mass_univariate.py
   :start-after: from nilearn.input_data import NiftiMasker
   :end-before: ### Load Haxby dataset

.. literalinclude:: ../../plot_haxby_mass_univariate.py
   :start-after: ### Perform massively univariate analysis with permuted OLS
   :end-before: neg_log_pvals_unmasked

The number of tests performed is generally large when full-brain
analysis is performed (> 50,000 voxels). This increases the
probability of finding a significant activation by chance, a
phenomenon that is known to statisticians as the *multiple comparisons
problem*. It is therefore recommended to correct the p-values to take
into account the multiple tests. *Bonferroni correction* consists of
multiplying the p-values by the number of tests (while making sure the
p-values remain smaller than 1). Thus, we control the occurrence of one
false detection *at most*, the so-called *family-wise error control*.
A similar control can be performed when performing a permutation test:
For each permutation, only the maximum value of the F-statistic across
voxels is considered and is used to build the null distribution. It is
crucial to assume that the distribution of the signal is the same in
every voxel so that the F-statistics are comparable. This correction
strategy is applied in Nilearn's
:func:`nilearn.mass_univariate.permuted_ols` function.

.. figure:: ../auto_examples/images/plot_haxby_mass_univariate_1.png
   :target: ../auto_examples/plot_haxby_searchlight.html
   :align: center
   :scale: 60

We observe that the results obtained with a permutation test are less
conservative than the ones obtained with a Bonferroni correction
strategy.

.. [1]

    The *p-value* is the probability of getting the observed values
    assuming that nothing happens (i.e. under the null hypothesis).
    Therefore, a small *p-value* indicates that there is a small chance
    of getting this data if no real difference existed, so the observed
    voxel must be significant.

.. [2]

    When the tested variate is a scalar (test of the *intercept*)
    --which corresponds to a one sample test--, no swapping can be
    performed but one can estimate the null distribution by assuming
    symmetry about some reference value. When this value is zero, one can
    randomly swap the sign of the target variates (the imaging
    signal). Nilearn's
    :func:`nilearn.mass_univariate.permuted_ols` function automatically
    adopts the suitable strategy according to the input data.
