.. _searchlight:
.. currentmodule:: nilearn.decoding

===========================================================
Searchlight : finding voxels containing information
===========================================================

This page overviews searchlight analyses and how they are approached
in nilearn with the :class:`SearchLight` estimator.


Principle of the Searchlight
============================

:class:`SearchLight` analysis was introduced in [:footcite:t:`Kriegeskorte2006`], and consists of scanning the brain with a *searchlight*.
Briefly, a ball of given radius is scanned across the brain volume and the prediction accuracy of a classifier trained on the corresponding :term:`voxels<voxel>` is measured.

Searchlights are also not limited to :term:`classification`; :term:`regression` (e.g., [:footcite:t:`Kahnt2011`]) and representational similarity analysis (e.g., [:footcite:t:`Clarke2014`]) are other uses of searchlights.
Currently, only :term:`classification` and :term:`regression` are supported in nilearn.

.. topic:: **Further Reading**

    For a critical review on searchlights, see [:footcite:t:`Etzel2013`].


Preparing the data
==================

:class:`SearchLight` requires a series of brain volumes as input, ``X``, each with
a corresponding label, ``y``. The number of brain volumes therefore correspond to
the number of samples used for decoding.

Masking
-------

One of the main elements that distinguish :class:`SearchLight` from other
algorithms is the notion of structuring element that scans the entire volume.
This has an impact on the masking procedure.

Two masks are used with :class:`SearchLight`:

- *mask_img* is the anatomical mask
- *process_mask_img* is a subset of the brain mask and defines the boundaries
  of where the searchlight scans the volume. Often times we are interested in
  only performing a searchlight within a specific area of the brain (e.g.,
  frontal cortex). If no *process_mask_img* is set, then :class:`nilearn.decoding.SearchLight`
  defaults to performing a searchlight over the whole brain.

*mask_img* ensures that only :term:`voxels<voxel>` with usable signals are included in the
searchlight. This could be a full-brain mask or a gray-matter mask.


Setting up the searchlight
==========================

Classifier
----------

The classifier used by default by :class:`SearchLight` is LinearSVC with C=1 but
this can be customized easily by passing an estimator parameter to the
Searchlight. See scikit-learn documentation for :sklearn:`other classifiers
<supervised_learning.html>`. You can
also pass scikit-learn :sklearn:`Pipelines <modules/compose.html>`
to the :class:`SearchLight` in order to combine estimators and preprocessing steps
(e.g., feature scaling) for your searchlight.

Score function
--------------

Metrics can be specified by the "scoring" argument to the :class:`SearchLight`, as
detailed in the :sklearn:`scikit-learn documentation
<modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules>`

Cross validation
----------------

:class:`SearchLight` will iterate on the volume and give a score to each :term:`voxel`.
This score is computed by running a classifier on selected :term:`voxels<voxel>`.
In order to make this score as accurate as possible (and avoid overfitting),
cross-validation is used.

Cross-validation can be defined using the "cv" argument. As it
is computationally costly, *K*-Fold cross validation with *K* = 3 is set as the
default. A :sklearn:`scikit-learn cross-validation generator
<modules/classes.html#splitter-classes>` can also
be passed to set a specific type of cross-validation.

Leave-one-run-out cross-validation (LOROCV) is a common approach for searchlights.
This approach is a specific use-case of grouped cross-validation, where the
cross-validation folds are determined by the acquisition runs. The held-out fold
in a given iteration of cross-validation consist of data from a separate run,
which keeps training and validation sets properly independent. For this reason,
LOROCV is often recommended. This can be performed by using :sklearn:`LeaveOneGroupOut
<modules/generated/sklearn.model_selection.LeaveOneGroupOut.html>`,
and then setting the group/run labels when fitting the estimator.

Sphere radius
-------------

An important parameter is the radius of the sphere that will run through
the data. The sphere size determines the number of voxels/features to use
for :term:`classification` (i.e. more :term:`voxels<voxel>` are included with larger spheres).

.. note::

    :class:`SearchLight` defines sphere radius in millimeters; the number
    of :term:`voxels<voxel>` included in the sphere will therefore depend on the
    :term:`voxel` size.

    For reference, [:footcite:t:`Kriegeskorte2006`] use a 4mm radius because it yielded
    the best detection performance in their simulation of 2mm isovoxel data.

Visualization
=============

Searchlight
-----------

The results of the searchlight can be found in the ``scores_`` attribute of the
:class:`SearchLight` object after fitting it to the data. Below is a
visualization of the results from :ref:`Searchlight analysis of face
vs house recognition <sphx_glr_auto_examples_02_decoding_plot_haxby_searchlight.py>`.
The searchlight was restricted to a slice in the back of the brain. Within
this slice, we can see that a cluster of :term:`voxels<voxel>` in visual cortex
contains information to distinguish pictures showed to the volunteers,
which was the expected result.

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_searchlight_001.png
   :target: ../auto_examples/02_decoding/plot_haxby_searchlight.html
   :align: center
   :scale: 80

.. seealso::

   * :ref:`plotting`

Comparing to massively univariate analysis: F_score or SPM
----------------------------------------------------------

The standard approach to brain mapping is performed using *Statistical
Parametric Mapping* (:term:`SPM`), using :term:`ANOVA` (analysis of
variance), and parametric tests (F-tests ot t-tests).
Here we compute the *p-values* of the :term:`voxels<voxel>` [1]_.
To display the results, we use the negative log of the p-value.

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_searchlight_002.png
   :target: ../auto_examples/02_decoding/plot_haxby_searchlight.html
   :align: center
   :scale: 80

Parametric scores can be converted into p-values using a reference
theoretical distribution, which is known under specific assumptions
(hence the name *parametric*). In practice, neuroimaging signal has a
complex structure that might not match these assumptions. An exact,
non-parametric *permutation test* can be performed as an alternative
to the parametric test: the residuals of the model are permuted so as
to break any effect and the corresponding decision statistic is
recomputed. One thus builds the distribution of the decision statistic
under the hypothesis that there is no relationship between the tested
variates and the target variates.  In neuroimaging, this is generally
done by swapping the signal values of all :term:`voxels<voxel>` while the tested
variables remain unchanged [2]_. A voxel-wise analysis is then
performed on the permuted data. The relationships between the image
descriptors and the tested variates are broken while the value of the
signal in each particular :term:`voxel` can be observed with the same
probability than the original value associated to that :term:`voxel`.
Note that it is hereby assumed that the signal distribution is the same in
every :term:`voxel`. Several data permutations are performed (typically
10,000) while the scores for every :term:`voxel` and every data permutation
is stored. The empirical distribution of the scores is thus
constructed (under the hypothesis that there is no relationship
between the tested variates and the neuroimaging signal, the so-called
*null-hypothesis*) and we can compare the original scores to that
distribution: The higher the rank of the original score, the smaller
is its associated p-value. The
:func:`nilearn.mass_univariate.permuted_ols` function returns the
p-values computed with a permutation test.

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
:term:`voxels<voxel>` is considered and is used to build the null distribution.
It is crucial to assume that the distribution of the signal is the same in
every :term:`voxel` so that the F-statistics are comparable.
This correction strategy is applied in nilearn
:func:`nilearn.mass_univariate.permuted_ols` function.

.. figure:: ../auto_examples/07_advanced/images/sphx_glr_plot_haxby_mass_univariate_001.png
   :target: ../auto_examples/07_advanced/plot_haxby_mass_univariate.html
   :align: center
   :scale: 60

We observe that the results obtained with a permutation test are less
conservative than the ones obtained with a Bonferroni correction
strategy.

In nilearn :func:`nilearn.mass_univariate.permuted_ols` function, we
permute a parametric t-test. Unlike F-test, a t-test can be signed
(*one-sided test*), that is both the absolute value and the sign of an
effect are considered. Thus, only positive effects
can be focused on.  It is still possible to perform a two-sided test
equivalent to a permuted F-test by setting the argument
``two_sided_test`` to ``True``. In the example above, we do perform a two-sided
test but add back the sign of the effect at the end using the t-scores obtained
on the original (non-permuted) data. Thus, we can perform two one-sided tests
(a given contrast and its opposite) for the price of one single run.
The example results can be interpreted as follows: viewing faces significantly
activates the Fusiform Face Area as compared to viewing houses, while viewing
houses does not reveal significant supplementary activations as compared to
viewing faces.


.. [1]

    The *p-value* is the probability of getting the observed values
    assuming that nothing happens (i.e. under the null hypothesis).
    Therefore, a small *p-value* indicates that there is a small chance
    of getting this data if no real difference existed, so the observed
    voxel must be significant.

.. [2]

    When the variate tested is a scalar (test of the *intercept*)
    --which corresponds to a one sample test--, no swapping can be
    performed but one can estimate the null distribution by assuming
    symmetry about some reference value. When this value is zero, one can
    randomly swap the sign of the target variates (the imaging
    signal). nilearn
    :func:`nilearn.mass_univariate.permuted_ols` function automatically
    adopts the suitable strategy according to the input data.

.. topic:: **Example code**

   All the steps discussed in this section can be seen implemented in
   :ref:`a full code example <sphx_glr_auto_examples_02_decoding_plot_haxby_searchlight.py>`.

References
==========

.. footbibliography::
