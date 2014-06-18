.. _massively_univariate_analysis:


=============================
Massively univariate analysis
=============================

.. currentmodule:: nilearn.mass_univariate

Basics
======

We consider a standard neuroimaging analysis framework, where observations
are several brain volumes that have been registered. The volumes may come from
various acquisitions performed on the same subject (intra-subject analysis)
or on different subjects (inter-subject analysis).
In *massively univariate analysis*, a linear model is fit independently
at each voxel across realigned images. The purpose is to uncover
voxels where the brain activity has a significant correlation with a
given explanatory variate (e.g. performed cognitive task, or any
subject-specific characteristic).

The standard approach to brain mapping is performed using *Statistical
Parametric Mapping* (SPM), using ANOVA (analysis of variance), and
parametric tests (F-tests ot t-tests).
For instance, in the plot_haxby_searchlight example we compute the
*p-values* of the voxels [1]_.  To display the results, we use the
negative log of the p-value.

.. figure:: ../auto_examples/images/plot_haxby_searchlight_2.png
   :target: ../auto_examples/plot_haxby_searchlight.html
   :align: center
   :scale: 60

.. literalinclude:: ../../plot_haxby_searchlight.py
    :start-after: ### F_score results

Parametric scores can be converted into p-values using a reference
theoretical distribution, which is known under specific assumptions
(hence the name *parametric*).

The number of tests performed is generally large when full-brain
analysis is performed (> 50,000 voxels). This increases the
probability of finding a significant activation by chance, a
phenomenon that is known to statisticians as the *multiple comparisons
problem*. It is therefore recommended to correct the p-values to take
into account the multiple tests. *Bonferroni correction* consists of
multiplying the p-values by the number of tests (while making sure the
p-values remain smaller than 1). Thus, we control the occurrence of one
false detection *at most*, the so-called *family-wise error control*.

Advanced techniques
===================

Obtaining a better detection accuracy with non-parametric testing
-----------------------------------------------------------------

In practice, neuroimaging signal has a complex structure that might
not match these assumptions. An exact, non-parametric *permutation
test* can be performed as an alternative to the parametric test: the
residuals of the model are permuted so as to break any effect and the
corresponding decision statistic is recomputed. One thus builds the
distribution of the decision statistic under the hypothesis that there
is no relationship between the tested variates and the target
variates.  In neuroimaging, this is generally done by swapping the
signal values of all voxels while the tested variables remain
unchanged [2]_. A voxel-wise analysis is then performed on the
permuted data. The relationships between the image descriptors and the
tested variates are broken while the value of the signal in each
particular voxel can be observed with the same probability than the
original value associated to that voxel. Note that it is hereby
assumed that the signal distribution is the same in every
voxel. Several data permutations are performed (typically 10,000)
while the scores for every voxel and every data permutation is
stored. The empirical distribution of the scores is thus constructed
(under the hypothesis that there is no relationship between the tested
variates and the neuroimaging signal, the so-called *null-hypothesis*)
and we can compare the original scores to that distribution: The
higher the rank of the original score, the smaller is its associated
p-value. The :func:`nilearn.mass_univariate.permuted_ols` function
returns the p-values computed with a permutation test.

.. literalinclude:: ../../plot_haxby_mass_univariate.py
   :start-after: from nilearn.input_data import NiftiMasker
   :end-before: ### Load Haxby dataset

.. literalinclude:: ../../plot_haxby_mass_univariate.py
   :start-after: ### Perform massively univariate analysis with permuted OLS
   :end-before: neg_log_pvals_unmasked

A family-wise error control (see Basics section) can be performed when
performing a permutation test: For each permutation, only the maximum
value of the F-statistic across voxels is considered and is used to
build the null distribution. It is crucial to assume that the
distribution of the signal is the same in every voxel so that the
F-statistics are comparable. This correction strategy is applied in
Nilearn's :func:`nilearn.mass_univariate.permuted_ols` function.

.. figure:: ../auto_examples/images/plot_haxby_mass_univariate_1.png
   :target: ../auto_examples/plot_haxby_searchlight.html
   :align: center
   :scale: 60

We observe that the results obtained with a permutation test are less
conservative than the ones obtained with a Bonferroni correction
strategy.

In Nilearn's :func:`nilearn.mass_univariate.permuted_ols` function, we
permute a parametric t-test. Unlike F-test, a t-test can be signed
(*one-sided test*), that is both the absolute value and the sign of an
effect are considered. Thus, only positive effects
can be focused on.  It is still possible to perform a two-sided test
equivalent to a permuted F-test by setting the argument
`two_sided_test` to `True`. In the example above, we do perform a two-sided
test but add back the sign of the effect at the end using the t-scores obtained
on the original (non-permuted) data. Thus, we can perform two one-sided tests
(a given contrast and its opposite) for the price of one single run.
The example results can be interpreted as follows: viewing faces significantly
activates the Fusiform Face Area as compared to viewing houses, while viewing
houses does not reveals significant supplementary activations as compared to
viewing faces.


Achieving more sensitivity with Randomized Parcellation Based Inference
-----------------------------------------------------------------------

Massively univariate analysis with a standard F/t-test has been shown
to yield results that are poorly reproducible across groups of
subjects.  The reproducibility as well as the sensitivity of the
method can be improved by smoothing the data prior to
analysis. Another alternative (*parcel-based analysis/inference*) is
to use signal averages within predefined parcels as new data
descriptors. Not only this approach reduces the number of features and
hinders the multiple comparison problem, but it also convey more
sensitivity than a raw voxel-wise analysis as it compensates for
subject misalignement (just as smoothing does).

Unfortunately, it turns out that the results of parcel-based analysis
highly depend on the choice of the pre-defined brain parcellation and
are therefore not reproducible neither.  *Randomized Parcellation
Based Inference (RPBI)* stabilizes the parcel-based approach by viewing the
parcellation as a random variable over which the analysis results can
be integrated. Equivalently, it relaxes the choice of a specific
parcellation.  It also has the advantage over a standard parce-based
analysis that it yields results at the voxel level (i.e. one obtain
continuous maps).
On top of being more reproducible, the results also convey more sensitivity.

Several Nilearn's examples illustrate the gain in sensitivity associated to
Randomized Parcellation Based Inference:

They all start with loading and masking the data

.. literalinclude:: ../../plot_haxby_rpbi.py
   :start-after:  ### Mask data
   :end-before: ### Restrict to faces and houses

The Haxby example shows that only a few observations can be used to uncover
relevant activation patterns.

.. literalinclude:: ../../plot_haxby_rpbi.py
   :start-after: from nilearn.mass_univariate import randomized
   :end-before: ### Load Haxby dataset

.. literalinclude:: ../../plot_haxby_rpbi.py
   :start-after: ### Randomized Parcellation Based Inference
   :end-before: ### Visualization

.. figure:: ../auto_examples/images/plot_haxby_rpbi_1.png
   :target: ../auto_examples/plot_haxby_rpbi.html
   :align: center
   :scale: 60

The Localizer example shows how sensitive the method can be as compared to
standard voxel-level inference:

.. literalinclude:: ../../plot_localizer_rpbi.py
   :start-after: from nilearn.mass_univariate import randomized
   :end-before: ### Mask data

.. literalinclude:: ../../plot_localizer_rpbi.py
   :start-after: ### Randomized Parcellation Based Inference
   :end-before: ### Visualization

.. figure:: ../auto_examples/images/plot_localizer_rpbi_1.png
   :target: ../auto_examples/plot_localizer_rpbi.html
   :align: center
   :scale: 60

The Oasis example shows the behavior of Randomized Parcellation Based Inference
on a voxel-based morphometry dataset:

.. literalinclude:: ../../plot_oasis_rpbi.py
   :start-after: ### Randomized Parcellation Based Inference
   :end-before: ### Show results

.. figure:: ../auto_examples/images/plot_oasis_rpbi_1.png
   :target: ../auto_examples/plot_oasis_rpbi.html
   :align: center
   :scale: 60

.. figure:: ../auto_examples/images/plot_oasis_rpbi_2.png
   :target: ../auto_examples/plot_oasis_rpbi.html
   :align: center
   :scale: 60

.. topic:: **Runtime**

   On a typical fMRI group study involving 25 subjects described by
   ~40,000 voxels each, Randomized Parcellation Based Inference
   finished after about 30 minutes using one single CPU (Intel Core 2
   Duo, T9600 @ 2.80GHz) and default arguments (100 parcellations of
   1,000 parcels each and 10,000 permutations).  The computation time
   can easily be lowered by using several, more recent CPUs (ideally,
   one would set `n_jobs=-1` and use a workstation).



.. [1]

    The *p-value* is the probability of getting the observed values
    assuming that no effect is present (i.e. under the null hypothesis).
    Therefore, a small *p-value* indicates that there is a small chance
    of getting this data if no real difference existed; in that sense, the
    effect is considered as *significant*.

.. [2]

    When the variate tested is a scalar (test of the *intercept*)
    --which corresponds to a one sample test--, no swapping can be
    performed but one can estimate the null distribution by assuming
    symmetry about some reference value. When this value is zero, one can
    randomly swap the sign of the target variates (the imaging
    signal). Nilearn's
    :func:`nilearn.mass_univariate.permuted_ols` function automatically
    adopts the suitable strategy according to the input data.
