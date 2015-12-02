==================================
Group-sparse covariance estimation
==================================

..
    Explain which case is implemented (p=2, unpenalized diagonal)

.. currentmodule:: nilearn.connectome


This page gives technical information on the
:func:`group_sparse_covariance` function and related. This is mainly
useful for developers or people that want to know more about
implementation.

Description
===========

:func:`group_sparse_covariance`, and :class:`GroupSparseCovariance` are
two different interfaces to an implementation of the algorithm described
in this article:

    Jean Honorio and Dimitris Samaras.
    "Simultaneous and Group-Sparse Multi-Task Learning of Gaussian Graphical
    Models". arXiv:1207.4255 (17 July 2012). http://arxiv.org/abs/1207.4255.

The goal of the algorithm is to take a set of K covariance matrices as
input, and estimate a set of K sparse precision matrices, using a
penalized maximum-likelihood criterion. The penalization has been
devised to enforce a common sparsity pattern in all precision
matrices. The structure is of a block coordinate descent, with a line
search as innermost loop.

The present implementation relies solely on NumPy, SciPy and Scikit-Learn.
Nilearn contains only Python code.

In addition to the basic algorithm described in the article, several
additions were implemented:

- computation of bounds for the regularization parameter
- several stopping criteria
- an ad-hoc cross-validation algorithm
- signals synthesis for testing purposes

These are described in the rest of this page. An overview of the design
choices and the history of the development is also given.


Numerical stability
===================

The algorithm proved to be rather numerically stable for a wide range
of inputs. It turned out that the condition numbers of the input
matrices do not have any significant effect on numerical stability.
What is relevant is:

- covariance matrix symmetry: input covariances matrices in
  :func:`group_sparse_covariance` must be as symmetric as possible.
  This is true in general: a small discrepancy in symmetry tends to be
  amplified. For this reason, our functions computing covariances ensure
  symmetry.
- covariance matrix normalization: using correlation matrices or
  signals with unit variance is mandatory when a large number of
  signals is to be processed.
- normalization of the number of samples: the objective to be
  optimized contains a sum of terms weighted by the number of samples
  available for each subject. The sum of these weights must be
  normalized to a small constant number (1 in the current
  implementation). Failing to do this leads quickly to instability,
  because too large numbers are used in the computation.
- an on-line computation of an inverse is performed in function
  `_update_submatrix`. For large problems, this is faster than
  computing the full inverse each time, but gives unfortunately less
  precision. In particular, symmetry is not always perfect, that's why
  it is enforced at the end on the final result.
- the Newton-Raphson tolerance value has no influence on numerical
  stability, unless very large values (like 0.5) are used.

The `debug` keyword in :func:`group_sparse_covariance` activates a set
of numerical consistency checks (mainly that matrices are s.p.d.) that
can be useful to track down numerical instability problems.


Execution time
==============

The `line profiler <http://pythonhosted.org/line_profiler/>`_ from
Robert Kern was used to locate execution time bottlenecks. Its
overhead proved not to be negligible (around 50% more execution time
when activated), and not evenly distributed in code lines. Global
execution times have also been measured to ensure that the findings
were valid. As the code in :func:`group_sparse_covariance` is highly
serial, and rather low-level, some lines have to be executed a very
large number of times (10^6 times is easily reached), one of the
bottlenecks is thus the Python interpreter overhead. Optimizing then
boils down to reducing the number of code lines and function calls in
the most executed parts: the Newton-Raphson (line search) loop. It is
for this reason that it has been written inline instead of calling
Scipy's version (it saves a lot of lines and calls). A lot of small
optimizations of this kind have been done. It is possible that some of
these optimizations give less numerical precision that the naive
operation. But the original author could not tell.

Speed optimization has been performed by checking the wall-clock time
required to get to a given precision, and not the number of
iterations. This is what "fast" means in practice: short overall
execution time. Tuning of the Newton-Raphson (NR) loop gives a good
example of the validity of this approach: the goal was to set the
tolerance on the result. Using a large value reduces the number of
iterations for NR, saving a lot of time. On the other hand, a loose
tolerance increases the number of iterations in the coordinate descent
loop, therefore increasing the overall execution time. Measurement
proved that tight tolerances were leading to faster convergence rates.

Care has been taken to use proper ordering of data in arrays. In
particular, three-dimensional arrays containing precision matrices are
in Fortran order, to get prec[..., k] contiguous for any k. This is
important to avoid copies by lapack/atlas functions, such as matrix
inverse or dot product. It is also consistent with arrays returned by
`nibabel.load`.

An optimization that can be performed, but couldn't be implemented
short of having proper linalg functions for it is to process only half
of each matrix: all are symmetric. This would improve numerical
stability while saving some execution time. Part of this could be done
with versions of Scipy that weren't available on the targeted systems
at the time of writing (Ubuntu 10.04 and 12.04).

Memory optimization hasn't been performed, because all functions
process covariance matrices only, that are quite small compared to the
signals from which they are generated.


Synthetic dataset
=================
For testing purposes, a function for synthesis of signals based on
sparse precision matrices has been written:
`nilearn._utils.testing.generate_group_sparse_gaussian_graphs`.
Synthesizing such signals is a hard problem that wasn't solved in the
present implementation. It is hopefully good enough.

This function generates n_subjects time, n_features signals with a
variable number of samples. Every subject has the same number of
features (i.e. signals), for a given subject every signal has the same
number of samples, but between two subjects, the sample number can
differ. This structure is close to what is available in practice.

Here is how signals are generated:

- a "topology" matrix containing only zero and ones is generated. This
  will govern the sparsity pattern of the precision matrices.
- for each subject, a precision matrix is generated by replacing every
  1 in the topology matrix by a random positive number, then
  multiplying the resulting matrix by its transpose to get a positive
  definite matrix. This is a way to get a sparse definite positive
  matrix.
- inverting precision matrices gives covariance matrices, that are in
  turn used to generate signals.

The hardest part is generating sparse symmetric positive definite
matrices, while controling the sparsity level. With the present
scheme, only the location of zeros in the *square root* of the
precision matrices can be specified. Therefore the final sparsity
level depends not only on the initial sparsity level, but also on the
precise location of zeros. Two different sparsity patterns with the
same number of zeros can lead to two significantly different sparsity
level in precision matrices. In practice, it means that for a given
value of the `density` parameter in
`nilearn._utils.testing.generate_group_sparse_gaussian_graphs`,
the actual number of zeros in the precision matrices can fluctuate
widely depending on the random number generation.

The condition number of the precision matrices depends on the range of
numbers used to fill the off-diagonal part. The shorter the range (and
the closer to zero) the lower the condition number.

This generator is useful for debugging and testing. However, the
signals obtained are significantly different from those from
experimental data. Some unrealistic features: each signal has a
perfectly white spectrum (any two samples are decorrelated), and there
is no global additive noise (no confounds whatsoever).


Stopping criteria
=================

As with any iterative algorithm, iteration should be stopped at some
point, which is still mostly an open problem. Several heuristic
techniques have been tested and implemented.


Maximum number of iterations
----------------------------

The simplest way of stopping optimization is to always execute a fixed
number of iterations. This is simple but most of the time gives slow
or bad results. The convergence rate highly depends on the number of
features (size of one covariance matrix), and on the value of the
regularization parameter (high values give fast convergence, and low
values slow convergence). If the requested iteration number is too low,
large or weakly regularized problems will be far from the optimum. On
the other hand, if the requested iteration number is too large, a lot
of time is wasted for almost no gain.

Duality gap
-----------

A better way to stop iteration is to use an upper bound on the duality
gap value, since the problem is convex. This is performed in
`group_sparse_covariance_costs`. The article by Honorio &
Samaras gives the formula for the dual cost, and proves that the
derived bound at optimum is tight (strong duality holds). However, the
dual problem is *not* solved by this algorithm, thus bounding the
duality gap away from the optimum implies finding a feasible dual
point. This proved to be quite hard in practice, because one has to
compute positive semi-definite matrices under a norm constraint.

What is done is computing an estimate for a dual point using the
formula relating the primal and dual points at optimum. This estimate
does not satisfies in general the norm constraint. It is then
projected on the corresponding ball. Most of the time, this is enough
to ensure the required positive definiteness of another quantity. As
the primal point is coming close to the optimal, the estimate for the
dual point also comes close to the optimal, and the initial estimate
is closer and closer to the norm ball.

But there are cases for which the projection is not enough to get to a
feasible point. No solution to this problem (simultaneous projection
on a norm ball and on a set of positive definite matrices) has been
found. In that case, an easier to compute but non-tight bound is used
instead.

In practice, using the duality gap value to stop iteration leads to
guaranteed uncertainty on the objective value, in any case. No time is
lost on over optimizing rapidly converging problems. However, the
duality gap criterion can lead to prohibitive computation time on
slowly converging cases. In practice, finding a proper value for the
duality gap uncertainty can be tricky, because it is most easily given
as an absolute uncertainty on an objective whose value highly depends
on input data.


Variation of norm of estimate
-----------------------------

Depending on the application at hand, giving an uncertainty on the
precision matrices instead of the objective can be useful. This is
partly achieved by computing the change of the precision estimate
between two iterations. Optimization is stopped once this value goes
below a threshold. The maximum norm (maximum of the absolute value of
the difference) is used in the current implementation. It ensures that
all coefficients vary less than the threshold when optimization is
stopped.

This technique it is only a way to stop iterating based on the
estimate value instead of the criterion value. It does *not* ensure a
given uncertainty on the estimate. This has been tested on synthetic
and real fMRI data: using two different starting points leads to two
estimates that can differ (in max norm) by more than the threshold
(see next paragraph). However, it has the same property as the duality
gap criterion: quickly converging cases use fewer iterations than
slower cases. From a performance point of view, this is a good thing.

One advantage of this criterion is that the threshold value does not
depend significantly on the input data. Matrix coefficients can be
requested to change less than e.g. 0.1 for any size of the input.


Initial estimate value
----------------------

One of the possible way to reduce the computation time of an iterating
algorithm is to start with a initial guess that is as close as
possible to the optimum. In the present case, two initializations were
tested: using a diagonal matrix (with variance of input signals), or
using a Ledoit-Wolf estimate. It turned out that even if the
Ledoit-Wolf initialization allows for starting with a better value for
the objective, the difference with the diagonal matrix initialization
dwindles rather fast. It does not allow any significant speedup
in practice.

Only initialization by the diagonal matrix, as
in the original paper, has been implemented.


Modifying the stopping criterion
--------------------------------

Modifying the stopping criterion is more complicated than specifying
the initial estimate, since it requires to gain access to the
algorithm internals. This is achieved by a technique close to
aspect-oriented programming: a function can be provided by the user,
that will be called after each iteration, with all internal values as
parameter. If that function returns True, iteration is stopped.
Changing the stopping criterion is thus just a matter of writing a
function and passing it to :func:`group_sparse_covariance`. The same
feature can be used to study the algorithm convergence properties. An
example is the `EarlyStopProbe` class used by the
cross-validation object.


Cross-validation algorithm
==========================

An ad-hoc cross-validation scheme has been implemented in the
:class:`GroupSparseCovarianceCV` class. This implementation is
significantly faster than the "naive" cross-validation scheme.

The cross-validating object performs to distinct tasks: the first one
is to select a value for the regularization parameter, the second is
fitting the precision matrices for the selected parameter. The latter
is identical to what has been described in the previous parts, we thus
focus only on the former.


Principle of cross-validation
-----------------------------

Cross-validation consists in splitting the input samples into two
different sets: train and test. For several values of the
regularization parameter, a model is fit on the train set, and the
generalization performance is assessed on the test set, by computing
the unpenalized criterion (log-likelihood) using the precisions
matrices obtained on the train set with the empirical covariances of
the test set. The chosen regularization parameter is given by the best
criterion on the test set.

The simplest scheme is here to fit many models, for many values of the
regularization parameter alpha, and pick up the best value afterward.
It works in any case, but is very time-consuming. A cleverer scheme
is used, that is very close to that used in the graph lasso
implementation in Scikit-Learn.


Bounds on alpha
---------------

The simplest and fastest thing is to get bounds for the value of
alpha. Above a critical value, the optimal precision matrices are
fully sparse (i.e. diagonal). This critical value depends on the input
covariance matrices, and can be obtained by `compute_alpha_max`.
The formula for computing this critical value can be obtained with
techniques presented in:

    Duchi, John, Stephen Gould, and Daphne Koller. 'Projected Subgradient
    Methods for Learning Sparse Gaussians'. ArXiv E-prints 1206 (1 June
    2012): 3249.

This very same method can be also used for determining a lower
critical value, for which the optimal precision matrices are fully
dense (no zero values). In practice, this critical value is zero if
there is a zero in the input matrices. For this reason, the second
value returned by `compute_alpha_max` is that under which all
coefficients *that can be non-zero* are non-zero in the optimal
precision matrices.


Iterative grid search
---------------------

Getting the regularization parameter optimal value is equivalent to
finding the location of the maximum on the curve log-likelihood vs
regularization parameter. In practice this curve is rather smooth,
with only a single maximum. This can be exploited to reduce the number
of parameter values to try. The strategy used in this implementation
consists of a iterative grid search: the maximum value is searched on
a very loose grid of parameter values (by default, only 4 values are
used), then a tighter grid near the found maximum is computed, and so
on. This allows for a very precise determination of the maximum
location while reducing a lot the required evaluation number. The code
is very close to what is done in
:class:`sklearn.covariance.GraphLassoCV`.


Warm restart
------------

During each step of the grid search, a set of regularization
parameters has to been tested. The straighforward strategy consists of
running independently each fit, each optimization being started with
basically the same initial value (diagonal matrices). Execution time
can be reduced by running all optimizations sequentially, and using
the final result of one as initial value for the next. This goes
faster because it saves part of the optimization trajectory starting
with the second one. However, there is a real gain in execution time
only if the parameter values are ordered from the largest to the
smallest (and not the other way).

The usefulness of this scheme depends on several things. First, using
warm restart does not gives exactly the same result as running
independant optimizations, because optimization paths are not the
same. This is not an issue for cross-validation, since there are many
other larger sources of fluctuations. It has been checked that in
practice, the selected value does not change. Second, using warm
restart forces running all optimization one after another: there is no
parallelism at all. However, this is true only for a given fold: when
n folds are used, n such evaluations can be run in parallel. Thus, the
fact that warm restart gives faster evaluation compared to fixed
initialization depends on the number of folds, and the number of
computation cores. No more cores that the number of folds can be used
at the same time. Thus, if the number of folds is much smaller than
the number of usable cores, warm restart slows down computation (note
that if the goal is energy efficiency, not speed, warm restart is
always a good idea.) This argument is also valid for the iterative
grid search: if many cores are available, the brute-force grid search
is faster than the iterative scheme, just because every point can be
explored simultaneously, without waiting for the previous step to
finish. Many more evaluations are performed, but the overall running
time is limited of by the slowest evaluation. The choice of these
schemes (iterative grid search and warm restart) has been made in the
present implementation because the targeted hardware is a commodity
computer, with a moderate number of cores (4 to 16). More cores (and
memory) will probably be available in future years, these schemes
could be removed easily.


Stopping criterion
------------------

Finding the regularization parameter optimal value is equivalent to
finding a maximum. But since only the location of the maximum (not its
value) is of interest, any curve that peaks at the same location than
the log-likelihood can be used.

Implicitely, the curve whose maximum is sought is supposed to be
obtained after convergence for any value of alpha. This is never the
case in practice: a stopping criterion has to be used. In the present
implementation, the variation criterion gives results that seem to be
consistent with what would be obtained at convergence (that is: the
log-likelihood-vs-alpha curve seems to be close to convergence). This
can be pushed one step further: any stopping criterion that gives the
*same maximum location* can be used instead. We stress that only the
location is important: the curve can be anything apart from that.

It was found that stopping iteration just after the log-likelihood has
reached a maximum works in most cases. The obtained log-likelihood vs
alpha curve is different, but its maximum is the same as with the
variation criterion stopping. It is also faster (2 times to 4 times
in our tests).

In more detail: for a given value of alpha, start optimization. After
each step, compute the log-likelihood on the test set. If the current
value is smaller than the previous one, then stop. The variation
criterion is also computed for the rare cases when the log-likelihood
never decreases, and a maximum number of iterations is enforced,
to limit the time spent optimizing in any case.

It is possible to disable the first criterion with `the
early_stopping` keyword in :class:`GroupSparseCovarianceCV`. In that
case, only the two latter criteria are used. This provides a mean to
test for the validity of the heuristic.
