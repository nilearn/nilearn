"""Massively Univariate Linear Model estimated \
with OLS and permutation test."""
# Author: Benoit Da Mota, <benoit.da_mota@inria.fr>, sept. 2011
#         Virgile Fritsch, <virgile.fritsch@inria.fr>, jan. 2014
import sys
import time
import warnings

import joblib
import nibabel as nib
import numpy as np
from nilearn import image
from nilearn.masking import apply_mask
from nilearn.mass_univariate._utils import (
    _calculate_cluster_measures,
    _calculate_tfce,
    _normalize_matrix_on_axis,
    _null_to_p,
    _orthonormalize_matrix,
    _t_score_with_covars_and_normalized_design,
)
from scipy import stats
from scipy.ndimage import generate_binary_structure, label
from sklearn.utils import check_random_state


def _permuted_ols_on_chunk(
    scores_original_data,
    tested_vars,
    target_vars,
    thread_id,
    threshold=None,
    confounding_vars=None,
    masker=None,
    n_perm=10000,
    n_perm_chunk=10000,
    intercept_test=True,
    two_sided_test=True,
    tfce=False,
    tfce_original_data=None,
    random_state=None,
    verbose=0,
):
    """Perform massively univariate analysis with permuted OLS on a data chunk.

    To be used in a parallel computing context.

    Parameters
    ----------
    scores_original_data : array-like, shape=(n_descriptors, n_regressors)
        t-scores obtained for the original (non-permuted) data.

    tested_vars : array-like, shape=(n_samples, n_regressors)
        Explanatory variates.

    target_vars : array-like, shape=(n_samples, n_targets)
        fMRI data. F-ordered for efficient computations.

    thread_id : int
        process id, used for display.

    threshold : :obj:`float`
        Cluster-forming threshold in t-scale.
        This is only used for cluster-level inference.
        If ``threshold`` is not None, but ``masker`` is, an exception will be
        raised.

        .. versionadded:: 0.9.2

    confounding_vars : array-like, shape=(n_samples, n_covars), optional
        Clinical data (covariates).

    masker : None or :class:`~nilearn.maskers.NiftiMasker` or \
            :class:`~nilearn.maskers.MultiNiftiMasker`, optional
        A mask to be used on the data.
        This is used for cluster-level inference and :term:`TFCE`-based
        inference, if either is enabled.
        If ``threshold`` is not None, but ``masker`` is, an exception will be
        raised.

        .. versionadded:: 0.9.2

    n_perm : int, optional
        Total number of permutations to perform, only used for
        display in this function. Default=10000.

    n_perm_chunk : int, optional
        Number of permutations to be performed. Default=10000.

    intercept_test : boolean, optional
        Change the permutation scheme (swap signs for intercept,
        switch labels otherwise). See :footcite:`Fisher1935`.
        Default=True.

    two_sided_test : boolean, optional
        If True, performs an unsigned t-test. Both positive and negative
        effects are considered; the null hypothesis is that the effect is zero.
        If False, only positive effects are considered as relevant. The null
        hypothesis is that the effect is zero or negative.
        Default=True

    tfce : :obj:`bool`, optional
        Whether to perform :term:`TFCE`-based multiple comparisons correction
        or not.
        Calculating TFCE values in each permutation can be time-consuming, so
        this option is disabled by default.
        The TFCE calculation is implemented as described in
        :footcite:t:`Smith2009a`.
        Default=False.

        .. versionadded:: 0.9.2

    tfce_original_data : None or array-like, \
            shape=(n_descriptors, n_regressors), optional
        TFCE values obtained for the original (non-permuted) data.

        .. versionadded:: 0.9.2

    random_state : int or None, optional
        Seed for random number generator, to have the same permutations
        in each computing units.

    verbose : int, optional
        Defines the verbosity level. Default=0.

    Returns
    -------
    scores_as_ranks_part : array-like, shape=(n_regressors, n_descriptors)
        The ranks of the original scores in ``h0_fmax_part``.
        When ``n_descriptors`` or ``n_perm`` are large, it can be quite long to
        find the rank of the original scores into the whole H0 distribution.
        Here, it is performed in parallel by the workers involved in the
        permutation computation.

    h0_fmax_part : array-like, shape=(n_perm_chunk, n_regressors)
        Distribution of the (max) t-statistic under the null hypothesis
        (limited to this permutation chunk).

    h0_csfwe_part, h0_cmfwe_part : array-like, \
            shape=(n_perm_chunk, n_regressors)
        Distribution of max cluster sizes/masses under the null hypothesis.
        Only calculated if ``masker`` is not None.
        Otherwise, these will both be None.

        .. versionadded:: 0.9.2

    tfce_scores_as_ranks_part : array-like, shape=(n_regressors, n_descriptors)
        The ranks of the original TFCE values in ``h0_tfce_part``.
        When ``n_descriptors`` or ``n_perm`` are large, it can be quite long to
        find the rank of the original scores into the whole H0 distribution.
        Here, it is performed in parallel by the workers involved in the
        permutation computation.

        .. versionadded:: 0.9.2

    h0_tfce_part : array-like, shape=(n_perm_chunk, n_regressors)
        Distribution of the (max) TFCE value under the null hypothesis
        (limited to this permutation chunk).

        .. versionadded:: 0.9.2

    References
    ----------
    .. footbibliography::

    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    n_samples, n_regressors = tested_vars.shape
    n_descriptors = target_vars.shape[1]

    # run the permutations
    t0 = time.time()
    h0_fmax_part = np.empty((n_regressors, n_perm_chunk))
    scores_as_ranks_part = np.zeros((n_regressors, n_descriptors))

    # Preallocate null arrays for optional outputs
    # Any unselected outputs will just return a None
    if tfce:
        h0_tfce_part = np.empty((n_regressors, n_perm_chunk))
        tfce_scores_as_ranks_part = np.zeros((n_regressors, n_descriptors))
    else:
        h0_tfce_part, tfce_scores_as_ranks_part = None, None

    if threshold is not None:
        h0_csfwe_part = np.empty((n_regressors, n_perm_chunk))
        h0_cmfwe_part = np.empty((n_regressors, n_perm_chunk))
    else:
        h0_csfwe_part, h0_cmfwe_part = None, None

    for i_perm in range(n_perm_chunk):
        if intercept_test:
            # sign swap (random multiplication by 1 or -1)
            target_vars = target_vars * (
                rng.randint(2, size=(n_samples, 1)) * 2 - 1
            )

        else:
            # shuffle data
            # Regarding computation costs, we choose to shuffle testvars
            # and covars rather than fmri_signal.
            # Also, it is important to shuffle tested_vars and covars
            # jointly to simplify t-scores computation (null dot product).
            shuffle_idx = rng.permutation(n_samples)
            tested_vars = tested_vars[shuffle_idx]
            if confounding_vars is not None:
                confounding_vars = confounding_vars[shuffle_idx]

        # OLS regression on randomized data
        perm_scores = np.asfortranarray(
            _t_score_with_covars_and_normalized_design(
                tested_vars, target_vars, confounding_vars
            )
        )

        # find the rank of the original scores in h0_fmax_part
        # (when n_descriptors or n_perm are large, it can be quite long to
        #  find the rank of the original scores into the whole H0 distribution.
        #  Here, it is performed in parallel by the workers involved in the
        #  permutation computation)
        # NOTE: This is not done for the cluster-level methods.
        if two_sided_test:
            # Get maximum absolute value for voxel-level FWE
            h0_fmax_part[:, i_perm] = np.nanmax(np.fabs(perm_scores), axis=0)
            scores_as_ranks_part += (
                h0_fmax_part[:, i_perm].reshape((-1, 1))
                < np.fabs(scores_original_data).T
            )
        else:
            # Get maximum value for voxel-level FWE
            h0_fmax_part[:, i_perm] = np.nanmax(perm_scores, axis=0)
            scores_as_ranks_part += (
                h0_fmax_part[:, i_perm].reshape((-1, 1))
                < scores_original_data.T
            )

        # Prepare data for cluster thresholding
        if tfce or (threshold is not None):
            arr4d = masker.inverse_transform(perm_scores.T).get_fdata()

        if tfce:
            # The TFCE map will contain positive and negative values if
            # two_sided_test is True, or positive only if it's False.
            # In either case, the maximum absolute value is the one we want.
            h0_tfce_part[:, i_perm] = np.nanmax(
                np.fabs(
                    _calculate_tfce(
                        arr4d,
                        bin_struct=_binary_structure(),
                        two_sided_test=two_sided_test,
                    )
                ),
                axis=(0, 1, 2),
            )
            tfce_scores_as_ranks_part += h0_tfce_part[:, i_perm].reshape(
                (-1, 1)
            ) < np.fabs(tfce_original_data.T)

        if threshold is not None:
            (
                h0_csfwe_part[:, i_perm],
                h0_cmfwe_part[:, i_perm],
            ) = _calculate_cluster_measures(
                arr4d,
                threshold,
                bin_struct=_binary_structure(),
                two_sided_test=two_sided_test,
            )

        if verbose > 0:
            step = 11 - min(verbose, 10)
            if i_perm % step == 0:
                # If there is only one job, progress information is fixed
                crlf = "\n"
                if n_perm == n_perm_chunk:
                    crlf = "\r"

                percent = float(i_perm) / n_perm_chunk
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                remaining = (100.0 - percent) / max(0.01, percent) * dt
                sys.stderr.write(
                    f"Job #{thread_id}, processed {i_perm}/{n_perm_chunk} "
                    f"permutations ({percent:0.2f}%, {remaining} seconds "
                    f"remaining){crlf}"
                )

    return (
        scores_as_ranks_part,
        h0_fmax_part,
        h0_csfwe_part,
        h0_cmfwe_part,
        tfce_scores_as_ranks_part,
        h0_tfce_part,
    )


def _binary_structure():
    return generate_binary_structure(3, 1)


def permuted_ols(
    tested_vars,
    target_vars,
    confounding_vars=None,
    model_intercept=True,
    n_perm=10000,
    two_sided_test=True,
    random_state=None,
    n_jobs=1,
    verbose=0,
    masker=None,
    tfce=False,
    threshold=None,
    output_type="legacy",
):
    """Massively univariate group analysis with permuted OLS.

    Tested variates are independently fitted to target variates descriptors
    (e.g. brain imaging signal) according to a linear model solved with an
    Ordinary Least Squares criterion.
    Confounding variates may be included in the model.
    Permutation testing is used to assess the significance of the relationship
    between the tested variates and the target variates
    :footcite:p:`Anderson2001`, :footcite:p:`Winkler2014`.
    A max-type procedure is used to obtain family-wise corrected p-values
    based on t-statistics (voxel-level FWE), cluster sizes, cluster masses,
    and TFCE values.

    The specific permutation scheme implemented here is the one of
    :footcite:t:`Freedman1983`.
    Its has been demonstrated in :footcite:t:`Anderson2001` that
    this scheme conveys more sensitivity than alternative schemes. This holds
    for neuroimaging applications, as discussed in details in
    :footcite:t:`Winkler2014`.

    Permutations are performed on parallel computing units.
    Each of them performs a fraction of permutations on the whole dataset.
    Thus, the max t-score amongst data descriptors can be computed directly,
    which avoids storing all the computed t-scores.

    The variates should be given C-contiguous.
    ``target_vars`` are fortran-ordered automatically to speed-up computations.

    Parameters
    ----------
    tested_vars : array-like, shape=(n_samples, n_regressors)
        Explanatory variates, fitted and tested independently from each others.

    target_vars : array-like, shape=(n_samples, n_descriptors)
        fMRI data to analyze according to the explanatory and confounding
        variates.

        In a group-level analysis, the samples will typically be voxels
        (for volumetric data) or vertices (for surface data), while the
        descriptors will generally be images, such as run-wise z-statistic
        maps.

    confounding_vars : array-like, shape=(n_samples, n_covars), optional
        Confounding variates (covariates), fitted but not tested.
        If None, no confounding variate is added to the model
        (except maybe a constant column according to the value of
        ``model_intercept``).

    model_intercept : :obj:`bool`, optional
        If True, a constant column is added to the confounding variates
        unless the tested variate is already the intercept or when
        confounding variates already contain an intercept.
        Default=True.

    n_perm : :obj:`int`, optional
        Number of permutations to perform.
        Permutations are costly but the more are performed, the more precision
        one gets in the p-values estimation.
        If ``n_perm`` is set to 0, then no p-values will be estimated.
        Default=10000.

    two_sided_test : :obj:`bool`, optional
        If True, performs an unsigned t-test. Both positive and negative
        effects are considered; the null hypothesis is that the effect is zero.
        If False, only positive effects are considered as relevant. The null
        hypothesis is that the effect is zero or negative. Default=True.

    random_state : :obj:`int` or None, optional
        Seed for random number generator, to have the same permutations
        in each computing units.

    n_jobs : :obj:`int`, optional
        Number of parallel workers.
        If -1 is provided, all CPUs are used.
        A negative number indicates that all the CPUs except (abs(n_jobs) - 1)
        ones will be used. Default=1.

    verbose : :obj:`int`, optional
        verbosity level (0 means no message). Default=0.

    masker : None or :class:`~nilearn.maskers.NiftiMasker` or \
            :class:`~nilearn.maskers.MultiNiftiMasker`, optional
        A mask to be used on the data.
        This is required for cluster-level inference, so it must be provided
        if ``threshold`` is not None.

        .. versionadded:: 0.9.2

    threshold : None or :obj:`float`, optional
        Cluster-forming threshold in p-scale.
        This is only used for cluster-level inference.
        If None, cluster-level inference will not be performed.
        Default=None.

        .. warning::

            Performing cluster-level inference will increase the computation
            time of the permutation procedure.

        .. versionadded:: 0.9.2

    tfce : :obj:`bool`, optional
        Whether to calculate :term:`TFCE` as part of the permutation procedure
        or not.
        The TFCE calculation is implemented as described in
        :footcite:t:`Smith2009a`.
        Default=False.

        .. warning::

            Performing TFCE-based inference will increase the computation
            time of the permutation procedure considerably.
            The permutations may take multiple hours, depending on how many
            permutations are requested and how many jobs are performed in
            parallel.

        .. versionadded:: 0.9.2

    output_type : {'legacy', 'dict'}, optional
        Determines how outputs should be returned.
        The two options are:

        -   'legacy': return a pvals, score_orig_data, and h0_fmax.
            This option is the default, but it is deprecated until 0.13,
            when the default will be changed to 'dict'.
            It will be removed in 0.15.
        -   'dict': return a dictionary containing output arrays.
            This option will be made the default in 0.13.
            Additionally, if ``tfce`` is True or ``threshold`` is not None,
            ``output_type`` will automatically be set to 'dict'.

        .. deprecated:: 0.9.2

            The default value for this parameter will change from 'legacy' to
            'dict' in 0.13, and the parameter will be removed completely in
            0.15.

        .. versionadded:: 0.9.2

    Returns
    -------
    pvals : array-like, shape=(n_regressors, n_descriptors)
        Negative log10 p-values associated with the significance test of the
        n_regressors explanatory variates against the n_descriptors target
        variates. Family-wise corrected p-values.

        .. note::

            This is returned if ``output_type`` == 'legacy'.

        .. deprecated:: 0.9.2

            The 'legacy' option for ``output_type`` is deprecated.
            The default value will change to 'dict' in 0.13,
            and the ``output_type`` parameter will be removed in 0.15.

    score_orig_data : numpy.ndarray, shape=(n_regressors, n_descriptors)
        t-statistic associated with the significance test of the n_regressors
        explanatory variates against the n_descriptors target variates.
        The ranks of the scores into the h0 distribution correspond to the
        p-values.

        .. note::

            This is returned if ``output_type`` == 'legacy'.

        .. deprecated:: 0.9.2

            The 'legacy' option for ``output_type`` is deprecated.
            The default value will change to 'dict' in 0.13,
            and the ``output_type`` parameter will be removed in 0.15.

    h0_fmax : array-like, shape=(n_regressors, n_perm)
        Distribution of the (max) t-statistic under the null hypothesis
        (obtained from the permutations). Array is sorted.

        .. note::

            This is returned if ``output_type`` == 'legacy'.

        .. deprecated:: 0.9.2

            The 'legacy' option for ``output_type`` is deprecated.
            The default value will change to 'dict' in 0.13,
            and the ``output_type`` parameter will be removed in 0.15.

        .. versionchanged:: 0.9.2

            Return H0 for all regressors, instead of only the first one.

    outputs : :obj:`dict`
        Output arrays, organized in a dictionary.

        .. note::

            This is returned if ``output_type`` == 'dict'.
            This will be the default output starting in version 0.13.

        .. versionadded:: 0.9.2

        Here are the keys:

        ============= ============== ==========================================
        key           shape          description
        ============= ============== ==========================================
        t             (n_regressors, t-statistic associated with the
                      n_descriptors) significance test of the n_regressors
                                     explanatory variates against the
                                     n_descriptors target variates.
                                     The ranks of the scores into the h0
                                     distribution correspond to the p-values.
        logp_max_t    (n_regressors, Negative log10 p-values associated with
                      n_descriptors) the significance test of the n_regressors
                                     explanatory variates against the
                                     n_descriptors target variates.
                                     Family-wise corrected p-values, based on
                                     ``h0_max_t``.
        h0_max_t      (n_regressors, Distribution of the max t-statistic under
                      n_perm)        the null hypothesis (obtained from the
                                     permutations). Array is sorted.
        tfce          (n_regressors, TFCE values associated with the
                      n_descriptors) significance test of the n_regressors
                                     explanatory variates against the
                                     n_descriptors target variates.
                                     The ranks of the scores into the h0
                                     distribution correspond to the TFCE
                                     p-values.
        logp_max_tfce (n_regressors, Negative log10 p-values associated with
                      n_descriptors) the significance test of the n_regressors
                                     explanatory variates against the
                                     n_descriptors target variates.
                                     Family-wise corrected p-values, based on
                                     ``h0_max_tfce``.

                                     Returned only if ``tfce`` is True.
        h0_max_tfce   (n_regressors, Distribution of the max TFCE value under
                      n_perm)        the null hypothesis (obtained from the
                                     permutations). Array is sorted.

                                     Returned only if ``tfce`` is True.
        size          (n_regressors, Cluster size values associated with the
                      n_descriptors) significance test of the n_regressors
                                     explanatory variates against the
                                     n_descriptors target variates.
                                     The ranks of the scores into the h0
                                     distribution correspond to the size
                                     p-values.

                                     Returned only if ``threshold`` is not
                                     None.
        logp_max_size (n_regressors, Negative log10 p-values associated with
                      n_descriptors) the cluster-level significance test of
                                     the n_regressors explanatory variates
                                     against the n_descriptors target
                                     variates.
                                     Family-wise corrected, cluster-level
                                     p-values, based on ``h0_max_size``.

                                     Returned only if ``threshold`` is not
                                     None.
        h0_max_size   (n_regressors, Distribution of the max cluster size
                      n_perm)        value under the null hypothesis (obtained
                                     from the permutations). Array is sorted.

                                     Returned only if ``threshold`` is not
                                     None.
        mass          (n_regressors, Cluster mass values associated with the
                      n_descriptors) significance test of the n_regressors
                                     explanatory variates against the
                                     n_descriptors target variates.
                                     The ranks of the scores into the h0
                                     distribution correspond to the mass
                                     p-values.

                                     Returned only if ``threshold`` is not
                                     None.
        logp_max_mass (n_regressors, Negative log10 p-values associated with
                      n_descriptors) the cluster-level significance test of
                                     the n_regressors explanatory variates
                                     against the n_descriptors target
                                     variates.
                                     Family-wise corrected, cluster-level
                                     p-values, based on ``h0_max_mass``.

                                     Returned only if ``threshold`` is not
                                     None.
        h0_max_mass   (n_regressors, Distribution of the max cluster mass
                      n_perm)        value under the null hypothesis (obtained
                                     from the permutations). Array is sorted.

                                     Returned only if ``threshold`` is not
                                     None.
        ============= ============== ==========================================

    References
    ----------
    .. footbibliography::

    """
    output_type, n_jobs, tested_vars = _check_args_permuted_ols(
        target_vars,
        tested_vars,
        confounding_vars,
        n_jobs,
        output_type,
        tfce,
        threshold,
        masker,
    )

    # make target_vars F-ordered to speed-up computation
    n_descriptors = target_vars.shape[1]
    target_vars = np.asfortranarray(target_vars)  # efficient for chunking
    if np.any(np.all(target_vars == 0, axis=0)):
        warnings.warn(
            "Some descriptors in 'target_vars' have zeros across all samples. "
            "These descriptors will be ignored during null distribution "
            "generation."
        )

    tested_var_has_intercept = _check_for_intercept_in_tested_var(tested_vars)

    (
        confounding_vars,
        intercept_test,
        model_intercept,
    ) = _check_for_intercept_in_confounds(
        confounding_vars, tested_var_has_intercept, model_intercept
    )

    confounding_vars = _add_intercept_optionally(
        confounding_vars,
        model_intercept,
        intercept_test,
        n_samples=tested_vars.shape[0],
    )

    # OLS regression on original data
    # original regression (= regression on residuals + adjust t-score)
    # compute t score map of each tested var for original data
    # scores_original_data is in samples-by-regressors shape
    confounding_vars = _orthonormalize_confounding_vars(confounding_vars)
    target_vars = _prepare_target_vars(target_vars, confounding_vars)
    tested_vars = _prepare_tested_vars(tested_vars, confounding_vars)
    scores_original_data = _t_score_with_covars_and_normalized_design(
        tested_vars,
        target_vars.T,
        confounding_vars,
    )

    # Define connectivity for TFCE and/or cluster measures
    tfce_original_data = _get_tfce_original_data(
        tfce, masker, scores_original_data, two_sided_test
    )

    # 0 or negative number of permutations => original data scores only
    if n_perm <= 0:
        if output_type == "legacy":
            return np.asarray([]), scores_original_data.T, np.asarray([])

        out = {"t": scores_original_data.T}

        if tfce:
            out["tfce"] = tfce_original_data.T

        return out

    # Permutations
    # parallel computing units perform a reduced number of permutations each
    return _run_permutations(
        tested_vars,
        target_vars,
        confounding_vars,
        n_perm,
        random_state,
        two_sided_test,
        n_jobs,
        verbose,
        masker,
        tfce,
        threshold,
        output_type,
        n_descriptors,
        intercept_test,
        scores_original_data,
        tfce_original_data,
    )


def _check_args_permuted_ols(
    target_vars,
    tested_vars,
    confounding_vars,
    n_jobs,
    output_type,
    tfce,
    threshold,
    masker,
):
    """Validate input arguments for permuted_ols.

    Parameters
    ----------
    target_vars : array-like, shape=(n_samples, n_descriptors)
        fMRI data to analyze according to the explanatory and confounding
        variates.

    n_jobs : :obj:`int`
        Number of parallel workers.
        If -1 is provided, all CPUs are used.
        A negative number indicates that all the CPUs except (abs(n_jobs) - 1)

    output_type : {'legacy', 'dict'}, optional
        Determines how outputs should be returned.

    tfce : :obj:`bool`
        Whether to calculate :term:`TFCE` as part of the permutation procedure
        or not.
        The TFCE calculation is implemented as described in
        :footcite:t:`Smith2009a`.

    threshold : None or :obj:`float`
        Cluster-forming threshold in p-scale.
        This is only used for cluster-level inference.
        If None, cluster-level inference will not be performed.

    masker : None or :class:`~nilearn.maskers.NiftiMasker` or \
            :class:`~nilearn.maskers.MultiNiftiMasker`
        A mask to be used on the data.
        This is required for cluster-level inference, so it must be provided
        if ``threshold`` is not None.

    """
    if target_vars.ndim != 2:
        raise ValueError(
            "'target_vars' should be a 2D array. "
            f"An array with {target_vars.ndim} dimension(s) was passed."
        )

    if n_jobs == 0:  # invalid according to joblib's conventions
        raise ValueError(
            "'n_jobs == 0' is not a valid choice. "
            "Please provide a positive number of CPUs, or -1 for all CPUs, "
            "or a negative number (-i) for 'all but (i-1)' CPUs "
            "(joblib conventions)."
        )

    # check n_jobs (number of CPUs)
    if n_jobs < 0:
        n_jobs = max(1, joblib.cpu_count() - int(n_jobs) + 1)
    else:
        n_jobs = min(n_jobs, joblib.cpu_count())

    # check that masker is provided if it is needed
    if tfce and not masker:
        raise ValueError("A masker must be provided if tfce is True.")

    if (threshold is not None) and (masker is None):
        raise ValueError(
            'If "threshold" is not None, masker must be defined as well.'
        )

    output_type = _check_output_type_permuted_ols(output_type, tfce, threshold)

    # check explanatory variates' dimensions
    if tested_vars.ndim == 1:
        tested_vars = np.atleast_2d(tested_vars).T

    # check n_sample consistent across arguments
    assert target_vars.shape[0] == tested_vars.shape[0]
    if confounding_vars is not None:
        assert target_vars.shape[0] == confounding_vars.shape[0]

    return output_type, n_jobs, tested_vars


def _check_output_type_permuted_ols(output_type, tfce, threshold):
    """Check that the output type is valid \
    for the given threshold argument and depending if we are doing TFCE.

    output_type may be updated when necessary.

    Parameters
    ----------
    output_type : {'legacy', 'dict'}, optional
        Determines how outputs should be returned.

    tfce : :obj:`bool`
        Whether to calculate :term:`TFCE` as part of the permutation procedure
        or not.
        The TFCE calculation is implemented as described in
        :footcite:t:`Smith2009a`.

    threshold : None or :obj:`float`
        Cluster-forming threshold in p-scale.
        This is only used for cluster-level inference.
        If None, cluster-level inference will not be performed.
    """
    if tfce and output_type == "legacy":
        warnings.warn(
            'If "tfce" is set to True, "output_type" must be set to "dict". '
            "Overriding."
        )
        output_type = "dict"

    if (threshold is not None) and (output_type == "legacy"):
        warnings.warn(
            'If "threshold" is not None, "output_type" must be set to "dict". '
            "Overriding."
        )
        output_type = "dict"

    if output_type == "legacy":
        warnings.warn(
            category=DeprecationWarning,
            message=(
                'The "legacy" output structure for "permuted_ols" is '
                "deprecated. "
                'The default output structure will be changed to "dict" '
                "in version 0.13."
            ),
            stacklevel=3,
        )

    return output_type


def _check_for_intercept_in_tested_var(tested_vars):
    """Check if explanatory variates contain an intercept (constant) or not."""
    n_regressors = tested_vars.shape[1]
    return n_regressors == np.unique(tested_vars).size == 1


def _check_for_intercept_in_confounds(
    confounding_vars, tested_var_has_intercept, model_intercept
):
    """Check if confounding variates contain an intercept (constant) or not.

    If the tested var has an intercept, only this one is kept
    and all other intercept are removed from the confounds.

    Parameters
    ----------
    confounding_vars : array-like, shape=(n_samples, n_covars)
        Confounding variates (covariates), fitted but not tested.

    tested_var_has_intercept : :obj:`bool`
        Whether the tested variate already contains an intercept or not.

    model_intercept : :obj:`bool`
        If True, a constant column will be added to the confounding variates.
        This value may be updated if all intercepts were removed.

    """
    if confounding_vars is None:
        test_intercept = tested_var_has_intercept
        return confounding_vars, test_intercept, model_intercept

    # Search for all constant columns
    constants = [
        column
        for column in range(confounding_vars.shape[1])
        if np.unique(confounding_vars[:, column]).size == 1
    ]
    # check if multiple intercepts are defined across all variates
    if (tested_var_has_intercept and len(constants) == 1) or len(
        constants
    ) > 1:
        # remove all constant columns
        confounding_vars = np.delete(confounding_vars, constants, axis=1)
        # warn user if multiple intercepts are found
        warnings.warn(
            category=UserWarning,
            message=(
                "Multiple columns across 'confounding_vars' and/or "
                "'target_vars' are constant.\n"
                "Only one will be used as intercept."
            ),
        )
        # In case tested vars has no intercept,
        # and len(constants) > 1,
        # we will need to add an intercept (see _add_intercept_optionally)
        # because by now have removed all intercepts from confounding_vars.
        model_intercept = True

        # Remove confounding vars variable if it is empty
        if confounding_vars.size == 0:
            confounding_vars = None

    # intercept is only defined in confounding vars
    if not tested_var_has_intercept and len(constants) == 1:
        tested_var_has_intercept = True

    test_intercept = tested_var_has_intercept
    return confounding_vars, test_intercept, model_intercept


def _add_intercept_optionally(
    confounding_vars, model_intercept, intercept_test, n_samples
):
    """Add intercept to confounding variates if necessary.

    May be done because:
    - requested by the user or may be necessary with model_intercept=True
    - all intercepts have been removed from the confounds
      by _check_for_intercept_in_confounds and at least one must be added.

    Parameters
    ----------
    confounding_vars : array-like, shape=(n_samples, n_covars)
        Confounding variates (covariates), fitted but not tested.

    model_intercept : :obj:`bool`
        If True, a constant column will be added to the confounding variates.
        This value may be updated if all intercepts were removed.

    intercept_test : :obj:`bool`
        Whether an intercept is already present.

    n_samples : :obj:`int`
        number of samples
    """
    if model_intercept and not intercept_test:
        if confounding_vars is not None:
            confounding_vars = np.hstack(
                (confounding_vars, np.ones((n_samples, 1)))
            )
        else:
            confounding_vars = np.ones((n_samples, 1))
    return confounding_vars


def _orthonormalize_confounding_vars(confounding_vars):
    if confounding_vars is None:
        return None

    covars_orthonormalized = _orthonormalize_matrix(confounding_vars)
    return _make_array_contiguous(
        covars_orthonormalized, variable_name="Confounding"
    )


def _make_array_contiguous(array, variable_name):
    """Ensure array is contiguous for the sake of code efficiency."""
    if not array.flags["C_CONTIGUOUS"]:
        # useful to developer
        warnings.warn(f"{variable_name} variates not C_CONTIGUOUS.")
        array = np.ascontiguousarray(array)
    return array


def _prepare_target_vars(target_vars, confounding_vars):
    """Prepare target variables for OLS.

    Normalize target variables and remove effect of confounding variables
    if there are any.
    """
    target_vars_normalized = _normalize_matrix_on_axis(target_vars).T
    target_vars_normalized = _make_array_contiguous(
        target_vars_normalized, variable_name="Target"
    )

    if confounding_vars is None:
        return target_vars_normalized

    target_vars_resid_covars = _remove_effect_confounds(
        target_vars_normalized, confounding_vars
    )
    target_vars_resid_covars = _normalize_matrix_on_axis(
        target_vars_resid_covars, axis=1
    )
    return _make_array_contiguous(
        target_vars_resid_covars, variable_name="Target"
    )


def _remove_effect_confounds(array, confounding_vars):
    """Remove effect of confounding variables from an array."""
    beta_covars = np.dot(array, confounding_vars)
    array = array - np.dot(beta_covars, confounding_vars.T)
    return array


def _prepare_tested_vars(tested_vars, confounding_vars):
    """Prepare tested variables for OLS.

    Normalize tested variables and remove effect of confounding variables
    if there are any.
    """
    tested_vars_resid_covars = _normalize_matrix_on_axis(tested_vars).copy()

    if confounding_vars is None:
        return _make_array_contiguous(
            tested_vars_resid_covars, variable_name="Tested"
        )

    tested_vars_normalized = _normalize_matrix_on_axis(tested_vars.T, axis=1)
    tested_vars_resid_covars = _remove_effect_confounds(
        tested_vars_normalized, confounding_vars
    )
    tested_vars_resid_covars = _normalize_matrix_on_axis(
        tested_vars_resid_covars, axis=1
    ).T.copy()
    return _make_array_contiguous(
        tested_vars_resid_covars, variable_name="Tested"
    )


def _get_tfce_original_data(
    tfce, masker, scores_original_data, two_sided_test
):
    if not tfce:
        return None

    scores_4d = masker.inverse_transform(scores_original_data.T).get_fdata()
    tfce_original_data = _calculate_tfce(
        scores_4d,
        bin_struct=_binary_structure(),
        two_sided_test=two_sided_test,
    )
    tfce_original_data = apply_mask(
        nib.Nifti1Image(
            tfce_original_data,
            masker.mask_img_.affine,
            masker.mask_img_.header,
        ),
        masker.mask_img_,
    ).T

    return tfce_original_data


def _run_permutations(
    tested_vars,
    target_vars,
    confounding_vars,
    n_perm,
    random_state,
    two_sided_test,
    n_jobs,
    verbose,
    masker,
    tfce,
    threshold,
    output_type,
    n_descriptors,
    intercept_test,
    scores_original_data,
    tfce_original_data,
):
    if n_perm > n_jobs:
        n_perm_chunks = np.asarray([n_perm / n_jobs] * n_jobs, dtype=int)
        n_perm_chunks[-1] += n_perm % n_jobs
    elif n_perm > 0:
        warnings.warn(
            f"The specified number of permutations is {n_perm} and the number "
            f"of jobs to be performed in parallel has set to {n_jobs}. "
            f"This is incompatible so only {n_perm} jobs will be running. "
            "You may want to perform more permutations in order to take the "
            "most of the available computing resources."
        )
        n_perm_chunks = np.ones(n_perm, dtype=int)

    threshold_t = _determine_t_statistic_threshold(
        threshold, tested_vars, confounding_vars, two_sided_test
    )

    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    # actual permutations
    # seeded from a random integer
    # between 0 and maximum value represented by np.int32
    # (to have a large entropy).
    ret = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
        joblib.delayed(_permuted_ols_on_chunk)(
            scores_original_data,
            tested_vars,
            target_vars.T,
            thread_id=thread_id + 1,
            threshold=threshold_t,
            confounding_vars=confounding_vars,
            masker=masker,
            n_perm=n_perm,
            n_perm_chunk=n_perm_chunk,
            intercept_test=intercept_test,
            two_sided_test=two_sided_test,
            tfce=tfce,
            tfce_original_data=tfce_original_data,
            random_state=rng.randint(1, np.iinfo(np.int32).max - 1),
            verbose=verbose,
        )
        for thread_id, n_perm_chunk in enumerate(n_perm_chunks)
    )

    # reduce results
    (
        vfwe_scores_as_ranks_parts,
        h0_vfwe_parts,
        csfwe_h0_parts,
        cmfwe_h0_parts,
        tfce_scores_as_ranks_parts,
        h0_tfce_parts,
    ) = zip(*ret)

    n_regressors = tested_vars.shape[1]

    # Voxel-level FWE
    vfwe_h0 = np.hstack(h0_vfwe_parts)
    vfwe_scores_as_ranks = np.zeros((n_regressors, n_descriptors))
    for scores_as_ranks_part in vfwe_scores_as_ranks_parts:
        vfwe_scores_as_ranks += scores_as_ranks_part

    vfwe_pvals = (n_perm + 1 - vfwe_scores_as_ranks) / float(1 + n_perm)

    if output_type == "legacy":
        return (-np.log10(vfwe_pvals), scores_original_data.T, vfwe_h0)

    outputs = {
        "t": scores_original_data.T,
        "logp_max_t": -np.log10(vfwe_pvals),
        "h0_max_t": vfwe_h0,
    }

    if tfce:
        # We can use the same approach for TFCE that we use for vFWE
        h0_tfcemax = np.hstack(h0_tfce_parts)
        tfce_scores_as_ranks = np.zeros((n_regressors, n_descriptors))
        for tfce_scores_as_ranks_part in tfce_scores_as_ranks_parts:
            tfce_scores_as_ranks += tfce_scores_as_ranks_part

        tfce_pvals = (n_perm + 1 - tfce_scores_as_ranks) / float(1 + n_perm)
        neg_log10_tfce_pvals = -np.log10(tfce_pvals)

        outputs["tfce"] = tfce_original_data.T
        outputs["logp_max_tfce"] = neg_log10_tfce_pvals
        outputs["h0_max_tfce"] = h0_tfcemax

    if threshold is not None:
        cluster_stats = _compute_cluster_level_statistics(
            csfwe_h0_parts,
            cmfwe_h0_parts,
            vfwe_pvals,
            masker,
            scores_original_data,
            n_regressors,
            threshold_t,
            two_sided_test,
        )

        outputs["size"] = cluster_stats["size"]
        outputs["logp_max_size"] = -np.log10(cluster_stats["size_pvals"])
        outputs["h0_max_size"] = cluster_stats["size_h0"]
        outputs["mass"] = cluster_stats["mass"]
        outputs["logp_max_mass"] = -np.log10(cluster_stats["mass_pvals"])
        outputs["h0_max_mass"] = cluster_stats["mass_h0"]

    return outputs


def _determine_t_statistic_threshold(
    threshold, tested_vars, confounding_vars, two_sided_test
):
    """Return t-statistic threshold for cluster-level inference.

    Parameters
    ----------
    threshold : None or :obj:`float`
        Cluster-forming threshold in p-scale.
        This is only used for cluster-level inference.
        If None, cluster-level inference will not be performed.

    tested_vars : array-like, shape=(n_samples, n_regressors)
        Explanatory variates, fitted and tested independently from each others.

    confounding_vars : array-like, shape=(n_samples, n_covars), optional
        Confounding variates (covariates), fitted but not tested.

    two_sided_test : :obj:`bool`
        If True, performs an unsigned t-test. Both positive and negative
        effects are considered.
        If False, only positive effects are considered as relevant.

    Returns
    -------
    float
        critical t-statistic threshold
    """
    if threshold is None:
        return None

    n_samples, n_regressors = tested_vars.shape
    n_covars = 0 if confounding_vars is None else confounding_vars.shape[1]
    df = n_samples - (n_regressors + n_covars)

    threshold = threshold / 2 if two_sided_test else threshold
    return stats.t.isf(threshold, df=df)


def _compute_cluster_level_statistics(
    csfwe_h0_parts,
    cmfwe_h0_parts,
    vfwe_pvals,
    masker,
    scores_original_data,
    n_regressors,
    threshold_t,
    two_sided_test,
):
    """Compute several cluster levels statistics.

    - size
    - cluster-mass
    - FWE
    """
    cluster_stats = {
        "size_h0": np.hstack(csfwe_h0_parts),
        "mass_h0": np.hstack(cmfwe_h0_parts),
        "size": np.zeros_like(vfwe_pvals).astype(int),
        "mass": np.zeros_like(vfwe_pvals),
        "size_pvals": np.zeros_like(vfwe_pvals),
        "mass_pvals": np.zeros_like(vfwe_pvals),
    }

    scores_original_data_4d = masker.inverse_transform(
        scores_original_data.T
    ).get_fdata()

    for i_regressor in range(n_regressors):
        scores_original_data_3d = scores_original_data_4d[..., i_regressor]

        labeled_arr3d = _label_clusters(
            scores_original_data_3d, threshold_t, two_sided_test
        )

        cluster_labels, idx, cluster_stats["size_regressor"] = np.unique(
            labeled_arr3d,
            return_inverse=True,
            return_counts=True,
        )
        assert cluster_labels[0] == 0  # the background

        # Replace background's "cluster size" w zeros
        cluster_stats["size_regressor"][0] = 0

        cluster_stats["mass_regressor"] = _compute_cluster_masses(
            scores_original_data_3d, cluster_labels, labeled_arr3d, threshold_t
        )

        # Calculate p-values from size/mass values and associated h0s
        for metric in ("mass", "size"):
            p_vals = _null_to_p(
                cluster_stats[f"{metric}_regressor"],
                cluster_stats[f"{metric}_h0"][i_regressor, :],
                "larger",
            )
            p_map = p_vals[np.reshape(idx, labeled_arr3d.shape)]
            metric_map = cluster_stats[f"{metric}_regressor"][
                np.reshape(idx, labeled_arr3d.shape)
            ]

            # Convert 3D to image, then to 1D
            # There is a problem if the masker performs preprocessing,
            # so we use apply_mask here.
            cluster_stats[f"{metric}_pvals"][i_regressor, :] = np.squeeze(
                apply_mask(
                    image.new_img_like(masker.mask_img_, p_map),
                    masker.mask_img_,
                )
            )
            cluster_stats[metric][i_regressor, :] = np.squeeze(
                apply_mask(
                    image.new_img_like(masker.mask_img_, metric_map),
                    masker.mask_img_,
                )
            )
    return cluster_stats


def _label_clusters(scores_original_data_3d, threshold_t, two_sided_test):
    """Label clusters for both cluster mass and size inference."""
    labeled_arr3d, _ = label(
        input=scores_original_data_3d > threshold_t,
        structure=_binary_structure(),
    )

    if not two_sided_test:
        return labeled_arr3d

    # Add negative cluster labels
    temp_labeled_arr3d, _ = label(
        input=scores_original_data_3d < -threshold_t,
        structure=_binary_structure(),
    )
    n_negative_clusters = np.max(temp_labeled_arr3d)
    labeled_arr3d[labeled_arr3d > 0] += n_negative_clusters
    labeled_arr3d = labeled_arr3d + temp_labeled_arr3d
    del temp_labeled_arr3d

    return labeled_arr3d


def _compute_cluster_masses(
    scores_original_data_3d, cluster_labels, labeled_arr3d, threshold_t
):
    mass_regressor = np.zeros(cluster_labels.shape)
    for i in cluster_labels[1:]:  # skip background
        cluster_mass = np.sum(
            np.fabs(scores_original_data_3d[labeled_arr3d == i]) - threshold_t
        )
        mass_regressor[i] = cluster_mass
    return mass_regressor
