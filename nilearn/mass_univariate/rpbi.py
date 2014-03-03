"""
Randomized Parcellation Based Inference

"""
# Authors: Virgile Fritsch <virgile.fritsch@inria.fr>, feb. 2014
#          Benoit Da Mota <damota.benoit@gmail.com>, jun. 2013
import warnings
import numpy as np
import scipy.sparse as sps
import sklearn.externals.joblib as joblib
from sklearn.utils import gen_even_slices

from nilearn.mass_univariate.permuted_least_squares import (
    _permuted_ols_on_chunk)

from parietal.group_analysis.rpbi import build_parcellations


class GrowableSparseArray(object):
    """Data structure to contain data from numerous estimations.

    Examples of application are all resampling schemes
    (bootstrap, permutations, ...)

    GrowableSparseArray can be seen as a three-dimensional array that contains
    scores associated with three position indices corresponding to
    (i) an iteration (or estimation), (ii) a test variate and
    (iii) a target variate.
    Memory is pre-allocated to store a large number of scores. The structure
    can be indexed efficiently according to three dimensions to add new
    scores at the right position fast.
    The allocated space can be extended if needed, but we want to avoid this
    because it is costly. User should carefully initialize the structure.
    Only scores above a predetermined threshold are actually stored, others
    are ignored.

    Attributes
    ----------
    n_elts: int
      The total number of scores actually stored into the data structure
    n_iter: int
      Number of trials (using as many iterators)
    max_elts: int
      Maximum number of scores that can be stored into the structure
    data: array-like, own-designed dtype
      The actual scores corresponding to all the estimations.
      dtype is built so that every score is associated with three position
      GrowableSparseArray can be seen as a three-dimensional array so every
      score is associated with three position indices corresponding to
      (i) an iteration (or an estimator) ('iter_id'),
      (ii) a test variate ('x_id') and
      (iii) a target variate ('y_id').
    sizes: array-like, shape=(n_iter, )
      The number of scores stored for each estimation.
      Useful to select a range of values from iteration ids.
    threshold: float,
      Sparsity threshold used to discard scores that are to low to have a
      chance to correspond to a maximum value amongst all the scores of
      a given iteration.

    """
    def __init__(self, n_iter=10000, n_elts=0, max_elts=None,
                 threshold=-np.inf):
        self.n_elts = n_elts
        self.n_iter = n_iter
        self.max_elts = max(max_elts, n_elts)
        self.data = np.empty(
            self.max_elts,
            dtype=[('iter_id', np.int32), ('x_id', np.int32),
                   ('y_id', np.int32), ('score', np.float32)])
        self.sizes = np.zeros((n_iter))
        self.threshold = threshold

    def get_data(self):
        return self.data[:self.n_elts]

    def merge(self, others):
        """Copy one or several GrowableSparseArray into the current structure.

        Parameters
        ----------
        others: list of GrowableSparseArray or GrowableSparseArray
          The structures to be merged into the current structure.

        """
        if isinstance(others, GrowableSparseArray):
            return self.merge([others])
        if not isinstance(others, list) and not isinstance(others, tuple):
            raise Exception(
                '\'others\' is not a list/tuple of GrowableSparseArray '
                'or a GrowableSparseArray.')
        for gs_array in others:
            if not isinstance(gs_array, GrowableSparseArray):
                raise Exception('List element is not a GrowableSparseArray.')
            if gs_array.n_iter != self.n_iter:
                raise Exception('Cannot merge a structure with %d iterations '
                                'into a structure with %d iterations.'
                                % (gs_array.n_iter, self.n_iter))

        acc_sizes = [self.sizes]
        acc_data = [self.get_data()]
        for gs_array in others:
            # threshold the data to respect self.threshold
            if gs_array.threshold < self.threshold:
                gs_array_data_thresholded = (
                    gs_array.get_data()[gs_array.get_data()['score']
                                       >= self.threshold])
                acc_sizes.append([gs_array_data_thresholded.size])
                acc_data.append(gs_array_data_thresholded)
            elif gs_array.threshold > self.threshold:
                warnings.warn('Merging a GrowableSparseArray into another '
                              'with a lower threshold: parent array may '
                              'contain less scores than its threshold '
                              'suggests.')
                acc_sizes.append(gs_array.sizes)
                acc_data.append(gs_array.get_data())
            else:
                acc_sizes.append(gs_array.sizes)
                acc_data.append(gs_array.get_data())

        self.sizes = np.array(acc_sizes).sum(axis=0)
        self.data = np.concatenate(acc_data)
        self.n_elts = self.sizes.sum()
        self.max_elts = self.n_elts
        self.data = np.sort(self.data, order=['iter_id', 'x_id', 'y_id'])

        return

    def append(self, iter_id, iter_data, y_offset=0):
        """Add the data of one estimation (iteration) into the structure.

        This is done in a memory-efficient way, by taking into account
        pre-allocated space.

        Parameters
        ----------
        iter_id: int,
          ID of the estimation we are inserting into the structure
        iter_data: array-like, shape=(n_targets_chunk, n_regressors)
          Scores corresponding to the iteration chunk to be inserted into
          the data structure.
        y_offset: int,
          Position of the target variates chunk relative to the original
          dataset.

        """
        # we only store float32 to save space
        iter_data = iter_data.astype('float32')
        # we sparsify the matrix wrt. threshold using coordinates list
        y_idx, x_idx = (iter_data >= self.threshold).nonzero()
        score_size = len(x_idx)
        new_n_elts = score_size + self.n_elts
        if (new_n_elts > self.max_elts or
            self.sizes[iter_id + 1:].sum() > 0):  # insertion (costly)
            new_data = np.empty(score_size,
                        dtype=[('iter_id', np.int32), ('x_id', np.int32),
                               ('y_id', np.int32), ('score', np.float32)])
            new_data['x_id'][:] = x_idx
            new_data['y_id'][:] = y_idx + y_offset
            new_data['score'][:] = iter_data[y_idx, x_idx]
            new_data['iter_id'][:] = iter_id
            gs_array = GrowableSparseArray(
                self.n_iter, threshold=self.threshold)
            gs_array.data = new_data
            gs_array.sizes = np.zeros((gs_array.n_iter))
            gs_array.sizes[iter_id] = score_size
            gs_array.n_elts = score_size
            gs_array.max_elts = score_size
            self.merge(gs_array)
        else:  # it fits --> updates (efficient)
            self.data['x_id'][self.n_elts:new_n_elts] = x_idx
            self.data['y_id'][self.n_elts:new_n_elts] = y_idx + y_offset
            self.data['score'][self.n_elts:new_n_elts] = (
                iter_data[y_idx, x_idx])
            self.data['iter_id'][self.n_elts:new_n_elts] = iter_id
            self.sizes[iter_id] += score_size
            self.n_elts = new_n_elts
        return


def max_csr(csr_mat, n_x):
    """Fast computation of the max along each row of a CSR matrix.

    Parameters
    ----------
    max_csr: scipy.sparse.csr matrix
      The matrix from which to compute each line's max.

    Returns
    -------
    res: array-like, shape=(n_rows, )
      Max value of each row of the input CSR matrix.
      Empty lines will have a 0 max.

    """
    res = np.zeros(csr_mat.shape[0] / n_x)
    if len(csr_mat.data) == 0:
        return res

    # We use csr_mat.indptr to adress csr_mat.data.
    # csr_mat.indptr gives the offset to adress each row of the matrix.
    # The complex syntax only aims at putting 0 as the max value
    # for empty lines.
    res[np.diff(csr_mat.indptr[::n_x]) != 0] = np.maximum.reduceat(
        csr_mat.data,
        csr_mat.indptr[:-1:n_x][np.diff(csr_mat.indptr[::n_x]) > 0])
    return res


def binarization(X):
    """

    """
    return (X > 0).astype(int)


def _inverse_transform(perm_lot_results, perm_lot_slice,
                       n_regressors, parcellation_masks, n_parcellations,
                       n_parcels_all_parcellations):
    """
    """
    n_perm_in_perm_lot = perm_lot_slice.stop - perm_lot_slice.start

    # Convert chunk results to a CSR matrix
    regressors_ids = (
        (perm_lot_results['iter_id'] - perm_lot_slice.start) * n_regressors
        + perm_lot_results['x_id'])
    perm_lot_as_csr = sps.csr_matrix(
        (perm_lot_results['score'],
         (regressors_ids, perm_lot_results['y_id'])),
        shape=(n_perm_in_perm_lot * n_regressors, n_parcels_all_parcellations))
    # counting statistic as a dot product (efficient between CSR x CSC)
    counting_statistic = np.dot(perm_lot_as_csr, parcellation_masks)

    # Get counting statistic for original data and construct (a part of) H0
    if perm_lot_slice.start == 0:  # perm 0 of perm_lot 0 contains orig scores
        original_scores = np.asarray(
            counting_statistic[:n_regressors].todense())
        h0_samples = max_csr(counting_statistic[n_regressors:], n_regressors)
    else:  # all perm_lots but no. 0 contain scores obtained under the null
        original_scores = []
        h0_samples = max_csr(counting_statistic, n_regressors)

    return original_scores, h0_samples


def rpbi_core(tested_vars, parcelled_imaging_vars,
              n_parcellations, parcellations_labels,
              confounding_vars=None, model_intercept=True, threshold=1e-04,
              n_perm=1000, random_state=0, n_jobs=0):
    """Run RPBI from parcelled data.

    This is the core method for Randomized Parcellation Based Inference.

    Parameters
    ----------
    tested_vars: array-like, shape=(n_samples, n_regressors),
      Explanatory variates, fitted and tested independently from each others.
    parcelled_imaging_vars: array-like, shape=(n_parcels_tot, n_samples)
      Average signal within parcels of all parcellations, for every subject.
    n_parcellations: int,
      Number of (randomized) parcellations.
    parcellations_labels: array-like, shape=(n_parcellations * n_voxels, )
      All parcellation's labels ("labels to voxels" map).
    n_parcels: list of int,
      Number of parcels for the parcellations.
    confounding_vars: array-like, shape=(n_samples, n_confounds)
      Confounding variates (covariates), fitted but not tested.
      If None (default), no confounding variate is added to the model
      (except maybe a constant column according to the value of
      `model_intercept`)
    model_intercept: bool,
      If True (default), a constant column is added to the confounding variates
      unless the tested variate is already the intercept.
    threshold: float, 0. < threshold < 1.,
      RPBI's threshold to discretize individual parcel-based analysis results.
    n_perm: int, n_perm > 1,
      Number of permutation to convert the counting statistic into p-values.
      The higher n_perm, the more precise the results, at the cost of
      computation time.
    random_state: int,
      Random numbers seed for reproducible results.
    n_jobs: int,
      Number of parallel workers. Default is 1.
      If 0 is provided, all CPUs are used.
      A negative number indicates that all the CPUs except (|n_jobs| - 1) ones
      must be used.

    """
    if n_jobs == 0:
        n_jobs = joblib.cpu_count()
    elif n_jobs < 0:
        n_jobs = max(1, joblib.cpu_count() - int(n_jobs) + 1)

    n_samples = parcelled_imaging_vars.shape[1]
    n_regressors = tested_vars.shape[1]
    n_voxels_all_parcellations = parcellations_labels.size
    n_voxels = n_voxels_all_parcellations / n_parcellations
    unique_labels_all_parcellations = np.unique(parcellations_labels)
    n_parcels_all_parcellations = len(unique_labels_all_parcellations)

    # check if explanatory variates is intercept (constant) or not
    if (n_regressors == 1 and np.unique(tested_vars).size == 1):
        intercept_test = True
    else:
        intercept_test = False

    # optionally add intercept
    if model_intercept and not intercept_test:
        if confounding_vars is not None:
            confounding_vars = np.hstack(
                (confounding_vars, np.ones((n_samples, 1))))
        else:
            confounding_vars = np.ones((n_samples, 1))

    ### Use permuted OLS to actually perform permutations of the RPBI analysis
    # (because parcelled_data contains data from all parcellations)
    # We use the _permuted_ols_on_chunk function as a way to obtain the scores
    # of all permutations and to avoid conversion of the scores into p-values
    # which is useless at this stage.
    ret = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_permuted_ols_on_chunk)
          (tested_vars, parcelled_imaging_vars[chunk], confounding_vars,
           n_perm, sparsity_threshold=threshold,
           random_state=random_state, target_vars_chunk_position=chunk.start,
           intercept_test=intercept_test)
          for chunk in gen_even_slices(n_perm + 2, min(n_perm + 1, n_jobs)))
    # reduce results
    all_chunks_results, params = zip(*ret)
    all_results = GrowableSparseArray(
        n_perm + 1,
        threshold=all_chunks_results[0].threshold)  # same threshold everywhere
    all_results.merge(all_chunks_results)
    # scores binarization (to be summed later to yield the counting statistic)
    all_results.data['score'] = binarization(all_results.get_data()['score'])

    ### Inverse transforms (map back masked voxels into a brain)
    # Build parcellations labels as masks.
    # we need a CSC sparse matrix for efficient computation. we can build
    # it efficiently using a COO sparse matrix constructor.
    voxel_ids = np.arange(n_voxels_all_parcellations) % n_voxels
    parcellation_masks = sps.coo_matrix(
        (np.ones(n_voxels_all_parcellations),
         (parcellations_labels, voxel_ids)),
        shape=(n_parcels_all_parcellations, n_voxels)).tocsc()

    # Slice permutations to treat them in parallel
    perm_lots_slices = gen_even_slices(n_perm + 2, min(n_perm + 1, n_jobs))
    perm_lots_sizes = [np.sum(all_results.sizes[s]) for s in perm_lots_slices]
    perm_lots_cuts = np.cumsum(perm_lots_sizes)
    perm_lots = [all_results[perm_lots_cuts[i]:perm_lots_cuts[i + 1]]
                 for i in xrange(perm_lots_cuts.size - 1)]
    ret = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_inverse_transform)
          (perm_lot, perm_lot_slice, n_regressors, parcellation_masks,
           n_parcellations, n_parcels_all_parcellations)
          for perm_lot, perm_lot_slice in zip(perm_lots, perm_lots_slices))
    # reduce results
    counting_stat_original_data, h0 = zip(*ret)
    counting_stat_original_data = np.ravel(counting_stat_original_data)
    h0 = np.sort(np.ravel(h0))

    # Convert H1 to neg. log. p-values
    p_values = - np.log10(
        (n_perm + 1 - np.searchsorted(h0, counting_stat_original_data))
        / float(n_perm + 1))
    p_values = p_values.reshape((n_regressors, -1))

    return p_values, counting_stat_original_data, h0


def randomized_parcellation_based_inference(
    tested_vars, imaging_vars, mask, confounding_vars=None,
    model_intercept=True, n_parcellations=100, n_parcels=1000,
    threshold=0.0001, n_perm=1000, random_state=0, n_jobs=0, verbose=True):
    """Perform Randomized Parcellation Base Inference on a dataset.

    1. Randomized parcellation are built.
    2. Statistical inference is performed.

    Parameters
    ----------
    tested_vars: array-like, shape=(n_subjs, n_test_vars),
      Explanatory variates, fitted and tested independently from each others.
    imaging_vars: array-like, shape=(n_samples, n_descriptors)
      Masked subject images as an array.
      Imaging data to be explained by explanatory and confounding variates.
    mask: image-like
      Mask image that has been used to mask data in `imaging_vars`.
    confounding_vars: array-like, shape=(n_samples, n_covars)
      Confounding variates (covariates), fitted but not tested.
      If None (default), no confounding variate is added to the model
      (except maybe a constant column according to the value of
      `model_intercept`)
    model_intercept: bool,
      If True (default), a constant column is added to the confounding variates
      unless the tested variate is already the intercept.
    n_parcellations: int,
      Number of (randomized) parcellations.
    n_parcels: list of int,
      Number of parcels for the parcellations.
    threshold: float, 0. < threshold < 1.,
      RPBI's threshold to discretize individual parcel-based analysis results.
    n_perm: int, n_perm > 1,
      Number of permutation to convert the counting statistic into p-values.
      The higher n_perm, the more precise the results, at the cost of
      computation time.
    random_state: int,
      Random numbers seed for reproducible results.
    n_jobs: int,
      Number of parallel workers. Default is 1.
      If 0 is provided, all CPUs are used.
      A negative number indicates that all the CPUs except (|n_jobs| - 1) ones
      must be used.
    verbose: boolean,
      Activate verbose mode (default is False).

    """
    # check explanatory variates dimensions
    if tested_vars.ndim == 1:
        tested_vars = np.atleast_2d(tested_vars).T

    ### Build parcellations
    if verbose:
        print "Build parcellations"
    parcelled_imaging_vars, parcellations_labels = build_parcellations(
        imaging_vars, mask,
        n_wards=n_parcellations, ward_sizes=[n_parcels],
        seed=random_state, n_jobs=n_jobs)

    ### Statistical inference
    if verbose:
        print "Statistical inference"
    counting_stat_original_data, h0 = rpbi_core(
        tested_vars, parcelled_imaging_vars,
        n_parcellations, parcellations_labels,
        confounding_vars=confounding_vars, model_intercept=model_intercept,
        threshold=threshold, n_perm=n_perm,
        random_state=random_state, n_jobs=n_jobs)

    return h0, counting_stat_original_data, parcelled_imaging_vars
