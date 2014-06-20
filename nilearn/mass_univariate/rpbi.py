"""
Randomized Parcellation Based Inference

"""
# Authors: Virgile Fritsch <virgile.fritsch@inria.fr>, feb. 2014
#          Benoit Da Mota, jun. 2013
import numpy as np
from scipy import stats, sparse
import sklearn.externals.joblib as joblib
from sklearn.externals.joblib import Memory
from sklearn.utils import gen_even_slices, check_random_state
from sklearn.preprocessing import binarize

from nilearn._utils import check_niimg, check_n_jobs
from nilearn._utils.cache_mixin import cache
from .utils import (
    orthogonalize_design, t_score_with_covars_and_normalized_design)


### GrowableSparseArray data structure ########################################
class GrowableSparseArray(object):
    """Sparse data array that supports memory-efficient, fast data appending.

    GrowableSparseArray is a sparse numpy recarray.
    It is similar to a scipy.sparse COO matrix, except that memory is
    pre-allocated from an estimation of the number of scores the array
    will eventually contain.  The allocated space can be extended if
    needed, but we want to avoid this because it is costly. User
    should carefully initialize the structure.

    The structure can be indexed efficiently according to two
    dimensions ('i' and 'j' axes) to add new data at the right
    position fast.

    Only non-zero scores are actually stored; others are
    ignored. However, this is the responsibility of the user to
    provide thresholded values to the GrowableSparseArray.

    GrowableSparseArrays support memory-efficient, fast data appending.
    Several GrowableSparseArrays can thus be merged efficiently.
    The only constraint is that they should have the same number of rows
    (i.e. their 'i' axis should have the same size).

    Attributes
    ----------
    n_elts : int
      The total number of values actually stored into the data structure

    n_rows : int
      Number of rows ('i' axis) of the GrowableSparseArray.
      Only arrays with the same value of `n_rows` can be merged together.

    max_elts : int
      Maximum number of values that can be stored into the structure.
      It can be changed after the initialization of the object, but that
      operation is costly and one should therefore carefully estimate
      correctly `max_elts` at initialization time.

    data : array-like, own-designed dtype
      The actual data contained in the structure.
      These can be indexed with three dimensions ('i' and 'j' axes).

    sizes : array-like, shape=(n_perm, )
      The number of data stored for each estimation.
      Useful to select a range of values from row ids.


    """
    def __init__(self, n_rows=10000, n_elts=0, max_elts=None):
        self.n_elts = n_elts
        self.n_rows = n_rows
        self.max_elts = max(max_elts, n_elts)
        self.data = np.empty(
            self.max_elts,
            dtype=[('i', np.int32), ('j', np.int32), ('data', np.float32)])
        self.sizes = np.zeros((n_rows))

    def get_data(self):
        return self.data[:self.n_elts]

    def merge(self, others):
        """Copy one or several GrowableSparseArray into the current structure.

        Parameters
        ----------
        others : list of GrowableSparseArray or GrowableSparseArray
          The structures to be merged into the current structure.

        """
        if isinstance(others, GrowableSparseArray):
            return self.merge([others])
        if not isinstance(others, list) and not isinstance(others, tuple):
            raise TypeError("The 'others' parameter should be a list or a "
                            "tuple of GrowableSparseArray, you gave %s."
                            % type(others))
        for gs_array in others:
            if not isinstance(gs_array, GrowableSparseArray):
                raise TypeError("List element is not a GrowableSparseArray.")
            if gs_array.n_rows != self.n_rows:
                raise ValueError("Cannot merge a structure with %d rows "
                                 "into a structure with %d rows."
                                 % (gs_array.n_rows, self.n_rows))

        acc_sizes = [self.sizes]
        acc_data = [self.get_data()]
        for gs_array in others:
            acc_sizes.append(gs_array.sizes)
            acc_data.append(gs_array.get_data())

        self.sizes = np.array(acc_sizes).sum(axis=0)
        self.data = np.concatenate(acc_data)
        self.n_elts = self.sizes.sum()
        self.max_elts = self.n_elts
        self.data = np.sort(self.data, order=['i', 'j'])

        return

    def append(self, row_id, row_data):
        """Add data corresponding to one row (dimension indexed by 'i').

        This is done in a memory-efficient way, by taking into account
        pre-allocated space.

        Parameters
        ----------
        row_id : int,
          Index of the row we are inserting into the structure.

        row_data : array-like, shape=(n_targets_chunk, )
          Data to be inserted into the data structure.

        """
        # store values as float32 to save space
        row_data = np.ravel(row_data.astype('float32'))
        # sparsify the matrix using coordinates list
        j_idx = row_data.nonzero()[0]
        score_size = len(j_idx)
        if score_size == 0:  # early return if nothing to add
            return

        new_n_elts = score_size + self.n_elts
        # check whether or not the current structure is big enough for the
        # new data to be appended without increasing `self.max_elts`.
        if (new_n_elts > self.max_elts or
            self.sizes[row_id + 1:].sum() > 0):  # insertion (costly)
            new_data = np.empty(score_size,
                                dtype=[('i', np.int32), ('j', np.int32),
                                       ('data', np.float32)])
            new_data['i'][:] = row_id
            new_data['j'][:] = j_idx
            new_data['data'][:] = row_data[j_idx]
            gs_array = GrowableSparseArray(self.n_rows)
            gs_array.data = new_data
            gs_array.sizes = np.zeros((gs_array.n_rows))
            gs_array.sizes[row_id] = score_size
            gs_array.n_elts = score_size
            gs_array.max_elts = score_size
            self.merge(gs_array)
        else:  # it fits --> updates (efficient)
            self.data['i'][self.n_elts:new_n_elts] = row_id
            self.data['j'][self.n_elts:new_n_elts] = j_idx
            self.data['data'][self.n_elts:new_n_elts] = row_data[j_idx]
            self.sizes[row_id] += score_size
            self.n_elts = new_n_elts
        return


### Parcellation building routines ############################################
from sklearn.feature_extraction import image
from sklearn.cluster import WardAgglomeration


def _ward_fit_transform(all_subjects_data, fit_samples_indices,
                        connectivity, n_parcels, offset_labels):
    """Ward clustering algorithm on a subsample and apply to the whole dataset.

    Computes a brain parcellation using Ward's clustering algorithm on some
    images, then averages the signal within parcels in order to reduce the
    dimension of the images of the whole dataset.
    This function is used with Randomized Parcellation Based Inference, so we
    need to save the labels to further perform the inverse transformation
    operation. The function therefore needs an offset to be applied on the
    labels so that they are unique across parcellations.

    Parameters
    ----------
    all_subjects_data : array_like, shape=(n_samples, n_voxels)
      Masked subject images as an array.

    fit_samples_indices : array-like,
      Indices of the samples used to compute the parcellation.

    connectivity : scipy.sparse.coo_matrix,
      Graph representing the spatial structure of the images (i.e. connections
      between voxels).

    n_parcels : int,
      Number of parcels for the parcellations.

    offset_labels : int,
      Offset for labels numbering.
      The purpose is to have different labels in all the parcellations that
      can be built by multiple calls to the current function.

    Returns
    -------
    parcelled_data : numpy.ndarray, shape=(n_samples, n_parcels)
      Average signal within each parcel for each subject.

    labels : np.ndarray, shape=(n_voxels,)
      Labels giving the correspondance between voxels and parcels.

    """
    # fit part
    data_fit = all_subjects_data[fit_samples_indices]
    ward = WardAgglomeration(n_clusters=n_parcels, connectivity=connectivity)
    ward.fit(data_fit)
    # transform part
    labels = ward.labels_ + offset_labels  # unique labels across parcellations
    parcelled_data = ward.transform(all_subjects_data)
    return parcelled_data, labels


def _build_parcellations(all_subjects_data, mask, n_parcellations=100,
                         n_parcels=1000, n_bootstrap_samples=None,
                         random_state=None, memory=Memory(cachedir=None),
                         n_jobs=1, verbose=False):
    """Build the parcellations for the RPBI framework.

    Parameters
    ----------
    all_subjects_data : array_like, shape=(n_samples, n_voxels)
      Masked subject images as an array.

    mask : ndarray of booleans
      Mask that has been applied on the initial images to obtain
      `all_subjects_data`.

    n_parcellations : int,
      The number of parcellations to be built and used to extract
      signal averages from the data.

    n_parcels : int,
      Number of parcels for the parcellations.

    n_bootstrap_samples : int,
      Number of subjects to be used to build the parcellations. The subjects
      are randomly drawn with replacement.
      If set to None, n_samples subjects are drawn, which correspond to
      a bootstrap draw.

    random_state : int,
      Random numbers seed for reproducible results.

    memory : instance of joblib.Memory or string
      Used to cache the masking process.
      By default, no caching is done. If a string is given, it is the
      path to the caching directory.

    n_jobs : int,
      Number of parallel workers.
      If 0 is provided, all CPUs are used.
      A negative number indicates that all the CPUs except (|n_jobs| - 1) ones
      will be used.

    verbose : boolean,
      Activate verbose mode (default is False).

    Returns
    -------
    parcelled_data : np.ndarray, shape=(n_parcels_tot, n_subjs)
      Data for all subjects after mean signal extraction with all the
      parcellations that have been created.

    ward_labels : np.ndarray, shape=(n_vox * n_wards, )
      Voxel-to-parcel map for all the parcellations. Useful to perform
      inverse transforms.

    TODO
    ----
    - Deal with NaNs in the original data (WardAgglomeration cannot fit
      when NaNs are present in the data). Median imputation?

    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    # check n_jobs (number of CPUs)
    n_jobs = check_n_jobs(n_jobs)

    n_samples = all_subjects_data.shape[0]
    if n_bootstrap_samples is None:
        n_bootstrap_samples = n_samples

    # Compute connectivity
    shape = mask.shape
    connectivity = image.grid_to_graph(
        n_x=shape[0], n_y=shape[1], n_z=shape[2], mask=mask)

    # Build parcellations
    draw = rng.randint(n_samples, size=n_bootstrap_samples * n_parcellations)
    draw = draw.reshape((n_parcellations, -1))
    ret = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(
            cache(_ward_fit_transform, memory, verbose=verbose))
          (all_subjects_data, draw[i], connectivity, n_parcels, i * n_parcels)
          for i in range(n_parcellations))
    # reduce results
    parcelled_data_parts, ward_labels = zip(*ret)
    parcelled_data = np.hstack((parcelled_data_parts))
    ward_labels = np.ravel(ward_labels)

    return parcelled_data, np.ravel(ward_labels)


### Routines for RPBI #########################################################
def max_csr(csr_matrix):
    """Fast computation of the maximum value along each row of a CSR matrix.

    Empty lines will have a maximum equal to 0.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr matrix
      The matrix from which to compute each line's maximum value.

    Returns
    -------
    res : array-like, shape=(n_rows, )
      Max value of each row of the input CSR matrix.
      Empty lines will have a maximum set to 0.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from nilearn.mass_univariate.rpbi import max_csr
    >>> max_csr(sparse.eye(5, 5).tocsr())
    array([ 1.,  1.,  1.,  1.,  1.])
    >>> max_csr(sparse.dia_matrix(np.diag(np.arange(6))).tocsr())
    array([ 0.,  1.,  2.,  3.,  4.,  5.])

    """
    res = np.zeros(csr_matrix.shape[0])
    if len(csr_matrix.data) == 0:
        return res

    # We use csr_mat.indptr to adress csr_mat.data.
    # csr_mat.indptr gives the offset to adress each row of the matrix.
    # The complex syntax only aims at putting 0s as the max value
    # for empty lines.
    res[np.diff(csr_matrix.indptr) != 0] = np.maximum.reduceat(
        csr_matrix.data,
        csr_matrix.indptr[:-1][np.diff(csr_matrix.indptr) > 0])
    return res


def _compute_counting_statistic_from_parcel_level_scores(
    perm_lot_results, perm_lot_slice, parcellation_masks,
    n_parcellations, n_parcels_all_parcellations):
    """Transform scores at the voxel level and sum them up across parcellations

    This function performs an inverse transform operation on
    parcel-level scores so as to put them back at the voxel level. The
    parcel-level scores may come from various parcellations (all
    values are concatenated along the second axis). In that case, the
    resulting voxel-level values are summed across parcellation to
    obtain the so-called 'counting statistic' that is used in
    Randomized Parcellation Based Inference.

    As for Randomized Parcellation Based Inference, various
    permutations can be processed simultaneously by the function
    (defined by the `perm_lot_slice` argument), yielding as many
    counting statistic values as needed for the permutation test. The
    statistics obtained for the original, non-permuted data can be
    separated from the "null hypothesis statistics" because they
    correspond to permutation no. 0.  This information is retrieved
    from the `perm_lot_slice` argument.

    Parameters
    ----------
    perm_lot_results : GrowableSparseArray,
      Scores obtained for the permutations considered in the bunch.
      Keep in mind that GrowableSparseArray structures only contain scores
      above a given threshold (to save memory).

    perm_lot_slice : slice object,
      Slice that defines the permutations for which the scores are being
      inverse transformed.

    parcellation_masks : scipy.sparse.csc_matrix, shape=(n_parcels, n_voxels)
      Correspondance between the 3D-image voxels and the labels of the
      parcels of every parcellation. The mapping is encoded as a
      sparse matrix containing binary values: "1" indicates a correspondance
      between a parcel (defined by the row number) and a voxel
      (defined by the column number).

    n_parcellations : int,
      Total number of parcellations.

    n_parcels_all_parcellations : int,
      Total number of parcels (summed across all parcellations).

    Returns
    -------
    counting_stats_original_data : np.ndarray, shape=(n_voxels,)
      Counting statistics corresponding to the original data.

    h0_samples : np.ndarray, shape=(n_perm,)
      Counting statistics corresponding to the permuted data.
      Yields the maximum count across all voxels for each permutation.

    """
    n_perm_in_perm_lot = perm_lot_slice.stop - perm_lot_slice.start

    # Convert chunk results to a CSR matrix
    regressors_ids = perm_lot_results['i'] - perm_lot_slice.start
    perm_lot_as_csr = sparse.csr_matrix(
        (perm_lot_results['data'],
         (regressors_ids, perm_lot_results['j'])),
        shape=(n_perm_in_perm_lot, n_parcels_all_parcellations))
    # counting statistic as a dot product (efficient between CSR x CSC)
    counting_statistic = perm_lot_as_csr.dot(parcellation_masks)

    # Get counting statistic for original data and construct (a part of) H0
    if perm_lot_slice.start == 0:  # perm 0 of perm_lot 0 contains orig scores
        counting_stats_original_data = np.asarray(
            counting_statistic[0].todense())
        h0_samples = max_csr(counting_statistic[1:])
    else:  # all perm_lot but no. 0 only contain scores obtained under the null
        counting_stats_original_data = []
        h0_samples = max_csr(counting_statistic)

    return np.ravel(counting_stats_original_data), h0_samples


def _univariate_analysis_on_chunk(n_perm, perm_chunk_start, perm_chunk_stop,
                                  tested_var, target_vars,
                                  confounding_vars=None, lost_dof=0,
                                  intercept_test=True, sparsity_threshold=0.1,
                                  random_state=None):
    """Perform part of the permutations of a massively univariate analysis.

    In the context of RPBI, we choose to split the chunks on permutations,
    i.e. each chunk will perform a part of the total amount of permutations
    but processes the whole data.
    This is the converse in the permuted OLS approach. The reason for this
    choice here is that we use the GrowableSparseArray structure, that is
    filled efficiently by appending the complete range of values corresponding
    to a given permutation in-a-row.

    Parameters
    ----------
    n_perm : int,
      The total number of permutations performed in the complete analysis.

    perm_chunk_start : int
      Together with `perm_chunk_stop`, defines the permutations that are
      delegated to the current function.
      `perm_chunk_start` also serves as an offset regarding
      the total number of permutations performed by parallel workers.

    perm_chunk_stop : int
      Together with `perm_chunk_start`, defines the permutations that
      are delegated to the current function.

    tested_var : array-like, shape=(n_samples, 1),
      Explanatory variate, fitted and tested.

    target_vars: array-like, shape=(n_samples, n_parcels_tot)
      Average signal within parcels of all parcellations, for every subject.

    confounding_vars : array-like, shape=(n_samples, n_covars)
      Confounding variates (covariates), fitted but not tested.
      If None (default), no confounding variate is added to the model
      (except maybe a constant column according to the value of
      `model_intercept`)

    lost_dof : int,
      Degress of freedom that are lost during the model estimation.
      Beware that tested variates are fitted independently so `lost_dof`
      only depends on confounding variates.

    intercept_test : boolean,
      Change the permutation scheme (swap signs for intercept,
      switch labels otherwise). See [1]

    sparsity_threshold : float, 0. < sparsity_threshold <= 1.
      Proportion of scores that will be actually stored in memory.
      The higher the threshold, the more scores will be stored,
      potentially resulting in memory issues. Conversely, a threshold
      that is too low can miss significant scores.

    random_state : int or None,
      Seed for random number generator, to have the same permutations
      in each computing units.

    Returns
    -------
    gs_array : GrowableSparseArray,
      Structure containing the thresholded scores corresponding to some
      permutations, defined by `perm_chunk`.

    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    n_samples, n_descriptors = target_vars.shape
    n_perm_chunk = perm_chunk_stop - perm_chunk_start

    # We use a special data structure to store the results of the permutations.
    sparsity_threshold_as_t_value = stats.t(n_samples - lost_dof - 1).isf(
        sparsity_threshold)
    # max_elts is used to preallocate memory
    # We expect the number of stored values to decrease non-linearly
    # with the sparsity_threshold value. Using a square-root is thus useful
    # to better estimate max_elts.
    max_elts = int(n_descriptors * n_perm_chunk * np.sqrt(sparsity_threshold))
    gs_array = GrowableSparseArray(n_perm + 1, max_elts=max_elts)

    if perm_chunk_start == 0:  # add original data results as permutation 0
        scores_original_data = t_score_with_covars_and_normalized_design(
            tested_var, target_vars, confounding_vars)
        # threshold scores according to sparsity_threshold
        scores_original_data[
            scores_original_data < sparsity_threshold_as_t_value] = 0
        gs_array.append(0, scores_original_data)
        perm_chunk_start = 1

    # do the permutations
    for i in xrange(perm_chunk_start, perm_chunk_stop):
        if intercept_test:
            # sign swap (random multiplication by 1 or -1)
            target_vars = (target_vars
                           * (rng.randint(2, size=(n_samples, 1)) * 2 - 1))
        else:
            # shuffle data
            # Regarding computation costs, we choose to shuffle testvars
            # and covars rather than fmri_signal.
            # Also, it is important to shuffle tested_var and covars
            # jointly to simplify f_score computation (null dot product).
            shuffle_idx = rng.permutation(n_samples)
            tested_var = tested_var[shuffle_idx]
            if confounding_vars is not None:
                confounding_vars = confounding_vars[shuffle_idx]

        # OLS regression on randomized data
        perm_scores = t_score_with_covars_and_normalized_design(
            tested_var, target_vars, confounding_vars)
        # threshold scores according to sparsity_threshold
        perm_scores[perm_scores < sparsity_threshold_as_t_value] = 0
        gs_array.append(i, perm_scores)

    return gs_array


def rpbi_core(tested_var, target_vars,
              n_parcellations, parcellations_labels, n_parcels,
              confounding_vars=None, model_intercept=True, threshold=None,
              n_perm=1000, random_state=None, n_jobs=1):
    """Run RPBI from parcelled data.

    This is the core method for Randomized Parcellation Based Inference.

    Parameters
    ----------
    tested_var : array-like, shape=(n_samples, 1),
      Explanatory variate, fitted and tested.

    target_vars : array-like, shape=(n_samples, n_parcels_tot)
      Average signal within parcels of all parcellations, for every subject.

    n_parcellations : int,
      Number of (randomized) parcellations.

    parcellations_labels : array-like, (n_parcellations * n_voxels,)
      All parcellation's labels ("labels to voxels" map).

    n_parcels : list of int,
      Number of parcels for the parcellations.

    confounding_vars : array-like, shape=(n_samples, n_confounds)
      Confounding variates (covariates), fitted but not tested.
      If None (default), no confounding variate is added to the model
      (except maybe a constant column according to the value of
      `model_intercept`)

    model_intercept : bool,
      If True (default), a constant column is added to the confounding variates
      unless the tested variate is already the intercept.

    threshold : float, 0. < threshold < 1.,
      RPBI's threshold to discretize individual parcel-based analysis results.

    n_perm : int, n_perm > 1,
      Number of permutation to convert the counting statistic into p-values.
      The higher n_perm, the more precise the results, at the cost of
      computation time.

    random_state : int,
      Random numbers seed for reproducible results.

    n_jobs : int,
      Number of parallel workers. Default is 1.
      If 0 is provided, all CPUs are used.
      A negative number indicates that all the CPUs except (|n_jobs| - 1) ones
      must be used.

    Returns
    -------
    p-values : np.ndarray, shape=(n_voxels,)
      Negative log10 p-values associated with the significance test of the
      explanatory variate against the target variate, assessed with
      Randomized Parcellation Based Inference.
      Family-wise corrected p-values (max-type procedure).

    counting_stats_original_data : np.ndarray, shape=(n_voxels,)
      Counting statistic (i.e. RPBI score) associated with original
      (non-permuted) data.

    h0 : np.ndarray, shape=(n_perm,)
      Maximum value of the counting statistic (i.e. RPBI score)
      across voxels obtained under each permutation of the original data.

    """
    # initialize the seed of the random generator
    rng = check_random_state(random_state)

    # check n_jobs (number of CPUs)
    n_jobs = check_n_jobs(n_jobs)

    # make target_vars F-ordered to speed-up computation
    if target_vars.ndim != 2:
        raise ValueError("'target_vars' should be a 2D array. "
                         "An array with %d dimension%s was passed"
                         % (target_vars.ndim,
                            "s" if target_vars.ndim > 1 else ""))
    target_vars = np.asfortranarray(target_vars)

    # check explanatory variates dimensions
    if tested_var.ndim == 1:
        tested_var = np.atleast_2d(tested_var).T
    n_samples = tested_var.shape[0]

    # check if explanatory variates is intercept (constant) or not
    if np.unique(tested_var).size == 1:
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

    # orthogonalize design to speed up subsequent permutations
    orthogonalized_design = orthogonalize_design(tested_var, target_vars,
                                                 confounding_vars)
    tested_var_resid_covars = orthogonalized_design[0]
    target_vars_resid_covars = orthogonalized_design[1]
    covars_orthonormed = orthogonalized_design[2]
    lost_dof = orthogonalized_design[3]

    # set RPBI threshold
    # In RPBI, only the scores for which the associated p-value is
    # below the threshold are considered (we use a F distribution as
    # an approximation of the scores distribution)
    if threshold == 'auto' or threshold is None:
        threshold = 0.1 / n_parcels  # Bonferroni correction for parcels

    ### Permutation of the RPBI analysis
    # parallel computing units perform a reduced number of permutations each
    perm_chunks = [(x.start, x.stop)
                   for x in gen_even_slices(n_perm + 1, min(n_perm, n_jobs))]
    all_chunks_results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_univariate_analysis_on_chunk)
        (n_perm, perm_chunk_start, perm_chunk_stop,
         tested_var_resid_covars, target_vars_resid_covars,
         covars_orthonormed, lost_dof, intercept_test=intercept_test,
         sparsity_threshold=threshold,
         random_state=rng.random_integers(np.iinfo(np.int32).max))
        for (perm_chunk_start, perm_chunk_stop) in perm_chunks)
    # reduce results (merge chunks in one big GrowableSparseArray)
    n_chunks = len(perm_chunks)
    max_elts_chunk = all_chunks_results[0].max_elts
    all_results = GrowableSparseArray(n_perm + 1,
                                      max_elts=max_elts_chunk * n_chunks)
    all_results.merge(all_chunks_results)
    # scores binarization (to be summed later to yield the counting statistic)
    all_results.data['data'] = binarize(all_results.get_data()['data'])

    ### Inverse transforms (map back masked voxels into a brain)
    n_voxels_all_parcellations = parcellations_labels.size
    n_voxels = n_voxels_all_parcellations / n_parcellations
    unique_labels_all_parcellations = np.unique(parcellations_labels)
    n_parcels_all_parcellations = len(unique_labels_all_parcellations)

    # build parcellations labels as masks.
    # we need a CSC sparse matrix for efficient computation. we can build
    # it efficiently using a COO sparse matrix constructor.
    voxel_ids = np.arange(n_voxels_all_parcellations) % n_voxels
    parcellation_masks = sparse.coo_matrix(
        (np.ones(n_voxels_all_parcellations),
         (parcellations_labels, voxel_ids)),
        shape=(n_parcels_all_parcellations, n_voxels),
        dtype=np.float32).tocsc()
    # slice permutations to treat them in parallel
    perm_lots_slices = [s for s in
                        gen_even_slices(n_perm + 1, min(n_perm, n_jobs))]
    perm_lots_sizes = [np.sum(all_results.sizes[s]) for s in perm_lots_slices]
    perm_lots_cuts = np.concatenate(([0], np.cumsum(perm_lots_sizes)))
    perm_lots = [
        all_results.get_data()[perm_lots_cuts[i]:perm_lots_cuts[i + 1]]
        for i in xrange(perm_lots_cuts.size - 1)]
    # put back parcel-based scores to voxel-level scale
    ret = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_compute_counting_statistic_from_parcel_level_scores)
        (perm_lot, perm_lot_slice, parcellation_masks,
         n_parcellations, n_parcels_all_parcellations)
        for perm_lot, perm_lot_slice in zip(perm_lots, perm_lots_slices))
    # reduce results
    counting_stats_original_data, h0 = zip(*ret)
    counting_stats_original_data = counting_stats_original_data[0]
    h0 = np.sort(np.concatenate(h0))

    ### Convert H1 to neg. log. p-values
    p_values = - np.log10(
        (n_perm + 1 - np.searchsorted(h0, counting_stats_original_data))
        / float(n_perm + 1))

    return p_values, counting_stats_original_data, h0


def randomized_parcellation_based_inference(
    tested_var, imaging_vars, mask_img, confounding_vars=None,
    model_intercept=True, n_parcellations=100, n_parcels=1000,
    threshold='auto', n_perm=1000, random_state=None,
    memory=Memory(cachedir=None), n_jobs=1, verbose=True):
    """Perform Randomized Parcellation Base Inference on a dataset.

    1. Randomized parcellations are built.
    2. Statistical inference is performed.

    Parameters
    ----------
    tested_var : array-like, shape=(n_subjs, 1),
      Explanatory variate, fitted and tested.

    imaging_vars : array-like, shape=(n_samples, n_descriptors)
      Masked subject images as an array.
      Imaging data to be explained by explanatory and confounding variates.

    mask_img : niimg
      Mask image that has been used to mask data in `imaging_vars`.

    confounding_vars : array-like, shape=(n_samples, n_covars)
      Confounding variates (covariates), fitted but not tested.
      If None (default), no confounding variate is added to the model
      (except maybe a constant column according to the value of
      `model_intercept`)

    model_intercept : bool,
      If True (default), a constant column is added to the confounding variates
      unless the tested variate is already the intercept.

    n_parcellations : int,
      Number of (randomized) parcellations.

    n_parcels : list of int,
      Number of parcels for the parcellations.

    threshold : float, 0. < threshold < 1.,
      RPBI's threshold to discretize individual parcel-based analysis results.

    n_perm : int, n_perm > 1,
      Number of permutation to convert the counting statistic into p-values.
      The higher n_perm, the more precise the results, at the cost of
      computation time.

    random_state : int,
      Random numbers seed for reproducible results.

    memory : instance of joblib.Memory or string
      Used to cache the masking process.
      By default, no caching is done. If a string is given, it is the
      path to the caching directory.

    n_jobs : int,
      Number of parallel workers. Default is 1.
      If 0 is provided, all CPUs are used.
      A negative number indicates that all the CPUs except (|n_jobs| - 1) ones
      must be used.

    verbose : boolean,
      Activate verbose mode (default is False).

    Returns
    -------
    neg_log_pvals : np.ndarray, shape=(n_voxels,)
      Negative log10 p-values associated with the significance test of the
      explanatory variate against the target variate, assessed with
      Randomized Parcellation Based Inference.
      Family-wise corrected p-values (max-type procedure).

    counting_stats_original_data : np.ndarray, shape=(n_voxels,)
      Counting statistic (i.e. RPBI score) associated with original
      (non-permuted) data.

    h0 : np.ndarray, shape=(n_perm,)
      Maximum value of the counting statistic (i.e. RPBI score)
      across voxels obtained under each permutation of the original data.

    """
    # check explanatory variates dimensions
    if tested_var.ndim == 1:
        tested_var = np.atleast_2d(tested_var).T

    ### Build parcellations
    if not isinstance(mask_img, np.ndarray):
        mask_img = check_niimg(mask_img)
    mask_img = np.asarray(mask_img).astype(bool)
    if verbose:
        print "Build parcellations"
    parcelled_imaging_vars, parcellations_labels = _build_parcellations(
        imaging_vars, mask_img,
        n_parcellations=n_parcellations, n_parcels=n_parcels,
        random_state=random_state, memory=memory, n_jobs=n_jobs,
        verbose=verbose)

    ### Statistical inference
    if verbose:
        print "Statistical inference"
    neg_log_pvals, counting_stats_original_data, h0 = rpbi_core(
        tested_var, parcelled_imaging_vars,
        n_parcellations, parcellations_labels, n_parcels,
        confounding_vars=confounding_vars, model_intercept=model_intercept,
        threshold=threshold, n_perm=n_perm,
        random_state=random_state, n_jobs=n_jobs)

    return neg_log_pvals, counting_stats_original_data, h0
