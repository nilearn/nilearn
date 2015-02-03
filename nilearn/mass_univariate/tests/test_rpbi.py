"""
Tests for functions used in Randomized Parcellation Based Inference.

"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Feb. 2014
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_raises)
from sklearn.utils import check_random_state
from nilearn.mass_univariate.rpbi import (
    GrowableSparseArray, _ward_fit_transform, _build_parcellations,
    _compute_counting_statistic_from_parcel_level_scores, rpbi_core,
    randomized_parcellation_based_inference)


### Tests for the GrowableSparseArray class ###################################
def test_gsarray_append_data():
    """This function tests GrowableSparseArray creation and filling.

    """
    # Simplest example
    gs_array = GrowableSparseArray(n_rows=1)
    gs_array.append(0, np.ones(5))
    assert_array_equal(gs_array.get_data()['row'], np.zeros(5))
    assert_array_equal(gs_array.get_data()['col'], np.arange(5))
    assert_array_equal(gs_array.get_data()['data'], np.ones(5))

    # Append with no structure extension needed
    gs_array = GrowableSparseArray(n_rows=1, max_elts=10)
    gs_array.append(0, np.ones(5))
    assert_array_equal(gs_array.get_data()['row'], np.zeros(5))
    assert_array_equal(gs_array.get_data()['col'], np.arange(5))
    assert_array_equal(gs_array.get_data()['data'], np.ones(5))

    # Void array
    gs_array = GrowableSparseArray(n_rows=1)
    gs_array.append(0, np.zeros(5))
    assert_array_equal(gs_array.get_data()['row'], [])
    assert_array_equal(gs_array.get_data()['col'], [])
    assert_array_equal(gs_array.get_data()['data'], [])

    # Toy example
    gs_array = GrowableSparseArray(n_rows=10)
    for i in range(10):
        data = np.arange(10) - i
        data[data < 8] = 0
        gs_array.append(i, data)
    assert_array_equal(gs_array.get_data()['row'], np.array([0., 0., 1.]))
    assert_array_equal(gs_array.get_data()['col'], [8, 9, 9])
    assert_array_equal(gs_array.get_data()['data'], [8., 9., 8.])


def test_gsarray_merge():
    """This function tests GrowableSparseArray merging.

    Because of the specific usage of GrowableSparseArrays, only a reduced
    number of manipulations has been implemented.

    """
    # Basic merge
    gs_array = GrowableSparseArray(n_rows=1)
    gs_array.append(0, np.ones(5))
    gs_array2 = GrowableSparseArray(n_rows=1)
    gs_array2.merge(gs_array)
    assert_array_equal(gs_array.get_data()['row'],
                       gs_array2.get_data()['row'])
    assert_array_equal(gs_array.get_data()['col'],
                       gs_array2.get_data()['col'])
    assert_array_equal(gs_array.get_data()['data'],
                       gs_array2.get_data()['data'])

    # Merge list
    gs_array = GrowableSparseArray(n_rows=2)
    gs_array.append(0, np.ones(5))
    gs_array2 = GrowableSparseArray(n_rows=2)
    gs_array2.append(1, 2 * np.ones(5))
    gs_array3 = GrowableSparseArray(n_rows=2)
    gs_array3.merge([gs_array, gs_array2])
    assert_array_equal(gs_array3.get_data()['row'],
                       np.array([0.] * 5 + [1.] * 5))
    assert_array_equal(gs_array3.get_data()['col'], np.tile(np.arange(5), 2))
    assert_array_equal(gs_array3.get_data()['data'],
                       np.array([1.] * 5 + [2.] * 5))
    # failure case
    assert_raises(TypeError, gs_array3.merge, [gs_array, gs_array2, "foo"])

    # Test failure case (merging arrays with different n_rows)
    gs_array_wrong = GrowableSparseArray(n_rows=2)
    gs_array_wrong.append(0, np.ones(5))
    gs_array_wrong.append(1, np.ones(5))
    gs_array = GrowableSparseArray(n_rows=1)
    assert_raises(ValueError, gs_array.merge, gs_array_wrong)

    # Test failure case (merge a numpy array)
    gs_array = GrowableSparseArray(n_rows=1)
    assert_raises(TypeError, gs_array.merge, np.ones(5))


### Test the parcellations building framework #################################
from sklearn.feature_extraction import image


def test_ward_fit_transform():
    """Test parcellation building and associated signal extraction.

    """
    # Generate toy data
    # define data structure
    shape = (5, 5, 5)
    mask = np.ones(shape, dtype=bool)
    connectivity = image.grid_to_graph(n_x=5, n_y=5, n_z=5, mask=mask)
    # data generation
    data1 = np.ones(shape)
    data1[1:3, 1:3, 1:3] = 2.
    data2 = np.ones(shape)
    data2[3:, 3:, 3:] = 4.
    data = np.ones((4, np.prod(shape)))  # 4 ravelized images
    data[0] = np.ravel(data1)
    data[1] = np.ravel(data2)

    # One image used for train, transform all
    parcelled_data, labels = _ward_fit_transform(data, [0], connectivity, 2, 0)
    # check parcelled_data
    assert_equal(parcelled_data.shape, (4, 2))
    assert_array_equal(
        np.sort(np.unique(parcelled_data[0])),  # order is hard to predict
        [1, 2])
    assert_array_equal(parcelled_data[2], [1, 1])
    assert_array_equal(parcelled_data[3], [1, 1])
    # check labels
    assert_equal(len(labels.shape), 1)
    assert_array_equal(np.unique(labels), [0, 1])

    # Two images used for train, transform all, add offset to labels
    parcelled_data, labels = _ward_fit_transform(data, [0, 1],
                                                 connectivity, 3, 10)
    # check parcelled_data
    assert_equal(parcelled_data.shape, (4, 3))
    assert_array_equal(
        np.sort(np.unique(parcelled_data[0])),  # order is hard to predict
        [1, 2])
    assert_array_equal(
        np.sort(np.unique(parcelled_data[1])),  # order is hard to predict
        [1, 4])
    assert_array_equal(parcelled_data[2], [1, 1, 1])
    assert_array_equal(parcelled_data[3], [1, 1, 1])
    # check labels
    assert_equal(len(labels.shape), 1)
    assert_array_equal(np.unique(labels), [10, 11, 12])


def test_build_parcellations(random_state=0):
    """Test parcellations building.
    """
    # check random state
    rng = check_random_state(random_state)

    # Generate toy data
    # define data structure
    shape = (5, 5, 5)
    mask = np.ones(shape, dtype=bool)
    # data generation
    data1 = np.ones(shape)
    data1[1:3, 1:3, 1:3] = 2.
    data2 = np.ones(shape)
    data2[3:, 3:, 3:] = 4.
    data = np.ones((4, np.prod(shape)))  # 4 ravelized images
    data[0] = np.ravel(data1)
    data[1] = np.ravel(data2)

    # Test _build_parcellations function
    parcelled_data, labels = _build_parcellations(
        data, mask, n_parcellations=2, n_parcels=3,
        # make sure we use observations 1 and 2 at least once
        n_bootstrap_samples=8, random_state=rng)
    # check parcelled_data
    assert_equal(parcelled_data.shape, (4, 3 * 2))
    assert_array_equal(
        np.sort(np.unique(parcelled_data[0])),  # order is hard to predict
        [1, 2])
    assert_array_equal(
        np.sort(np.unique(parcelled_data[1])),  # order is hard to predict
        [1, 4])
    assert_array_equal(parcelled_data[2], [1, 1, 1, 1, 1, 1])
    assert_array_equal(parcelled_data[3], [1, 1, 1, 1, 1, 1])
    # check labels
    assert_equal(len(labels.shape), 1)
    assert_array_equal(np.unique(labels), np.arange(2 * 3))


### Test RPBI code ############################################################
from scipy import sparse


def test_compute_counting_statistic_from_parcel_level_scores(random_state=1):
    """Test the computation of RPBI's counting statistic.
    """
    # check random state
    rng = check_random_state(random_state)

    # Generate toy data
    # define data structure
    shape = (5, 5, 5)
    n_voxels = np.prod(shape)
    mask = np.ones(shape, dtype=bool)
    # data generation
    data1 = np.ones(shape)
    data1[1:3, 1:3, 1:3] = 2.
    data2 = np.ones(shape)
    data2[3:, 3:, 3:] = 4.
    data = np.ones((4, np.prod(shape)))  # 4 ravelized images
    data[0] = np.ravel(data1)
    data[1] = np.ravel(data2)

    # Parcellate data and extract signal averages
    n_parcellations = 2
    n_parcels = 3
    parcelled_data, labels = _build_parcellations(
        data, mask, n_parcellations=n_parcellations, n_parcels=n_parcels,
        # make sure we use observations 1 and 2 at least once
        n_bootstrap_samples=6, random_state=rng)
    parcel_level_results = GrowableSparseArray(n_rows=2)
    data_tmp = parcelled_data[0]
    data_tmp[data_tmp < 2] = 0
    parcel_level_results.append(0, data_tmp)
    data_tmp = parcelled_data[1]
    data_tmp[data_tmp < 2] = 0
    parcel_level_results.append(1, data_tmp)
    parcellation_masks = np.zeros((n_parcellations * n_parcels, n_voxels))
    for j in np.arange(n_parcellations):  # loop on parcellations
        label_slice = slice(j * n_voxels, (j + 1) * n_voxels)
        for l in np.unique(labels[label_slice]):
            parcellation_masks[l] = labels[label_slice] == l
    parcellation_masks = sparse.coo_matrix(
        parcellation_masks.astype(np.float32)).tocsr()

    # Transform back data
    # (transformed data should be similar to the original data (up to
    # thresholding and sum across parcellations) since by construction
    # the signal is homogeneous within each parcel for each subject)
    thresholded_data = data.copy()
    thresholded_data[thresholded_data < 2] = 0.
    thresholded_data *= 2.
    res = _compute_counting_statistic_from_parcel_level_scores(
        parcel_level_results.get_data(), slice(0, 2), parcellation_masks,
        n_parcellations, n_parcellations * n_parcels)
    counting_stats_original_data, h0 = res
    assert_array_equal(counting_stats_original_data,
                       thresholded_data[0])
    assert_array_equal(h0, [8])

    # Same thing but only for the permuted data
    res = _compute_counting_statistic_from_parcel_level_scores(
        parcel_level_results.get_data()[2:], slice(1, 2),
        parcellation_masks, n_parcellations, n_parcellations * n_parcels)
    counting_stats_original_data, h0 = res
    assert_array_equal(counting_stats_original_data, [])
    assert_array_equal(h0, [8])


def test_rpbi_core(random_state=2):
    """Test Randomized Parcellation Based Inference core function.
    """
    # check random state
    rng = check_random_state(random_state)

    # Generate toy data
    # define data structure
    shape = (5, 5, 5)
    n_voxels = np.prod(shape)
    mask = np.ones(shape, dtype=bool)
    # data generation
    data = np.zeros(shape)
    data[1:3, 1:3, 1:3] = 2.
    data = data.reshape((1, -1))
    data = np.repeat(data, 8, 0)
    # add noise to avoid constant columns
    data += 0.1 * rng.randn(data.shape[0], data.shape[1])

    # Parcellate data and extract signal averages
    n_parcellations = 2
    n_parcels = 3
    parcelled_data, labels = _build_parcellations(
        data, mask, n_parcellations=n_parcellations, n_parcels=n_parcels,
        # make sure we use observations 1 and 2 at least once
        n_bootstrap_samples=6, random_state=rng)

    # RPBI from already parcelled data
    rng = check_random_state(random_state)
    pvalues, counting_statistic_original_data, h0 = rpbi_core(
        np.ones((8, 1)), parcelled_data, n_parcellations, labels,
        n_parcels, model_intercept=False, threshold=0.1 / n_parcels,
        n_perm=9, random_state=rng)
    # check pvalues
    expected_pvalues = np.zeros(shape)
    expected_pvalues[1:3, 1:3, 1:3] = 1.
    expected_pvalues = np.ravel(expected_pvalues)
    print pvalues.shape, n_voxels
    assert_equal(pvalues.shape, (n_voxels,))
    assert_array_equal(pvalues, expected_pvalues)
    # check counting statistic
    assert_equal(counting_statistic_original_data.shape, (n_voxels,))
    assert_array_equal(counting_statistic_original_data, 2 * expected_pvalues)
    # h0
    assert_equal(h0.shape, (9,))
    assert_array_equal(h0, np.zeros(9))

    # Same thing with model_intercept=True
    rng = check_random_state(random_state)
    pvalues, counting_statistic_original_data, h0 = rpbi_core(
        np.ones((8, 1)), parcelled_data, n_parcellations, labels,
        n_parcels, model_intercept=True, threshold=0.1 / n_parcels,
        n_perm=9, random_state=rng)
    # check pvalues
    expected_pvalues = np.zeros(shape)
    expected_pvalues[1:3, 1:3, 1:3] = 1.
    expected_pvalues = np.ravel(expected_pvalues)
    assert_equal(pvalues.shape, (n_voxels,))
    assert_array_almost_equal(pvalues, expected_pvalues)
    # check counting statistic
    assert_equal(counting_statistic_original_data.shape, (n_voxels,))
    assert_array_almost_equal(counting_statistic_original_data,
                              2 * expected_pvalues)
    # h0
    assert_equal(h0.shape, (9,))
    assert_array_almost_equal(h0, np.zeros(9))

    # Replace intercept test with a more complex test
    rng = check_random_state(random_state)
    tested_var = np.ones((8, 1))
    tested_var[0:4] = 0
    parcelled_data[0:4] *= -1
    pvalues, counting_statistic_original_data, h0 = rpbi_core(
        tested_var, parcelled_data, n_parcellations, labels,
        n_parcels, model_intercept=True, threshold=0.05 / n_parcels,
        n_perm=9, random_state=rng)
    # check pvalues
    expected_pvalues = np.zeros(shape)
    expected_pvalues[1:3, 1:3, 1:3] = 1.
    expected_pvalues = np.ravel(expected_pvalues)
    assert_equal(pvalues.shape, (n_voxels,))
    assert_array_almost_equal(pvalues, expected_pvalues)
    # check counting statistic
    assert_equal(counting_statistic_original_data.shape, (n_voxels,))
    assert_array_almost_equal(counting_statistic_original_data,
                              2 * expected_pvalues)
    # h0
    assert_equal(h0.shape, (9,))
    assert_array_almost_equal(h0, np.zeros(9))


def test_rpbi_core_withcovars(random_state=0):
    """Test Randomized Parcellation Based Inference core function with covars.
    """
    # check random state
    rng = check_random_state(random_state)

    # Generate toy data
    # define data structure
    shape = (5, 5, 5)
    n_voxels = np.prod(shape)
    mask = np.ones(shape, dtype=bool)
    # data generation
    data = np.zeros(shape)
    data[1:3, 1:3, 1:3] = 2.
    data = data.reshape((1, -1))
    data = np.repeat(data, 8, 0)
    # add noise to avoid constant columns
    data += 0.1 * rng.randn(data.shape[0], data.shape[1])

    # Parcellate data and extract signal averages
    n_parcellations = 2
    n_parcels = 3
    parcelled_data, labels = _build_parcellations(
        data, mask, n_parcellations=n_parcellations, n_parcels=n_parcels,
        # make sure we use observations 1 and 2 at least once
        n_bootstrap_samples=6, random_state=rng)

    # Covariates (dummy)
    covars = 0.1 * rng.randn(8).reshape((-1, 1))

    # RPBI from already parcelled data
    rng = check_random_state(random_state)
    pvalues, counting_statistic_original_data, h0 = rpbi_core(
        np.ones((8, 1)), parcelled_data, n_parcellations, labels,
        n_parcels, confounding_vars=covars, model_intercept=False,
        threshold=0.05 / n_parcels, n_perm=9, random_state=rng)
    # check pvalues
    expected_pvalues = np.zeros(shape)
    expected_pvalues[1:3, 1:3, 1:3] = 1.
    expected_pvalues = np.ravel(expected_pvalues)
    assert_equal(pvalues.shape, (n_voxels,))
    assert_array_equal(pvalues, expected_pvalues)
    # check counting statistic
    assert_equal(counting_statistic_original_data.shape, (n_voxels,))
    assert_array_equal(counting_statistic_original_data, 2 * expected_pvalues)
    # h0
    assert_equal(h0.shape, (9,))
    assert_array_equal(h0, np.zeros(9))

    # Same thing with model_intercept=True
    rng = check_random_state(random_state)
    pvalues, counting_statistic_original_data, h0 = rpbi_core(
        np.ones((8, 1)), parcelled_data, n_parcellations, labels,
        n_parcels, confounding_vars=covars, model_intercept=True,
        threshold=0.05 / n_parcels, n_perm=9, random_state=rng)
    # check pvalues
    expected_pvalues = np.zeros(shape)
    expected_pvalues[1:3, 1:3, 1:3] = 1.
    expected_pvalues = np.ravel(expected_pvalues)
    assert_equal(pvalues.shape, (n_voxels,))
    assert_array_almost_equal(pvalues, expected_pvalues)
    # check counting statistic
    assert_equal(counting_statistic_original_data.shape, (n_voxels,))
    assert_array_almost_equal(counting_statistic_original_data,
                              2 * expected_pvalues)
    # h0
    assert_equal(h0.shape, (9,))
    assert_array_almost_equal(h0, np.zeros(9))

    # Replace intercept test with a more complex test
    rng = check_random_state(random_state)
    tested_var = np.ones(8)
    tested_var[0:4] = 0
    parcelled_data[0:4] *= -1
    pvalues, counting_statistic_original_data, h0 = rpbi_core(
        tested_var, parcelled_data, n_parcellations, labels,
        n_parcels, confounding_vars=covars, model_intercept=False,
        threshold=0.1 / n_parcels, n_perm=9, random_state=rng)
    # check pvalues
    expected_pvalues = np.zeros(shape)
    expected_pvalues[1:3, 1:3, 1:3] = 1.
    expected_pvalues = np.ravel(expected_pvalues)
    assert_equal(pvalues.shape, (n_voxels,))
    assert_array_almost_equal(pvalues, expected_pvalues)
    # check counting statistic
    assert_equal(counting_statistic_original_data.shape, (n_voxels,))
    assert_array_almost_equal(counting_statistic_original_data,
                              2 * expected_pvalues)
    # h0
    assert_equal(h0.shape, (9,))
    assert_array_almost_equal(h0, np.zeros(9))

    # Same thing with intercept modelling
    rng = check_random_state(random_state)
    pvalues, counting_statistic_original_data, h0 = rpbi_core(
        tested_var, parcelled_data, n_parcellations, labels,
        n_parcels, confounding_vars=covars, model_intercept=True,
        threshold=0.1 / n_parcels, n_perm=9, random_state=rng)
    # check pvalues
    expected_pvalues = np.zeros(shape)
    expected_pvalues[1:3, 1:3, 1:3] = 1.
    expected_pvalues = np.ravel(expected_pvalues)
    assert_equal(pvalues.shape, (n_voxels,))
    assert_array_almost_equal(pvalues, expected_pvalues)
    # check counting statistic
    assert_equal(counting_statistic_original_data.shape, (n_voxels,))
    assert_array_almost_equal(counting_statistic_original_data,
                              2 * expected_pvalues)
    # h0
    assert_equal(h0.shape, (9,))
    assert_array_almost_equal(h0, np.zeros(9))


def test_randomized_parcellation_based_inference(random_state=1):
    """Test RPBI API.
    """
    # check random state
    rng = check_random_state(random_state)

    # Generate toy data
    # define data structure
    shape = (5, 5, 5)
    n_voxels = np.prod(shape)
    mask = np.ones(shape, dtype=bool)
    # data generation
    data = np.zeros(shape)
    data[1:3, 1:3, 1:3] = 1.
    data = data.reshape((1, -1))
    data = np.repeat(data, 8, 0)
    # add noise to avoid constant columns
    data += 0.1 * rng.randn(data.shape[0], data.shape[1])

    # Randomized Parcellation Based Inference
    n_parcellations = 2
    n_parcels = 3
    neg_log_pvals, counting_statistic_original_data, h0 = (
        randomized_parcellation_based_inference(
            np.ones(8), data, mask, confounding_vars=None,
            model_intercept=True,
            n_parcellations=n_parcellations, n_parcels=n_parcels,
            threshold=0.05 / n_parcels, n_perm=9, random_state=rng,
            verbose=True))
    # check pvalues
    expected_neg_log_pvals = np.zeros(shape)
    expected_neg_log_pvals[1:3, 1:3, 1:3] = 1.
    expected_neg_log_pvals = np.ravel(expected_neg_log_pvals)
    assert_equal(neg_log_pvals.shape, (n_voxels,))
    assert_array_almost_equal(neg_log_pvals, expected_neg_log_pvals)
    # check counting statistic
    assert_equal(counting_statistic_original_data.shape, (n_voxels,))
    assert_array_almost_equal(counting_statistic_original_data,
                              2 * expected_neg_log_pvals)
    # h0
    assert_equal(h0.shape, (9,))
    assert_array_almost_equal(h0, np.zeros(9))

    ### Same test with 1-dimensional tested_vars
    # check random state
    rng = check_random_state(random_state)
    rng.randn(data.shape[0], data.shape[1])
    # Randomized Parcellation Based Inference
    n_parcellations = 2
    n_parcels = 3
    neg_log_pvals, counting_statistic_original_data, h0 = (
        randomized_parcellation_based_inference(
            np.ones(8), data, mask, confounding_vars=None,
            model_intercept=True,
            n_parcellations=n_parcellations, n_parcels=n_parcels,
            threshold=0.05 / n_parcels, n_perm=9, random_state=rng,
            verbose=True))
    # check pvalues
    expected_neg_log_pvals = np.zeros(shape)
    expected_neg_log_pvals[1:3, 1:3, 1:3] = 1.
    expected_neg_log_pvals = np.ravel(expected_neg_log_pvals)
    assert_equal(neg_log_pvals.shape, (n_voxels,))
    assert_array_almost_equal(neg_log_pvals, expected_neg_log_pvals)
    # check counting statistic
    assert_equal(counting_statistic_original_data.shape, (n_voxels,))
    assert_array_almost_equal(counting_statistic_original_data,
                              2 * expected_neg_log_pvals)
    # h0
    assert_equal(h0.shape, (9,))
    assert_array_almost_equal(h0, np.zeros(9))
