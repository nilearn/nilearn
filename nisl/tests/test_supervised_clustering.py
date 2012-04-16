# Author : Jean Kossaifi <jean.kossaifi@gmail.com>

import numpy as np
from nose.tools import assert_true
from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal
from scikits.learn.feature_extraction.image import grid_to_graph
from scikits.learn.linear_model import BayesianRidge
from scikits.learn.cluster import ward_tree

from .. import supervised_clustering

###############################################################################
# Generating data for test purpose
# X is composed of two convex parts
# X2 is of shape(n_samples, size**2)
n_samples = 100
size = 4  # image size
roi_size = 2
X = np.zeros(size**2)
X2 = X
#Generating two convexe parts
mask = np.zeros((size, size), dtype=bool)
mask[0:roi_size, 0:roi_size] = True
mask[-roi_size:, -roi_size:] = True
mask = mask.reshape(size**2)
X = X[mask]
# making n_samples
X2 = X2 + np.zeros((n_samples, 1))
X = X + np.arange(n_samples).reshape((n_samples, 1))
Y = np.arange(n_samples)
# Generating the connectivity grids and ward trees
A = grid_to_graph(n_x=size, n_y=size, mask=mask)
children, n_components, n_leaves = ward_tree(X.T, connectivity=A,
        n_components=2)
children = children.tolist()
A2 = grid_to_graph(n_x=size, n_y=size)
children2, n_components2, n_leaves2 = ward_tree(X2.T, connectivity=A2,
        n_components=1)
children2 = children2.tolist()


###############################################################################
# Test functions
def test_tree_roots():
    """
    Tests that the function returns the right roots.
    """
    roots1 = supervised_clustering.tree_roots(children,
            n_components, n_leaves)
    assert_equal(roots1, [12, 13])
    roots2 = supervised_clustering.tree_roots(children2,
            n_components2, n_leaves2)
    assert_equal(roots2, [30])


def test_find_children():
    """
    Tests that the function returns the right children,
    and only the right
    """
    children = [[0, 1], [4, 2], [3, 5], [6, 7]]
    child = supervised_clustering.find_children(7, children, n_leaves=5)
    child.sort()  # The order isn't important
    assert_true(np.array_equal(child, np.array([0, 1, 3])))


def test_average_signals():
    """
    Checks that the average_signal is correct for every node
    """
    X = np.arange(8, dtype=np.float)
    X = X.reshape((2, 4))
    children = [[0, 1], [4, 2], [5, 3]]
    n_leaves = 4
    avg_signals = supervised_clustering.average_signals(X, children,
            n_leaves)
    true_result = np.array([[0., 1., 2., 3., 1./2., 3./3., 6./4.],
                            [4., 5., 6., 7., 9./2., 15./3., 22./4.]],
                            dtype=np.float)
    assert_true(np.array_equal(avg_signals, true_result))


def test_parcel_based_signals():
    """
    Checks that the parcel_based_signal is correct
    """
    # if every voxel IS a parcel then the parcel_based_signal is X
    X = np.arange(3, dtype=np.int)
    X = X + np.zeros(3, dtype=np.int).reshape((3, 1))
    avg_signals = supervised_clustering.parcel_based_signals(X, X[0, :])
    assert_true(np.array_equal(avg_signals, X))
    assert_true(np.array_equal(avg_signals[X[0, :]], X))
    # A simple test
    X = np.arange(8, dtype=np.float)
    X = X.reshape((2, 4))
    labels = [1, 1, 1, 2]
    avg_signals = supervised_clustering.parcel_based_signals(X, labels)
    true_result = np.array([[1, 3],
                            [5, 7]], dtype=np.float)
    assert_true(np.array_equal(avg_signals, true_result))
    assert_true(avg_signals.shape[0] == X.shape[0])
    # An other test
    X = np.arange(10)
    X = X.reshape((2, 5))
    labels = [1, 1, 2, 1, 2]
    avg_signals = supervised_clustering.parcel_based_signals(X, labels)
    true_result = np.array([[4./3., 3.], [19./3., 8.]])
    assert_true(np.array_equal(avg_signals, true_result))


def test_split_parcellation():
    """
    Checks that the function splits correctly the parcellations
    """
    # Two convex parts
    parcellations = supervised_clustering.split_parcellation(
            supervised_clustering.tree_roots(
                children, n_components, n_leaves), children, n_leaves)
    for i in parcellations:  # the order of the parcels is not important
        i.sort()
    parcellations.sort()
    real_result = [[8, 9, 13], [10, 11, 12]]
    assert_true(parcellations, real_result)

    # One convex part
    parcellations = supervised_clustering.split_parcellation(
            supervised_clustering.tree_roots(
                children2, n_components2, n_leaves2), children2, n_leaves2)
    assert_true(np.array_equal(parcellations, [[29, 28]])
            or np.array_equal(parcellations, [[28, 29]]))

    # checking that the larger parcellation possible is not splitted
    # (how could it be, it's composed of leaves !)
    assert_true(supervised_clustering.split_parcellation(
        [0, 1, 2, 3, 4, 5, 6, 7], children, n_leaves) == [])


def test_parcellation_to_label():
    """
    Cheks that the function give a unique label to every feature of every
    parcel of the parcellation, and that every parcel has a distinct label
    associated
    """
    children = [[0, 1], [4, 2], [3, 5]]
    parcellation = [2, 4, 7]
    n_leaves = 5
    labels = supervised_clustering.parcellation_to_label(
            parcellation=parcellation, children=children, n_leaves=n_leaves)
    assert_true(len(labels) == 5)
    # Checking that there are the write number of labels
    assert_true(len(np.unique(labels)) == 3)
    # checking that every feature in the same parcel has the same label
    assert_true(labels[0] == labels[1] == labels[3])
    # checking that every parcel has a different label associated
    assert_true(labels[0] != labels[2] != labels[4])
    # This test may be carefully removed :
    # It checks that parcellation[0] has label 1, ...
    #                parcellation[n] has labal (n+1) ...
    assert_true(np.array_equal(np.array([3, 3, 1, 3, 2]), labels))


def test_inverse_transform():
    """
    Check that every voxel has the right coef
    """
    sc = supervised_clustering.SupervisedClusteringRegressor()
    children = [[0, 1], [4, 2], [3, 5]]
    parcellation = [2, 4, 7]
    sc.labels_ = np.array([1, 1, 2, 1, 3])
    sc.coef_ = [4, 10, 2]
    assert_true(np.array_equal(sc.inverse_transform(), [4/3, 4/3, 10, 4/3, 2]))


def test_fit():
    """
    A very frugal test for the fit method
    /!\ TODO
    """
    clf = BayesianRidge()
    hc = supervised_clustering.SupervisedClusteringRegressor(n_iterations=5,
            connectivity=A)
    hc.fit(X, Y)
    labels = hc.labels_


def test_predict():
    """
    Checks that the prediction is good at least for the train set
    """
    hc = supervised_clustering.SupervisedClusteringRegressor(connectivity=A,
            verbose=0, n_jobs=8)
    # The algorithm will (normally) stop before 40 iterations
    # (there are not enought leaves)
    hc.fit(X, Y)
    # Checking that the prediction is correct for an over-fitted set
    assert_array_almost_equal(hc.predict(X).tolist(), Y, 2)
    ### Same test using instead a classifior
    hc = supervised_clustering.SupervisedClusteringClassifier(connectivity=A,
            verbose=0, n_jobs=8)
    y = Y.copy()
    y[Y >= 50] = 1
    y[Y < 50] = 0
    # The algorithm will (normally) stop before 40 iterations
    # (there are not enought leaves)
    hc.fit(X, y)
    # Checking that the prediction is correct for an over-fitted set
    assert_array_almost_equal(hc.predict(X).tolist(), y, 2)
