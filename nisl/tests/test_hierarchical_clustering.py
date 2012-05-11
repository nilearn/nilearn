
# Author : Jean Kossaifi

from .. import hierarchical_clustering

import numpy as np
from nose.tools import assert_true
from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal
from scikits.learn.feature_extraction.image import grid_to_graph
from scikits.learn.linear_model import BayesianRidge
from scikits.learn.linear_model import RidgeCV
from scikits.learn.cluster import ward_tree


###############################################################################
# Generating data for test purpose
# X is composed of two convex parts
# X2 is of shape(n_samples, size)
n_samples = 200
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
    roots1 = hierarchical_clustering.tree_roots(children,
            n_components, n_leaves)
    assert_equal(roots1, [12, 13])
    roots2 = hierarchical_clustering.tree_roots(children2,
            n_components2, n_leaves2)
    assert_equal(roots2, [30])


def test_average_signals():
    """
    Check that the average_signal is correct for every node
    """
    X = np.arange(8, dtype=np.float)
    X = X.reshape((2, 4))
    children = [[0, 1], [4, 2]]
    n_leaves = 4
    avg_signals = hierarchical_clustering.average_signals(X, children, n_leaves)
    true_result = np.array([[0., 1., 2., 3., 1./2., 3./3.],
                            [4., 5., 6., 7., 9./2., 15./3.]], dtype=np.float)
    assert_true(np.array_equal(avg_signals, true_result))
    assert_true(avg_signals.shape[0] == X.shape[0])
    # Checking that if there are no nodes, the result is X
    children = []
    avg_signals = hierarchical_clustering.average_signals(X, children, n_leaves)
    assert_true(np.array_equal(avg_signals, X))
    

def test_parcel_based_signals():
    """
    Check that the parcel_based_signal is correct
    """
    # if every voxel is a parcel then the parcel_based_signal is X
    # that's what we're checking
    X = np.arange(3, dtype=np.int)
    X = X + np.zeros(3, dtype=np.int).reshape((3, 1))
    avg_signals = hierarchical_clustering.parcel_based_signals(X, X[0, :])
    assert_true(np.array_equal(avg_signals, X))
    assert_true(np.array_equal(avg_signals[X[0, :]], X))
    # A simple test
    X = np.arange(8, dtype=np.float)
    X = X.reshape((2, 4))
    labels = [1, 1, 1, 2]
    avg_signals = hierarchical_clustering.parcel_based_signals(X, labels)
    true_result = np.array([[1, 3],
                            [5, 7]], dtype=np.float)
    assert_true(np.array_equal(avg_signals, true_result))
    assert_true(avg_signals.shape[0] == X.shape[0])



def test_split_parcel():
    """
    Check that the parcel is spitted well
    """
    # Splitting a parcellation with a single parcel
    # Wich parcel is the result of the merging of two features
    X = np.arange(4)
    X = X.reshape((2, 2))
    children = [[0, 1]]
    n_leaves = 2
    parcellation = hierarchical_clustering.split_parcel([2], 0, children,
            n_leaves)
    parcellation.sort()
    assert_true(parcellation == [0, 1])
    


def test_split_parcellation():
    """
    Checks that the function splits correctly the parcellations
    """
    # Two convex parts
    parcellations = hierarchical_clustering.split_parcellation(
            hierarchical_clustering.tree_roots(
                children, n_components, n_leaves), children, n_leaves) 
    # -- the order of the parcels is not important
    for i in parcellations:
        i.sort()
    parcellations.sort()
    real_result = [[8, 9, 13], [10, 11, 12]]
    assert_true(parcellations, real_result)
    # One convex part
    parcellations = hierarchical_clustering.split_parcellation(
            hierarchical_clustering.tree_roots(
                children2, n_components2, n_leaves2), children2, n_leaves2)
    assert_true(np.array_equal(parcellations, [[29, 28]]) 
            or np.array_equal(parcellations, [[28, 29]] ))
    # checking that the larger parcellation possible is not splitted
    # (how could it be, it's composed of features !)
    assert_true(hierarchical_clustering.split_parcellation(
        [0, 1, 2, 3, 4, 5, 6, 7], children, n_leaves) == [])


def test_give_label():
    """
    Checks that the function give the given label to every children
    of the node 'root'
    """
    children = [[0, 1], [4, 2]]
    labels = np.zeros(4)
    hierarchical_clustering.give_label(label=1, 
            root=5, children=children, labels=labels, n_leaves=4)
    assert_true(labels.tolist() == [1, 1, 1, 0])


def test_parcellation_to_label():
    """
    Cheks that the function give a unique label to every feature of every
    parcel of the parcellation, and that every parcel has a distinct label
    associated
    """
    children = [[0, 1], [4, 2]]
    parcellation = [5, 3]
    n_leaves = 4
    labels = hierarchical_clustering.parcellation_to_label(
            parcellation=parcellation, children=children, n_leaves=n_leaves)
    assert_true(labels.shape == (4, ))
    # checking that every feature in the same parcel has the same label
    assert_true(labels[0] == labels[1] == labels[2])
    # checking that every parcel has a different label associated
    assert_true(labels[0] != labels[3])


def test_fit():
    """
    A very frugal test for the fit method
    """
    clf = BayesianRidge()
    hc = hierarchical_clustering.HierarchicalClustering(clf, A)
    # after no iterations, the parcellation correspond to the roots of the tree
    tab = hc.fit(X, Y, 0)
    assert_true(tab.tolist() == [1, 1, 1, 1, 2, 2, 2, 2])


def test_predict():
    """
    Checks that the prediction is good at least for over-fitting problem
    """
    clf = BayesianRidge()
    hc = hierarchical_clustering.HierarchicalClustering(clf, A)
    tab = hc.fit(X, Y, 50)
    # Checking that the prediction is correct for an over-fitted set
    assert_array_almost_equal(hc.predict(X).tolist(), Y, 6)


###############################################################################
# Tests
# Uncomment to run tests whithout nosetest
#test_tree_roots()
#test_average_signals()
#test_parcel_based_signals()
#test_split_parcel()
#test_split_parcellation()
#test_give_label()
#test_parcellation_to_label()
#test_fit()
#test_predict()
