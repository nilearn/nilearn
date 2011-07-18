"""
=====================
Supervised clustering
=====================

supervised_clustering is a module to reduce the number of features on wich
we perform regression or classification by performing a hierarchical clustering
with cross validation and spatial considerations.

Firstly, ward tree is computed, and then, we find the best cut of this tree, 
iteratively starting from the root(s)
"""

# Author: Jean Kossaifi <jean.kossaifi@gmail.com>

import numpy as np

from scikits.learn.cluster import ward_tree
from scikits.learn import cross_val
from scipy import sparse
from scikits.learn.externals.joblib import Parallel, delayed


###############################################################################
def tree_roots(children, n_components, n_leaves):
    """
   Computes a list of all the roots of the tree.

    Parameters
    ----------
    children : [int, int] list
        ward_tree, it's a binary tree
        The element i represent the children of the node (n_leaves + i)

    n_components : int
        number of connected components

    n_leaves : int
        number of leaves in the tree

    Returns
    -------
    int list : list of the tree roots
    """
    if n_components == 1:
        #Only one component, the only root is the result of the last merge :
        #2*n_leaves-n_components-1 becauses indices start up to zero
        return [2 * n_leaves - 2]
    else:
        #A node is a root if it's not in the list of children
        #(it's not merged with another one)
        return list(set(range(2*n_leaves-n_components))
                .difference(set(np.array(children).flatten())))


###############################################################################
def average_signals(X, children, n_leaves):
    """ 
    Computes a 2D-array of the average signal per node.
    (ie per parcel since every node represents a possible parcel)

    Parameters
    ----------
    X : ndarray of shape = (n_samples, n_features)
    children : [int, int] list
        ward_tree

    n_leaves : int
        number of leaves in the tree

    Returns
    -------
    ndarray of shape (n_samples, n_nodes)
    (assuming a node may be a leaf)
    It's column i represents the average signal for the parcel represented
    by the node i

    """
    if children == []:
        return X
    avg_signals = []  # average signal of the corresponding element in children
    weigths = []  # number of voxels merged by the corresponding root
    for l in children:
        r = 0
        w = 0
        for i in l:
            if  i < n_leaves:  # if the element is a leave
                r += X[:, i]  # the average is the value of the voxel
                w += 1  # he is the result of the merging of 1 voxel
            else:  # the element is a root
                r += avg_signals[i % n_leaves] * weigths[i % n_leaves]
                w += weigths[i % n_leaves]
        avg_signals.append(r / w)
        weigths.append(w)
    return np.concatenate((X, (np.array(avg_signals).T)), axis=1)


###############################################################################
def parcel_based_signals(X, labels):
    """
    Computes a 2D-array of the average signal per parcel.

    Parameters
    ----------
    X : ndarray of shape = (n_samples, n_features)

    labels : nparray, dtype=int, shape=n_features
        Represents a parcellation
        Array of the labels attributed to each voxel
        (ie to each feature)


    Returns
    -------
    ndarray of shape (n_samples, n_parcels)
    where n_parcels = number of parcels in the parcellation
    and column i represent the average signal for the parcel i
    """
    avg_signals = []
    for i in np.unique(labels):
        avg_signals.append(X[:, labels == i].sum(axis=-1) / (labels == i).sum())
    return np.array(avg_signals).T


###############################################################################
def split_parcel(parcellation, i, children, n_leaves):
    """
    Splits the element i of parcellation in two, returns the new parcel.

    WARNING : we assume that the indice i given
    does not correspond to a leave

    Parameters
    ----------
    parcellation : int list
        parcellation in wich we want to split a parcel in two

    i : int
        indice of the parcel we want to split in the parcellation

    children : [int, int] list
        ward_tree

    n_leaves : int
        number of leaves in the tree

    Returns :
    ---------
    int list
    the parcellation obtained by the splitting the parcel i
    of the parcellation in two
    (the parcel i is replaced by it's two children in the ward tree)
    """
    # /!\ if you do l = children[...], you're getting a reference, not a copy!
    a, b = children[parcellation[i] % n_leaves]
    l = [a, b]
    l.extend(parcellation[:])
    l.remove(parcellation[i])

    return l


###############################################################################
def split_parcellation(parcellation, children, n_leaves):
    """ 
    Computes a list of all possible splited parcellations obtained
    by splitting one and only one parcel of parcellation

    Parameters
    ----------
    parcellation : int list
        parcellation

    children : (int * int) list
        ward_tree

    n_leaves : int
        number of leaves in the tree

    Returns
    -------
    (int list) list
    A list of the parcellations obtained by splitting one and only one
    of the current parcellation in two
    (ie splitting each element of parcellation in two when it's possible)
    """
    parcellations = [split_parcel(parcellation, i, children, n_leaves)\
            for i in range(len(parcellation)) if parcellation[i] >= n_leaves]

    return parcellations


##############################################################################
def give_label(label, root, children, labels, n_leaves):
    """
    Fills the nparray labels.
    Gives to every children of 'root' the label.

    Parameters
    ----------
    label : int
        the label to give to 'root' and it's descendents

    root : int
        the node to witch we want to give the label

    children : (int* int) list
        ward tree

    labels : nparray, dtype=int
        array of the labels attributed to each voxel

    n_leaves : int
        number of leaves in the tree

    Returns
    -------
    labels : nparray, dtype=int
        updates labels, giving the same label to the whole parcel
        represented by 'root'
    """
    if (root < n_leaves):  # we give the label to this child
        labels[root] = label
    else:  # we give the label to the two children of this root
        a, b = children[root % n_leaves]
        give_label(label, a, children, labels, n_leaves)
        give_label(label, b, children, labels, n_leaves)


###############################################################################
def parcellation_to_label(parcellation, children, n_leaves):
    """
    Computes a 2D array where every voxel in the same cluster
    have the same number (label).

    Parameters
    ----------
    parcellation : int list

    children : (int * int) list
        ward tree

    n_features :
        number of features of the clustered data

    n_leaves :
        number of leaves in the ward tree

    Returns
    -------
    a numpy array, shape=n_features, dtype=1,
    where every voxel in the same parcel
    have the same number/label (ie, labels[i]==labels[j] if
    X[i] and X[j] are in the same parcel in parcellation
    """
    labels = np.zeros(n_leaves)
    for i in range(len(parcellation)):
        give_label(i+1, parcellation[i], children, labels, n_leaves)

    return labels


###############################################################################
def select_best_parcellation(parcellations, clf, avg_signals, y, n_jobs,
        verbose):
    """
    Returns the best parcellation of the parcellations given in arguments.

    Parameters
    ----------
    parcellations : (int list) list
        list of all the parcellations possible for
        the current iteration

    clf : classifier

    avg_signals : nparray
        average_signals for each node

    y : ndarray of shape = (n_samples)

    n_jobs : int
        number of cpu to use

    verbose : int, optional
        does it really need explanations?

    Returns
    -------
    parcellation : int list
        best parcellation of the parcellations given in argument
    """
    # Computing scores for each parcellation
    scores = Parallel(n_jobs)(delayed(cross_val.cross_val_score)
            (estimator=clf, X=avg_signals[:, j], y=y,
                #cv=cross_val.KFold(avg_signals.shape[0], 5),
                    n_jobs=1)
            for j in parcellations)

    if verbose >= 2:
        print " Scores of each parcellation for current iteration :"
        print scores

    scores = Parallel(n_jobs)(delayed(np.mean)(i) for i in scores)
    indice = scores.index(max(scores))
    return parcellations[indice], scores[indice]


###############################################################################
# Class for using hierarchical clustering

class HierarchicalClustering():
    """
    A classifier using hierarchical clustering to reduce  features number

    Parameters
    ----------
    clf : classifier, with methods fit and predict

    A : sparse matrix, optionnal
        connectivity matrix

    copy : bool
        default : true
        true if you want to use a copy of A

    Attributes
    ----------
    scores : float list
        the score of the best parcellation of each iteration
        the element i correspond to the score of the best parcellation
        at iteration i
    coef_ : the coefficients of the classifier fitted to the data
        (using the hierarchical clustering)
    labels : (int list) 
        the parcellation chosen
    """

    def __init__(self, clf, A=None, n_jobs=-1, copy=True):
        if copy and A is not None:
            self.A = A.copy()
        else:
            self.A = A
        self.copy = copy
        self.clf = clf
        self.n_jobs = n_jobs
        if A is not None:
            self.n_components_A = sparse.cs_graph_components(A)[0]
        else:
            self.n_components_A = 1

    def fit(self, X, y, n_iterations, verbose=0):
        """
        Fits Hierarchical CLustering.

        Parameters
        ----------
        X : ndarray of shape = (n_samples, n_features)
        
        Y : ndarray of shape = (n_samples)

        A : sparse matrix
            connectivity matrix

        n_iterations : int
            number of iterations = max number of parcel we want

        verbose : int, optional
            does it really need explanations?

        Returns
        -------
        tab : ndarray
            a list of labels of shape (n_features)
            two features have the same label if they are in the same parcel.
            The smaller label correspond to the smaller parcel
        """
        # Computing the ward tree
        children, n_components, n_leaves = ward_tree(X.T,
                connectivity=self.A, n_components=self.n_components_A)
        # Converting children from numpy array to list (faster)
        children = children.tolist()
        # Computing the parcel_based_signal for each parcel
        avg_signals = average_signals(X, children, n_leaves)
        # The first parcellations is the list of the tree roots
        parcellation = tree_roots(children, n_components, n_leaves)
        parcellations = [parcellation]  # List of the best parcellations
        self.scores = []
        if verbose:
            print "\n# First parcellation (=tree roots) : %s" % parcellations

        for i in range(1, n_iterations+1):  # for verbose mode
            if verbose:
                print "# Iteration %d" % i
            # Computing all the parcellations obtainable by splitting a parcel
            # of the current parcellation
            iteration_parcellations = split_parcellation(parcellation,
                    children, n_leaves)

            if (len(iteration_parcellations) == 0):
                # No parcellation can be splitted
                print " UserWARNING : n_iterations is too big :"
                print " Ending function at iteration %d." % i
                break

            # Selecting the best parcellation for current iteration
            parcellation, score = select_best_parcellation(
                    iteration_parcellations, self.clf, avg_signals, y,
                    self.n_jobs, verbose)

            parcellations.append(parcellation)
            self.scores.append(score)

        # We select the best parcel of those "pre-selected"
        #parcellation = select_best_parcellation(parcellations,
        #    self.clf, avg_signals, y, self.n_jobs, verbose)
        if self.scores != []: # Otherwise, max is not defined
            parcellation = parcellations[self.scores.index(max(self.scores))]
        # Sorting the parcellation, so the smaller label correspond 
        # to the smaller parcel
        parcellation.sort()
        # Computing the corresponding labels array
        self.tab = parcellation_to_label(parcellation, children, n_leaves)
        self.clf.fit(avg_signals[:, parcellation], y)
        self.coef_ = self.clf.coef_
        return self.tab

    def predict(self, X):
        """ 
        Predicts target values according to the fitted model

        Parameters
        ----------
        X : ndarray of shape = (n_samples, n_features)

        Returns
        -------
        The return type is the result of the clf predict function applied to
        the previously computed parcellation based signal.
        """

        avg_signals = parcel_based_signals(X, self.tab)
        return self.clf.predict(avg_signals)
