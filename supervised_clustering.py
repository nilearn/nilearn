"""
=====================
Supervised clustering
=====================

An estimator that performs classification or regression,
    using a given estimator
It performs feature agglomeration to reduce the number of features.
"""

# Author: Jean Kossaifi <jean.kossaifi@gmail.com>

import numpy as np
from scikits.learn.utils._csgraph import cs_graph_components
from scikits.learn.cluster import ward_tree
from scikits.learn.externals.joblib import Parallel, delayed
from scikits.learn.base import BaseEstimator
from scikits.learn.linear_model import BayesianRidge
from scikits.learn.linear_model import SGDClassifier
from scikits.learn.cross_val import cross_val_score


###############################################################################
def tree_roots(children, n_components, n_leaves):
    """
   Computes a list of all the roots of the tree.

    Parameters
    ----------
    children : [int, int] list
        ward_tree, it's a binary tree
        The element i represent the two children of the node (n_leaves + i)

    n_components : int
        number of connected components

    n_leaves : int
        number of leaves in the tree

    Returns
    -------
    int list : list of the tree roots

    Note
    ----
    There are (2*n_leaves - n_components) nodes in a graph
    """
    if n_components == 1:
        #Only one component, the only root is the result of the last merge :
        #2*n_leaves-n_components-1 becauses indices start up to zero
        return [2 * n_leaves - 2]
    else:
        #A node is a root if it's not in the list of children
        #(it's not merged with another one)
        return list(set(range(2 * n_leaves - n_components))
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
        # children doesn't contain entries for leaves nodes. We'll append them
        # at the end
        node_signal = 0
        node_weight = 0
        for i in l:
            if  i < n_leaves:  # if the element is a leave
                node_signal += X[:, i]  # the average is the value of the voxel
                node_weight += 1  # he is the result of the merging of 1 voxel
            else:  # the element is a root
                node_signal += avg_signals[i - n_leaves] * weigths[i - n_leaves]
                node_weight += weigths[i - n_leaves]
        avg_signals.append(node_signal / node_weight)
        weigths.append(node_weight)
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
        Array of the labels attributed to each feature


    Returns
    -------
    ndarray of shape (n_samples, n_parcels)
    where n_parcels = number of parcels in the parcellation
    and column i represent the average signal for the parcel i
    """
    avg_signals = []
    for i in np.unique(labels):
        avg_signals.append(X[:, labels == i].mean(axis=-1))
    return np.array(avg_signals).T


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
    parcellations : (int list) list
    A list of the parcellations obtained by splitting one and only one
    of the current parcellation in two
    (ie splitting each element of parcellation in two when it's possible)
    """
    parcellations = []
    for i in range(len(parcellation)):
        if parcellation[i] >= n_leaves:  # We can't cut the leaves
            # l = list(p) gets a copy of p, not l = p !!!
            l = list(parcellation)
            l.remove(parcellation[i])
            l.extend(children[parcellation[i] - n_leaves])
            parcellations.append(l)

    return parcellations


##############################################################################
def find_children(root, children, n_leaves):
    """
    Find every children of root

    Parameters
    ----------
    root : int

    children : (int* int) list
        ward tree

    n_leaves : int
        number of leaves in the tree

    Returns
    -------
    numpy array : a list of all the children of 'root'
    """
    if (root < n_leaves):
        return [root]
    else:
        a, b = children[root - n_leaves]
        l1 = find_children(a, children, n_leaves)
        l1.extend(find_children(b, children, n_leaves))
        return l1


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
    labels = np.empty(n_leaves, dtype=np.int)
    for i, root in enumerate(parcellation):
        labels[find_children(root, children, n_leaves)] = (i + 1)

    return labels


###############################################################################
# Class for using hierarchical clustering

class BaseSupervisedClustering(BaseEstimator):
    """
    A classifier using hierarchical clustering to reduce  features number

    Parameters
    ----------
    estimator : estimator, with methods fit and predict

    n_iterations : int
        default : 50
        number of iterations = max number of parcel we want

    connectivity : sparse matrix, optionnal
        connectivity matrix

    cv : scikits cross_validation model, optional
        the type of cross validation to use

    copy : bool, optional
        true if you want to use a copy of connectivity
        default is true

    n_jobs : int, optional
        number of cpu to use
        default is 1


    verbose : int, optional
        level of verbosity (0, 1 or 2)
        Default is 0


    Returns
    -------
    self


    Attributes
    ----------
    scores : float list
        the score of the best parcellation of each iteration
        the element i correspond to the score of the best parcellation
        at iteration i

    coef_ : the coefficients of the classifier fitted to the data
        (using the hierarchical clustering)

    labels_ : ndarray of shape (n_features)
        the parcellation chosen
        Two features have the same label if they are in the same parcel.
        The smaller label correspond to the smaller parcel

    Notes
    -----

    1) Firstly, ward tree is computed. Its a binary tree.
    The ward tree is represented by a list of pairsi
    Thus, the element i of this list represent the two children
    of the node i.
    /!\ The leaves are NOT represented in this list.

    2) Every cut of this tree gives a possible parcellation.
    Thus, we set the first parcellation as the root(s) of the tree, and,
    at each iteration, we construct a list of all the parcellations obtainable
    by cutting one of the parcels of the current parcellation in two.
    (we replace the parcel by its two children)

    3)We select the best parcellation of this list by cross validation
    This is the iteration parcellation.

    4)Finally, we obtain a list of al the parcellations choosen at
    each iteration.
    We choose the best by- selecting the best delta, ie the
    parcellation i+1, where
    score_of_parcellation[i+1] - score_of_parcellation[i] is max.

    5)We then compute a list wich associate to each voxel the label of its
    parcellation.
    Lastly, we fit the given estimator with this parcel-based signal.

    Comments
    --------

    After n iterations we have (n+1) parcellations
    (n choosen, plus the root(s)).

    The selected parcellation of the iteration i has n_components+n parcels.

    The root n is the element (n - n_leaves) of the ward tree
    (we assume that the features are referenced by integers from 0 to n_leaves,
    and the ward tree, an (int * int) list represent the roots of the tree by
    its two children)

    Reference
    ---------

    http://hal.inria.fr/docs/00/58/92/01/PDF/supervised_clustering_vm_review.pdf
    """

    def __init__(self, estimator, n_iterations=50, connectivity=None,
            copy=True, cv=None, n_jobs=1, verbose=0):
        if copy and connectivity is not None:
            self.connectivity = connectivity.copy()
        else:
            self.connectivity = connectivity

        self.estimator = estimator
        self.cv = cv
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.copy = copy

    def fit(self, X, y):
        """
        Fits Supervised Clustering.

        Parameters
        ----------
        X : ndarray of shape = (n_samples, n_features)

        Y : ndarray of shape = (n_samples)

        Returns
        -------
        self
        """
        # n_components computed here because the user can change connectivity
        if self.connectivity is not None:
            self.n_components = cs_graph_components(self.connectivity)[0]
        else:
            self.n_components = 1

        children, n_components, n_leaves = ward_tree(X.T,
                connectivity=self.connectivity, n_components=self.n_components)
        children = children.tolist()  # Faster with a list
        avg_signals = average_signals(X, children, n_leaves)
        # The first parcellations is the list of the tree roots
        parcellation = tree_roots(children, n_components, n_leaves)
        parcellations = []  # List of the best parcellations
        self.scores_ = []
        if self.verbose >= 2:
            print "\n# First parcellation (=tree roots) : %s" % parcellations

        ## EXPLORATION LOOP
        for i in range(1, self.n_iterations+1):  # for verbose mode

            if self.verbose:
                print "# Iteration %d" % i
            iteration_parcellations = split_parcellation(parcellation,
                    children, n_leaves)

            if (len(iteration_parcellations) == 0):
                # No parcellation can be splitted
                print " UserWARNING : n_iterations is too big :"
                print " Ending function at iteration %d." % i
                break

            # Selecting the best parcellation for current iteration
            scores = Parallel(n_jobs=self.n_jobs)(delayed(cross_val_score)
                (estimator=self.estimator, X=avg_signals[:, j], y=y,
                cv=self.cv, n_jobs=1, verbose=self.verbose)
                for j in iteration_parcellations)
            scores = np.mean(scores, axis=1)
            indice = np.argmax(scores)
            parcellation = np.copy(iteration_parcellations[indice])
            parcellations.append(np.copy(parcellation))
            self.scores_.append(np.copy(scores[indice]))

        ## SELECTION LOOP
        # We select the parcellation for wich the variation of score is
        # the bigger, only if it score is > score_max / 2
        score_min = 5 * (np.max(self.scores_) / 6)
        max = 0
        indice = 0
        self.delta_scores = []
        for i in range(1, len(self.scores_)-1):
            if self.scores_[i+1] >= score_min:
                current_delta = self.scores_[i+1] - self.scores_[i]
                if current_delta > max:
                    max = current_delta
                    indice = i
                self.delta_scores_.append(current_delta)
        else:
            self.delta_scores_.append(0)

        parcellation = parcellations[indice]

        # Computing the corresponding labels array
        self.labels_ = parcellation_to_label(parcellation, children, n_leaves)
        X = avg_signals[:, parcellation]
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std==0] = 1

        self.estimator.fit((avg_signals[:, parcellation] - self.mean)
                / self.std, y)

        if hasattr(self.estimator, 'coef_'):
            self.coef_ = self.estimator.coef_

        return self

    def inverse_transform(self):
        """
        Returns a numpy array of shape n_features where the element i
        correspond to the coefficient attributed to the feature i
        by the supervised clustering
        """
        if self.coef_ is None:
            return None

        coefs = np.empty(len(self.labels_))
        for i, label in enumerate(np.unique(self.labels_)):
            coefs[self.labels_ == label] =\
                    self.coef_[i] / np.sum(self.labels_ == label)
        return coefs

    def transform(self, X):
        """
        Returns the average signal on the selected parcels.

        Parameters
        ----------
        X : ndarray of shape = (n_samples, n_features)

        Returns
        -------
        ndarray of shape = (n_samples, len(np.unique(self.labels_)))
        The parcel based-signal.
        """
        return parcel_based_signals(X, self.labels_)

    def predict(self, X):
        """
        Predicts target values according to the fitted model

        Parameters
        ----------
        X : ndarray of shape = (n_samples, n_features)

        Returns
        -------
        The return type is the result of the estimator predict function applied
        to the previously computed parcellation based signal.
        """
        avg_signals = self.transform(X)
        return self.estimator.predict((avg_signals - self.mean) / self.std)

    def score(self, X, y):
        """
        Returns the error of the classifier self.estimator,
        using the parcel_based_signals constructed with X

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]
            Training set.

        y : array_like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        z : float

        Note
        ----
        See the estimator score function for more details
        """
        avg_signals = self.transform(X)
        return self.estimator.score((avg_signals - self.mean) / self.std, y)


def SupervisedClusteringClassifier(
        estimator=SGDClassifier(loss="hinge", penalty="l1"),
        n_iterations=50, connectivity=None, copy=True,
        cv=None, n_jobs=1, verbose=0):
    """
    A classifier using hierarchical clustering to reduce features number

    Parameters
    ----------
    estimator : estimator, with methods fit and predict

    n_iterations : int
        default : 50
        number of iterations = max number of parcel we want

    connectivity : sparse matrix, optionnal
        connectivity matrix

    cv : scikits cross_validation model, optional
        the type of cross validation to use

    copy : bool, optional
        true if you want to use a copy of connectivity
        default is true

    n_jobs : int, optional
        number of cpu to use
        default is 1


    verbose : int, optional
        level of verbosity (0, 1 or 2)
        Default is 0


    Returns
    -------
    self


    Attributes
    ----------
    scores : float list
        the score of the best parcellation of each iteration
        the element i correspond to the score of the best parcellation
        at iteration i

    coef_ : the coefficients of the classifier fitted to the data
        (using the hierarchical clustering)

    labels_ : ndarray of shape (n_features)
        the parcellation chosen
        Two features have the same label if they are in the same parcel.
        The smaller label correspond to the smaller parcel

    Notes
    -----

    1) Firstly, ward tree is computed. Its a binary tree.
    The ward tree is represented by a list of pairsi
    Thus, the element i of this list represent the two children
    of the node i.
    /!\ The leaves are NOT represented in this list.

    2) Every cut of this tree gives a possible parcellation.
    Thus, we set the first parcellation as the root(s) of the tree, and,
    at each iteration, we construct a list of all the parcellations obtainable
    by cutting one of the parcels of the current parcellation in two.
    (we replace the parcel by its two children)

    3)We select the best parcellation of this list by cross validation
    This is the iteration parcellation.

    4)Finally, we obtain a list of al the parcellations choosen at
    each iteration.
    We choose the best by- selecting the best delta, ie the
    parcellation i+1, where
    score_of_parcellation[i+1] - score_of_parcellation[i] is max.

    5)We then compute a list wich associate to each voxel the label of its
    parcellation.
    Lastly, we fit the given estimator with this parcel-based signal.

    Comments
    --------

    After n iterations we have (n+1) parcellations
    (n choosen, plus the root(s)).

    The selected parcellation of the iteration i has n_components+n parcels.

    The root n is the element (n - n_leaves) of the ward tree
    (we assume that the features are referenced by integers from 0 to n_leaves,
    and the ward tree, an (int * int) list represent the roots of the tree by
    its two children)

    Reference
    ---------

    http://hal.inria.fr/docs/00/58/92/01/PDF/supervised_clustering_vm_review.pdf
    """
    return BaseSupervisedClustering(estimator, n_iterations=n_iterations,
            connectivity=connectivity, copy=copy, cv=cv, n_jobs=n_jobs,
            verbose=verbose)


def SupervisedClusteringRegressor(
        estimator=BayesianRidge(fit_intercept=True, normalize=True),
        n_iterations=50, connectivity=None, copy=True,
        cv=None, n_jobs=1, verbose=0):
    """
    A regressor using hierarchical clustering to reduce features number

    Parameters
    ----------
    estimator : estimator, with methods fit and predict, optional
    default is BayesanRidge(fit_intercept=True, normalize=True)

    n_iterations : int
        default : 50
        number of iterations = max number of parcel we want

    connectivity : sparse matrix, optionnal
        connectivity matrix

    cv : scikits cross_validation model, optional
        the type of cross validation to use

    copy : bool, optional
        true if you want to use a copy of connectivity
        default is true

    n_jobs : int, optional
        number of cpu to use
        default is 1


    verbose : int, optional
        level of verbosity (0, 1 or 2)
        Default is 0


    Returns
    -------
    self


    Attributes
    ----------
    scores : float list
        the score of the best parcellation of each iteration
        the element i correspond to the score of the best parcellation
        at iteration i

    coef_ : the coefficients of the classifier fitted to the data
        (using the hierarchical clustering)

    labels_ : ndarray of shape (n_features)
        the parcellation chosen
        Two features have the same label if they are in the same parcel.
        The smaller label correspond to the smaller parcel

    Notes
    -----

    1) Firstly, ward tree is computed. Its a binary tree.
    The ward tree is represented by a list of pairsi
    Thus, the element i of this list represent the two children
    of the node i.
    /!\ The leaves are NOT represented in this list.

    2) Every cut of this tree gives a possible parcellation.
    Thus, we set the first parcellation as the root(s) of the tree, and,
    at each iteration, we construct a list of all the parcellations obtainable
    by cutting one of the parcels of the current parcellation in two.
    (we replace the parcel by its two children)

    3)We select the best parcellation of this list by cross validation
    This is the iteration parcellation.

    4)Finally, we obtain a list of al the parcellations choosen at
    each iteration.
    We choose the best by- selecting the best delta, ie the
    parcellation i+1, where
    score_of_parcellation[i+1] - score_of_parcellation[i] is max.

    5)We then compute a list wich associate to each voxel the label of its
    parcellation.
    Lastly, we fit the given estimator with this parcel-based signal.

    Comments
    --------

    After n iterations we have (n+1) parcellations
    (n choosen, plus the root(s)).

    The selected parcellation of the iteration i has n_components+n parcels.

    The root n is the element (n - n_leaves) of the ward tree
    (we assume that the features are referenced by integers from 0 to n_leaves,
    and the ward tree, an (int * int) list represent the roots of the tree by
    its two children)

    Reference
    ---------

    http://hal.inria.fr/docs/00/58/92/01/PDF/supervised_clustering_vm_review.pdf
    """
    return BaseSupervisedClustering(estimator, n_iterations=n_iterations,
            connectivity=connectivity, copy=copy, cv=cv, n_jobs=n_jobs,
            verbose=verbose)
