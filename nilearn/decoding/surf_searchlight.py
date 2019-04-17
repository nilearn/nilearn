"""
The searchlight is a widely used approach for the study of the
fine-grained patterns of information in fMRI analysis, in which
multivariate statistical relationships are iteratively tested in the
neighborhood of each location of a domain.
"""
# Authors : Vincent Michel (vm.michel@gmail.com)
#           Alexandre Gramfort (alexandre.gramfort@inria.fr)
#           Philippe Gervais (philippe.gervais@inria.fr)
#           Sylvain Takerkart (Sylvain.Takerkart@univ-amu.fr
#
# License: simplified BSD

import time
import sys
import warnings
from distutils.version import LooseVersion

import numpy as np

from sklearn import svm, neighbors
from sklearn.base import BaseEstimator

from .._utils.compat import _basestring
from .searchlight import search_light


ESTIMATOR_CATALOG = dict(svc=svm.LinearSVC, svr=svm.SVR)

def _apply_surfmask_and_get_affinity(mesh_coords, giimgs_data, radius, surfmask=None):
    """Utility function used for searchlight decoding.
    Parameters
    ----------
    mesh_coords: 2D array. n_vertex x 3
        Coordinates of the vertices of the mesh.
        We here recommand using a sphere.
    giimgs_data: 2D array, n_vertex x n_sample
        Surface textures to process, as e.g extracted by
        nilearn.surface.load_surf_data("*.gii")
    radius: float
        Indicates, in millimeters, the radius for the sphere around the seed.
    surfmask: 1D array, optional
        Mask to apply to regions before extracting signals.
        Should have two values, one of them being zero.
    """

    clf = neighbors.NearestNeighbors(radius=radius)
    clf.fit(mesh_coords)
    A = clf.radius_neighbors_graph(mesh_coords)
    A = A.tolil()

    if surfmask is not None:
        A = A[surfmask != 0,:]

    X = giimgs_data.T

    return X, A


##############################################################################
### Class for surface searchlight ############################################
##############################################################################
class SurfSearchLight(BaseEstimator):
    """Implements surface searchlight decoding.

    Parameters
    -----------
    mask_img : niimg
        boolean image giving location of voxels containing usable signals.

    process_mask_img : niimg, optional
        boolean image giving voxels on which searchlight should be
        computed.

    radius : float, optional
        radius of the searchlight surface disk, in millimeters. Defaults to 2.

    estimator : 'svr', 'svc', or an estimator object implementing 'fit'
        The object to use to fit the data

    n_jobs : int, optional. Default is -1.
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

    verbose : int, optional
        Verbosity level. Defaut is False

    Notes
    ------
    The searchlight [Kriegeskorte 06] is a widely used approach for the
    study of the fine-grained patterns of information in fMRI analysis.
    Its principle is relatively simple: a small group of neighboring
    features is extracted from the data, and the prediction function is
    instantiated on these features only. The resulting prediction
    accuracy is thus associated with all the features within the group,
    or only with the feature on the center. This yields a map of local
    fine-grained information, that can be used for assessing hypothesis
    on the local spatial layout of the neural code under investigation.

    Nikolaus Kriegeskorte, Rainer Goebel & Peter Bandettini.
    Information-based functional brain mapping.
    Proceedings of the National Academy of Sciences
    of the United States of America,
    vol. 103, no. 10, pages 3863-3868, March 2006
    """

    def __init__(self, orig_mesh, surfmask_tex, process_surfmask_tex=None, radius=2.,
                 estimator='svc',
                 n_jobs=1, scoring=None, cv=None,
                 verbose=0):
        self.orig_mesh = orig_mesh
        self.surfmask_tex = surfmask_tex
        self.process_surfmask_tex = process_surfmask_tex
        self.radius = radius
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose

    def fit(self, giimgs_data, y, groups=None):
        """Fit the searchlight

        Parameters
        ----------
        giimgs_data: 2D array, n_vertex x n_sample
            Surface textures to process, as e.g extracted by
            nilearn.surface.load_surf_data("*.gii")

        y : 1D array-like
            Target variable to predict. Must have exactly as many elements as
            1D textures in giimgs.

        groups : array-like, optional
            group label for each sample for cross validation. Must have
            exactly as many elements as 3D images in img. default None
            NOTE: will have no effect for scikit learn < 0.18

        Attributes
        ----------
        `scores_` : numpy.ndarray
            search_light scores. Same shape as input parameter
            process_mask_img.
        """

        mesh_coords = self.orig_mesh[0]
        X, A = _apply_surfmask_and_get_affinity(mesh_coords, giimgs_data,
                                                self.radius,
                                                surfmask = self.process_surfmask_tex)

        estimator = self.estimator
        if isinstance(estimator, _basestring):
            estimator = ESTIMATOR_CATALOG[estimator]()

        # scores is an 1D array of CV scores with length equals to the number
        # of vertices in the processing mask
        scores = search_light(X, y, estimator, A, groups,
                              self.scoring, self.cv, self.n_jobs,
                              self.verbose)

        scores_fullbrain = np.zeros(self.process_surfmask_tex.shape)
        scores_fullbrain[self.process_surfmask_tex!=0] = scores
        self.scores_ = scores_fullbrain

        return self
