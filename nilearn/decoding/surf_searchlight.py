"""
The searchlight is a widely used approach for the study of the
fine-grained patterns of information in fMRI analysis, in which
multivariate statistical relationships are iteratively tested in the
neighborhood of each location of a domain.
"""
# Authors : Vincent Michel (vm.michel@gmail.com)
#           Alexandre Gramfort (alexandre.gramfort@inria.fr)
#           Philippe Gervais (philippe.gervais@inria.fr)
#
# License: simplified BSD

import time
import sys
import warnings
from distutils.version import LooseVersion

import numpy as np

from sklearn import svm
from sklearn.base import BaseEstimator


from .. import surf_masking
from ..input_data.gifti_spheres_masker import _apply_surfmask_and_get_affinity
from .._utils.compat import _basestring
from .searchlight import search_light


ESTIMATOR_CATALOG = dict(svc=svm.LinearSVC, svr=svm.SVR)



##############################################################################
### Class for search_light ###################################################
##############################################################################
class SurfSearchLight(BaseEstimator):
    """Implement surf_search_light analysis using an arbitrary type of classifier.

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

    def fit(self, giimgs, y, groups=None):
        """Fit the searchlight

        Parameters
        ----------
        giimgs : giimg
            2D texture (nbr of nodes x nbr of samples).

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

        # Get world coordinates
        mesh_coords = surf_masking._get_mesh_coords(self.orig_mesh)
        surfmask = surf_masking._load_surfmask_tex(self.surfmask_tex)
        surfmask_coords = mesh_coords[surfmask,:]
        process_surfmask = surf_masking._load_surfmask_tex(self.process_surfmask_tex)

        X, A = _apply_surfmask_and_get_affinity(
            surfmask_coords, giimgs, mesh_coords, self.radius, True,
            process_surfmask=process_surfmask)
        
        print(X.shape)

        estimator = self.estimator
        if isinstance(estimator, _basestring):
            estimator = ESTIMATOR_CATALOG[estimator]()

        # scores is an 1D array of CV scores with length equals to the number
        # of voxels in processing mask (columns in process_mask)
        scores = search_light(X, y, estimator, A, groups,
                              self.scoring, self.cv, self.n_jobs,
                              self.verbose)
                                   
        print(scores.shape)
                                   
        scores_3D = np.zeros(process_surfmask.shape)
        scores_3D[process_surfmask] = scores
        self.scores_ = scores_3D
        return self
