import numpy as np
from sklearn import covariance
from nilearn import signal
import pylab as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import ProbabilisticPCA
from sklearn.pipeline import Pipeline
from scipy.linalg.lapack import get_lapack_funcs
from matrix import untri
from sklearn import clone
#from .data import load


def covariance_matrix(series, gm_index, confounds=None):
    series = load(series)
    if series.shape[1] == 0:
        # Empty serie because region is empty
        return np.zeros((1, 1))

    if confounds is not None and np.ndim(confounds) == 3:
        confounds_ = []
        for c in confounds:
            c = load(c)
            if isinstance(c, basestring) or np.isfinite(c).all():
                confounds_.append(c)
        confounds = confounds_
    series = signal.clean(series, confounds=confounds)
    estimator = covariance.LedoitWolf()
    # Keep only gm regions
    series = series[:, np.array(gm_index)]
    try:
        estimator.fit(series)
        return estimator.covariance_, estimator.precision_
    except Exception as e:
        print e
        return np.eye(series.shape[1]), np.eye(series.shape[1])


def cov_embedding(covariances):
    """Returns for a list of matrices a list of transformed 
    matrices in matrix form, with 1. on the diagonal
    """
    ce = CovEmbedding()
    covariances = ce.fit_transform(covariances) # shape n(n+1)/2
    if covariances is None:
        return None
    return np.asarray([untri(c, k=0, fill=1.) for c in covariances]) # changed from k=1 to k=0 by Salma


def match_pairs(covariances, regions):
    # sym = regions[::-1]
    pass


def plot_cov(covariance, vmax=None, cmap='RdBu_r', title=None):
    if vmax is None:
        vmax = np.max(np.abs(covariance))
    pl.matshow(covariance, vmin=-vmax, vmax=vmax, cmap=cmap)
    if title is not None:
        pl.title(title)


def plot_covs(path, covariances, precisions, labels):
    """ Print a single matrix of mean covariance and mean precision.

    Parameters:
    -----------
        covariances: list of covariance matrices

        precisions: list of precision matrices

        labels: list of labels
    """
    assert(len(covariances) == len(precisions))
    assert(len(labels) == len(precisions))
    paths = []
    for l in np.unique(labels):
        mean_cov = np.mean(covariances[labels == l], axis=0)
        mean_prec = np.mean(precisions[labels == l], axis=0)
        np.fill_diagonal(mean_prec, 0.)

        plot_cov(mean_cov, vmax=1., title='Covariance label %d' % l)
        pl.colorbar()
        p = (path + '_cov.png') % l
        pl.savefig(p)
        pl.close()
        paths.append(p)

        plot_cov(mean_prec, title='Precision label %d' % l)
        pl.colorbar()
        p = (path + '_prec.png') % l
        pl.savefig(p)
        pl.close()
        paths.append(p)
    return paths


def my_stack(arrays):
    return np.concatenate([a[np.newaxis] for a in arrays])

# Bypass scipy for faster eigh (and dangerous: Nan will kill it)
my_eigh, = get_lapack_funcs(('syevr', ), np.zeros(1))


def inv_sqrtm(mat):
    """ Inverse of matrix square-root, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs / np.sqrt(vals), vecs.T)


def inv(mat):
    """ Inverse of matrix, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs / vals, vecs.T)


def sym_to_vec(sym):
    """ Returns the lower triangular part of
    sqrt(2) (sym + Id) + (1-sqrt(2)) / sqrt(2), shape n (n+1) /2, with n = 
    sym.shape[0]
    
    Parmaeters
    ==========
    sym: array
    """
    sym = np.copy(sym)
    sym -= np.eye(sym.shape[-1])
    # the sqrt(2) factor
    sym *= np.sqrt(2)
    sym += (1 - np.sqrt(2)) / np.sqrt(2) * np.diag(np.diag(sym))
    mask = np.tril(np.ones(sym.shape[-2:])).astype(np.bool)
    return sym[..., mask]


class CovEmbedding(BaseEstimator, TransformerMixin):
    """ Tranformer that returns the coefficients on a flat space to
    perform the analysis.    
    """

    def __init__(self, base_estimator=None, kind='tangent'):
        self.base_estimator = base_estimator 
        self.kind = kind
        if self.base_estimator == None:
            self.base_estimator_ = ...
        else:
            self.base_estimator_ = clone(base_estimator)

    def fit(self, covs):
        if self.kind == 'tangent':
            covs = my_stack(covs)
            #self.mean_cov = mean_cov = spd_manifold.log_mean(covs)
            mean_cov = mean_cov = np.mean(covs, axis=0)
            self.whitening = inv_sqrtm(mean_cov)
        return self

    def transform(self, covs):
        """Apply transform to covariances

        Parameters
        ----------
        covs: list of array        
            list of covariance matrices, shape (n_rois, n_rois)
            
        Returns
        -------
        list of array, transformed covariance matrices,
        shape (n_rois * (n_rois+1)/2,)        
        """        
        if self.kind == 'tangent':
            covs = [np.dot(np.dot(self.whitening, c), self.whitening)
                        for c in covs]
        elif self.kind == 'partial correlation':
            covs = [inv(g) for g in covs]
        return np.array([sym_to_vec(c) for c in covs])


if __name__ == '__main__':
    KIND = 'partial correlation'
    KIND = 'observation'
    KIND = 'tangent'

    # load the controls
    control_covs = np.load('/home/sb238920/CODE/NSAP/controls.npy')#controls.npy
#    control_covs = np.mean(control_covs, 1)
    n_controls, n_rois, _ = control_covs.shape

    # load the patients
    patient_covs = np.load('/home/sb238920/CODE/NSAP/patients.npy')
#    patient_covs = np.mean(patient_covs, 1)
    n_patients = len(patient_covs)
    patient_nbs = [4, 13, 18, 15, 16, 20, 22, 27, 30, 36]

    # 'test on control and patients'
    n_components = 0
    embedding = CovEmbedding(kind=KIND)
    pca = ProbabilisticPCA(n_components=n_components)
    model = Pipeline((('embedding', embedding),
                      ('pca', pca)))
    control_model = model.fit(control_covs)

    control_fits = control_model.score(control_covs)
    patient_fits = control_model.score(patient_covs)

    patient_fit_cv = np.zeros(n_patients)
    control_fit_cv = list()

    for n in range(n_controls):
        train = [control_covs[i]
                 for i in range(n_controls) if i != n]
        test = control_covs[n]
        control_model.fit(train)
        control_fit_cv.append(control_model.score([test]))
        patient_fit_cv += control_model.score(patient_covs)

    patient_fit_cv /= n_controls

    #pl.rcParams['text.usetex'] = True
    #pl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
    pl.figure(['tangent', 'observation',
               'partial correlation'].index(KIND),
            figsize=(2.5, 3))
    pl.clf()
    ax = pl.axes([.1, .15, .5, .7])
    pl.boxplot([control_fit_cv, patient_fit_cv], widths=.25)
    pl.plot(1.26 * np.ones(len(control_fit_cv)), control_fit_cv, '+k',
            markeredgewidth=1)
    pl.plot(2.26 * np.ones(len(patient_fits)),
            patient_fit_cv, '+k',
            markeredgewidth=1)
    pl.xticks((1.13, 2.13), ('controls', 'patients'), size=11)
    title = '%s%s \nspace' % (KIND[0].upper(), KIND[1:])
    pl.text(.05, .1, title,
            #'Partial\ncorrelation\nspace',
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='bottom',
            size=12)
    #pl.axis([0.7, 2.5, 401, 799])
    pl.xlim(.7, 2.5)
    #pl.ylim(401, 799)
    ax.yaxis.tick_right()
    pl.yticks(size=9)
    pl.ylabel('Log-likelihood', size=12)
    ax.yaxis.set_label_position('right')
    pl.title('N components=%i' % n_components)
    pl.draw()
