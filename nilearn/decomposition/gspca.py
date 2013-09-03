"""
Sparse recovery of group maps using dictionary learning
"""
from joblib import Memory

import numpy as np

import base_gspca
from base_gspca import BaseGSPCAModel

from ..proximal.operators import prox_l1


def l1(v):
    return np.sum(np.abs(v))


def prox_l1_(V, alpha):
    V_ = prox_l1(V, alpha)
    V_norms = l1(V_)

    return V_, V_norms


def group_dictionary(Ys, n_atoms, alpha, mu, maxit=100, minit=3, tol=1e-8,
            n_jobs=1, Us_init=None, Vs_init=None, V_init=None,
            n_shuffle_subjects=None,
            callback=None, verbose=False, copy=True, non_penalized=None):

    base_gspca.group_dictionary(Ys, n_atoms, prox_l1_, alpha, mu, maxit,
                                      minit, tol,
                                      n_jobs, Us_init, Vs_init, V_init,
                                      n_shuffle_subjects, callback, verbose,
                                      copy, non_penalized)

    return Us, Vs, V, E


###############################################################################
class GSPCAModel(BaseGSPCAModel):
    """ Learn our model using Group sparse PCA
    """

    def __init__(self, n_components, alpha=.5, mu='auto',
                        mem=Memory(cachedir=None),
                        non_penalized=None,
                        do_ica=True,
                        tol=1e-4,
                        max_iter=300,
                        n_shuffle_subjects=None,
                        n_jobs=1,
                        verbose=False):
        super(GSPCAModel, self).__init__(
            n_components,
            prox_l1_,
            alpha=alpha,
            mu=mu,
            mem=mem,
            non_penalized=non_penalized,
            do_ica=do_ica,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            n_shuffle_subjects=n_shuffle_subjects,
            n_jobs=n_jobs
        )


if __name__ == '__main__':

    # Generate toy data
    n_atoms = 2
    n_time_pts = 3
    n_subjects = 2
    n_samples = n_subjects * n_time_pts
    img_sz = (10, 10)
    n_features = img_sz[0] * img_sz[1]

    np.random.seed(0)
    Us = np.random.randn(n_subjects, n_time_pts, n_atoms)
    V = np.random.randn(n_atoms, n_features)
    Vs = [V, ] * n_subjects

    centers = [(3, 3), (6, 7), (8, 1)]
    sz = [1, 2, 1]
    for k in range(n_atoms):
        img = np.zeros(img_sz)
        xmin, xmax = centers[k][0] - sz[k], centers[k][0] + sz[k]
        ymin, ymax = centers[k][1] - sz[k], centers[k][1] + sz[k]
        img[xmin:xmax][:, ymin:ymax] = 1.0
        V[k, :] = img.ravel()

    # Y is defined by : Y = UV + noise
    Ys = list()
    for U_, V_ in zip(Us, Vs):
        Y_ = np.dot(U_, V_)
        Y_ += 0.1 * np.random.randn(Y_.shape[0], Y_.shape[1])  # Add noise
        Ys.append(Y_)

    # Estimate U,V
    alpha = 0.5
    mu = 1
    Us_estimated, Vs_estimated, V_estimated, E = group_dictionary(Ys,
                                n_atoms, alpha, mu, maxit=100,
                                n_jobs=1)

    # View results
    import pylab as pl
    pl.close('all')

    for k in range(n_atoms):
        pl.matshow(np.reshape(V_estimated[k, :], img_sz))
        pl.title('Atom %d' % k)
        pl.colorbar()

    pl.figure()
    pl.plot(E)
    pl.xlabel('Iteration')
    pl.ylabel('Cost function')
    pl.show()
