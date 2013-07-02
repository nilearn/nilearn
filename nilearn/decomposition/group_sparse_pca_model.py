"""
Sparse recovery of group maps using dictionary learning
"""
import sys

import numpy as np


from sklearn.cross_validation import KFold

from parietal.learn.proximal.operators import prox_l1
import base_gspca_model
from base_gspca_model import BaseGSPCAModel

from joblib import Memory


def l1(v):
    return np.sum(np.abs(v))


def prox_l1_(V, alpha):
    V_ = prox_l1(V, alpha)
    V_norms = l1(V_)

    return V_, V_norms


def group_dictionary(Ys, n_atoms, alpha, mu, maxit=100, minit=3, tol=1e-8,
            n_jobs=1, Us_init=None, Vs_init=None, V_init=None,
            Vs_method='solve', Vs_warm_restart=False,
            n_shuffle_subjects=None,
            callback=None, verbose=False, copy=True, non_penalized=None):

    base_gspca_model.group_dictionary(Ys, n_atoms, prox_l1_, alpha, mu, maxit,
                                      minit, tol,
                                      n_jobs, Us_init, Vs_init, V_init,
                                      Vs_method, Vs_warm_restart,
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
                        Vs_method='solve',
                        max_iter=300,
                        Vs_warm_restart=None,
                        V_norm=(l1, prox_l1),
                        V_outer_mask=False,
                        n_shuffle_subjects=None,
                        n_jobs=1,
                        verbose=False):
        self.n_components = n_components
        self.alpha = alpha
        self.mu = mu
        self.mem = mem
        self.max_iter = max_iter
        self.verbose = verbose
        self.non_penalized = non_penalized
        self.do_ica = do_ica
        self.tol = tol
        self.Vs_method = Vs_method
        if Vs_warm_restart is False:
            Vs_warm_restart = None
        self.Vs_warm_restart = Vs_warm_restart
        self.V_norm = V_norm
        self.V_outer_mask = V_outer_mask
        self.n_shuffle_subjects = n_shuffle_subjects
        self.n_jobs = n_jobs
        # We store the prox so we don't have to override fit
        self.prox = prox_l1_


###############################################################################
class GSPCAModelCV(GSPCAModel):
    """ Learn our model using GSPCA, sets the alpha by CV.
    """

    def __init__(self, n_components,
                    alphas=[.5, 1, 2, 5, 7, 10, 20, 50,
                            80],
                    mu='auto',
                    mem=Memory(cachedir=None),
                    verbose=False,
                    non_penalized=None,
                    do_ica=True,
                    tol=1e-4,
                    stratified_cv=False):
        self.n_components = n_components
        self.alphas = alphas
        self.mu = mu
        self.mem = mem
        self.verbose = verbose
        self.stratified_cv = stratified_cv
        self.non_penalized = non_penalized
        self.do_ica = do_ica
        self.tol = tol

    def fit(self, data, cv=None, refit=True,
                    n_jobs=-1, **params):
        self.set_params(**params)
        if self.stratified_cv:
            n_samples = [len(d) for d in data]
            if cv is None:
                cv = [KFold(n, n_folds=3) for n in n_samples]
                cv = zip(*cv)
                cv = [zip(*cv_) for cv_ in cv]
        else:
            n_samples = len(data)
            if cv is None:
                cv = KFold(n_samples, n_folds=3)
        scores = dict()
        if self.verbose == 1:
            print '[%s]' % self.__class__.__name__,
        for train, test in cv:
            V_init = None
            if self.verbose == 1:
                sys.stdout.write('_')
                sys.stdout.flush()
            gspca_model = GSPCAModel(n_components=self.n_components,
                                 mu=self.mu,
                                 mem=self.mem,
                                 verbose=max(self.verbose - 1, 0),
                                 do_ica=self.do_ica,
                                 tol=self.tol,
                                 non_penalized=self.non_penalized)
            if self.stratified_cv:
                if isinstance(train, tuple):
                    train_data = [d[t] for d, t in zip(data, train)]
                    test_data = [d[t] for d, t in zip(data, test)]
                else:
                    train_data = [d[train] for d in data]
                    test_data = [d[test] for d in data]
            else:
                train_data = data[train]
                test_data = data[test]
            for alpha in self.alphas:
                if self.verbose == 1:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                elif self.verbose:
                    print 'alpha=%s ' % alpha,
                gspca_model.alpha = alpha
                gspca_model.fit(train_data, V_init=V_init,
                              n_jobs=n_jobs)

                this_scores = scores.get(alpha, [])
                this_scores.append(gspca_model.score(test_data))
                scores[alpha] = this_scores
                V_init = gspca_model.maps_
                # need to pad with zeros if the number of maps is
                # smaller than the number of components
                if V_init is not None and len(V_init) < self.n_components:
                    V_init = np.resize(V_init,
                                (self.n_components, train_data[0].shape[-1]))
        if self.verbose == 1:
            # A line return
            print ""
        self.grid_points_scores = scores
        best_score = -np.inf
        best_alpha = np.nan
        for alpha, this_scores in scores.iteritems():
            if (np.all(np.isfinite(this_scores))
                            and np.mean(this_scores) > best_score):
                best_alpha = alpha
                best_score = np.mean(this_scores)
        self.best_alpha_ = best_alpha
        self.alpha = best_alpha
        if refit:
            GSPCAModel.fit(self, data, n_jobs=n_jobs)
        return self


###############################################################################
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
