"""
Sparse recovery of group maps using dictionary learning
"""
import time
import sys
from functools import partial

import numpy as np
from scipy import linalg

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator
from sklearn.utils.extmath import randomized_svd

from sklearn.decomposition.dict_learning import _update_dict
from ica_model import ICAModel

from base_model import learn_time_series

from joblib import Memory, MemorizedResult, NotMemorizedResult


def _load(data):
    if (isinstance(data, MemorizedResult) or
            isinstance(data, NotMemorizedResult)):
        return data.get()
    return data


def _update_Vs(Ys_, Us_, V, mu, column_wise=False):
    n_samples, n_atoms = Us_.shape
    # XXX: Need to check with Vincent or Alex on the scaling of mu
    # mu = mu / n_samples

    # Not exactly a Ridge: must change variables to account for V: we
    # apply a ridge on dV = Vs - V, thus Y -> this_Y = Y - np.dot(Us_, V)

    solver = partial(linalg.solve, sym_pos=True,
                     overwrite_a=True, overwrite_b=True)

    if n_atoms < n_samples:
        # w = inv(X^t X + alpha*Id) * X.T y
        cov = np.dot(Us_.T, Us_)
        this_Y = - np.dot(cov, V)
        this_Y += np.dot(Us_.T, Ys_)
        cov.flat[::n_atoms + 1] += mu
        if not column_wise:
            dVs = solver(cov, this_Y)
        else:
            coefs = np.empty((n_atoms, this_Y.shape[1]))
            for i in range(this_Y.shape[1]):
                coefs[:, i] = solver(cov, this_Y[:, i])[0]
            dVs = coefs
    else:
        # w = X.T * inv(X X^t + alpha*Id) y
        this_Y = - np.dot(Us_, V)
        this_Y += Ys_
        cov = np.dot(Us_, Us_.T)
        cov.flat[::n_samples + 1] += mu
        if not column_wise:
            dVs = solver(cov, this_Y)
            dVs = np.dot(Us_.T, dVs)
        else:
            coefs = np.empty((n_atoms, this_Y.shape[1]))
            for i in range(this_Y.shape[1]):
                coefs[:, i] = solver(cov, this_Y[:, i])[0]
            dVs = np.dot(Us_.t, coefs)
    return V + dVs, (dVs ** 2).sum()


def _update_V(Vs, prox, alpha, non_penalized=None):
    V_group = np.mean(Vs, axis=0)
    V_group[:non_penalized], V_norms = prox(V_group[:non_penalized], alpha)
    return V_group, V_norms


def _update_Vs_Us(Ys_, Us_, V, mu, id_subj=None, verbose=0):
    # Update Vs
    Ys__ = _load(Ys_)
    Vs_, residuals_dV_ = _update_Vs(Ys__, Us_, V, mu=mu)

    # Update Us
    Us_, r2_ = _update_dict(Us_, Ys__, Vs_, return_r2=True, verbose=verbose)
    del Ys__
    return Vs_, residuals_dV_, Us_, r2_


def group_dictionary(Ys, n_atoms, prox, alpha, mu, maxit=100, minit=3,
            tol=1e-8, n_jobs=1, Us_init=None, Vs_init=None, V_init=None,
            n_shuffle_subjects=None,
            callback=None, verbose=False, copy=True, non_penalized=None):

    # Avoid integer division problems
    alpha = float(alpha)
    mu = float(mu)
    n_group = len(Ys)
    alpha *= n_group

    t0 = 0

    if n_shuffle_subjects is not None:
        if n_shuffle_subjects * 2 > n_group:
            raise ValueError('Number of shuffled subjects must be lesser than '
                             'number of subjects div 2')

    def cost_function():
        # r2 is the sum of squares residuals
        return (.5 * (sum(r2) + mu * sum(residuals_dV))
                + alpha * np.sum(V_norms))

    # left to understand the behavior of the preceding function (and test it)
    def full_cost_function():
        # The cost function not using precomputed quantities
        y_uv_vs_v = 0
        for Ys_, Us_, Vs_ in zip(Ys, Us, Vs):
            Ys__ = _load(Ys_)
            y_uv_vs_v += (np.sum((Ys__ - np.dot(Us_, Vs_)) ** 2)
                          + mu * np.sum((Vs_ - V) ** 2))
            del Ys__
        return .5 * y_uv_vs_v + alpha * np.sum(V_norms)

    # There are 6 possible configurations:
    # - nothing is given, Vs_init is computed using PCA
    # - Vs_init is given (or computed above), V is computed using PCA
    # - Vs_init and V_init are given or computed, then U_init is computed
    #   using least squares method
    # - Us_init and Vs_init are given, then V is computed during the first
    #   step of the algorithm (update_V)
    # - Us_init and V_init are given, Vs_init is computed thanks to update_Vs
    # - everything is given, there is nothing to do

    residuals_dV = np.zeros(n_group)

    if Us_init is None:
        # Either Vs_init and V_init are set, or none of them
        if Vs_init is None:
            Vs = []
            for Ys_ in Ys:
                Ys__ = _load(Ys_)
                Vs.append(linalg.svd(Ys__, full_matrices=False)[-1][:n_atoms])
                del Ys__
            V = linalg.svd(np.concatenate(Vs, axis=0),
                           full_matrices=False)[-1][:n_atoms]

        # Learn Us, and relearn V to have the right scaling on U
        Us = list()
        Ss = list()
        for Ys_, Vs_ in zip(Ys, Vs):
            Ys__ = _load(Ys_)
            U = linalg.lstsq(Vs_.T, Ys__.T)[0]
            del Ys__
            S = np.sqrt((U ** 2).sum(axis=1))
            U /= S[:, np.newaxis]
            Us.append(U.T)
            Ss.append(S)
        V *= np.mean(Ss, axis=0)[:, np.newaxis]
        Vs = Parallel(n_jobs=n_jobs)(
                delayed(_update_Vs)(Ys_, Us_, V, mu=mu)
                for Ys_, Us_ in zip(Ys, Us))
        Vs, residuals_dV = zip(*Vs)
    else:
        # If Vs_init is None, then V_init is not None and we can compute it
        Us = Us_init
        if Vs_init is None:
            V = V_init
            Vs = Parallel(n_jobs=n_jobs)(
                    delayed(_update_Vs)(Ys_, Us_, V, mu=mu)
                    for Ys_, Us_ in zip(Ys, Us))
            Vs, residuals_dV = zip(*Vs)
        # V may be None here. We do not care because update_V is called right
        # after that.

    E = [np.inf]
    Vs = np.asarray(Vs)
    V_norms = np.zeros(n_atoms)
    #residuals_dV is already set
    r2 = np.zeros(n_group)
    dVs = 1e-2
    # For stochastic mode: used to make a last iteration using all subjects
    last_iteration = False

    if n_shuffle_subjects is not None:
        # Create the set of unseen subjects. These will have to be treated
        # specifically for update_V
        unseen = np.ones(n_group, dtype=bool)
        selected = []

    for ii in xrange(maxit):

        if hasattr(prox, 'set_V'):
            prox.set_V(V)
        if hasattr(prox, 'set_dVs'):
            prox.set_dVs(dVs)
        t0_ = time.time()
        V, V_norms = _update_V(Vs, prox, alpha / (mu * n_group),
                      non_penalized=non_penalized)
        t0 += time.time() - t0_
        dt = t0

        # Group subject updates
        current_cost = cost_function()
        if verbose > 1:
            print ("Iteration % 3i.1 "
                "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)" %
                    (ii, dt, dt / 60, current_cost))

        if n_shuffle_subjects is not None:
            # We select some random subjects
            ys = []
            us = []
            prev = selected
            selected = []
            for i in range(n_shuffle_subjects):
                r = None
                while r in selected or r in prev or r is None:
                    r = np.random.randint(n_group)
                ys.append(Ys[r])
                us.append(Us[r])
                selected.append(r)
                unseen[r] = False
            if verbose > 1:
                print "Subjects selected: ", str(selected)
            # We attribute group maps to unseen subjects
            Vs[unseen] = V
        else:
            ys = Ys
            us = Us

        t0_ = time.time()
        VsUs = Parallel(n_jobs=n_jobs)(delayed(_update_Vs_Us)
                (Ys_, Us_, V, mu, return_r2=True, id_subj=i, verbose=verbose)
                for i, (Ys_, Us_, Vs_warm_restart) in
                    enumerate(zip(ys, us)))
        t0 += time.time() - t0_
        dt = t0

        if Vs_warm_restart is None:
            Vs_, residuals_dV_, Us_, r2_ = zip(*VsUs)
        else:
            Vs_, residuals_dV_, Vs_warm_restart, Us_, r2_ = zip(*VsUs)

        if n_shuffle_subjects is not None:
            dVs = 0.
            for j, i in enumerate(selected):
                Us[i] = Us_[j]
                dVs += np.sum(((Vs_[j] - Vs[i]) / (n_group * mu * 2.)) ** 2)
                Vs[i] = Vs_[j]
                residuals_dV[i] = residuals_dV_[j]
                r2[i] = r2_[j]
        else:
            Vs_ = np.asarray(Vs_)
            dVs = np.sum(((Vs_ - Vs) / (n_group * mu * 2.)) ** 2)
            Vs = Vs_
            Us = Us_
            residuals_dV = residuals_dV_
            r2 = r2_

        if verbose > 2:
            print "Delta Vs: %f" % dVs

        current_cost = cost_function()
        E.append(current_cost)
        if verbose == 1:
            sys.stdout.write('.')
            sys.stdout.flush()
        elif verbose:
            print ("Iteration % 3i.2 "
                "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)" %
                    (ii, dt, dt / 60, current_cost))

        dE = E[-2] - E[-1]
        # We tolerate small increases in energy due to numerical
        # error, but not much
        #assert(dE > -tol * abs(E[-1]))
        if dE < 0:
            if verbose == 1:
                sys.stdout.write('!')
            elif verbose:
                print 'Energy increased by %.2f* tol' % (-dE / (tol * E[-1]))

        if last_iteration or dE < tol * abs(E[-1]):
            if n_shuffle_subjects is not None:
                last_iteration = True
                n_shuffle_subjects = None
                continue
            if verbose == 1:
                print ""
            elif verbose:
                print "--- Convergence reached after %d iterations" % ii
            if ii >= minit:
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())

    return Us, Vs, V, E


# Gave me negative results on big dataset...
def estimate_group_regularization(data, n_components,
                                mem=Memory(cachedir=None)):
    """ Rough estimate of the regularization at the group level from the
        ratio of the group variance and the subject variance
    """
    pcas = list()
    n_time_pts = float(len(data[0]))
    n_subjects = len(data)
    assert n_components < n_time_pts
    singular_values = list()
    total_variance = 0
    subset_size = 150
    for subject_data in data:
        subset_size = min(subset_size, subject_data.shape[0])
    for subject_data in data:
        total_variance += np.sum(subject_data ** 2) / (n_subjects * n_time_pts)
        U, S = mem.cache(linalg.svd)(subject_data.T,
                                         full_matrices=0)[:2]
        pcas.append(U.T[:subset_size] * S[:subset_size, np.newaxis])
        del U
        singular_values.append(np.sum(S[n_components + 1:] ** 2 / n_time_pts))
    subject_variance = np.mean(singular_values)
    group_variance = randomized_svd(np.concatenate(pcas, axis=1),
                                    n_components)[1]
    group_variance = total_variance \
                    - np.sum(group_variance ** 2) / (n_subjects * n_time_pts)
    n_components = float(n_components)
    out = n_components / n_time_pts * \
            1. / (group_variance / subject_variance - 1)
    return out


###############################################################################
class BaseGSPCAModel(BaseEstimator):
    """ Learn our model using Group sparse PCA
    """

    def __init__(self, n_components, prox, alpha=.5, mu='auto',
                        mem=Memory(cachedir=None),
                        non_penalized=None,
                        do_ica=True,
                        tol=1e-4,
                        Vs_method='solve',
                        max_iter=300,
                        Vs_warm_restart=None,
                        n_shuffle_subjects=None,
                        n_jobs=1,
                        verbose=False):
        self.n_components = n_components
        self.prox = prox
        self.alpha = alpha
        self.mu = mu
        self.mem = mem
        self.max_iter = max_iter
        self.verbose = verbose
        self.non_penalized = non_penalized
        self.do_ica = do_ica
        self.tol = tol
        self.n_shuffle_subjects = n_shuffle_subjects
        self.n_jobs = n_jobs

    def fit(self, data, V_init=None, **params):

        mu = self.mu
        if V_init is None:
            data_ = [_load(d) for d in data]
            ica_model = ICAModel(self.n_components,
                                    store_pca=True,
                                    kurtosis_thr=0,
                                    maps_only=True,
                                    mem=self.mem).fit(data_)
            del data_
            if self.do_ica:
                V_init = ica_model.maps_
            else:
                # We compute ICA even if we do not need it, this is not very
                # efficient...
                V_init = ica_model.pca_.T[:self.n_components]
            del ica_model
            if self.verbose > 1:
                print '[%s] Computed V_init' % self.__class__.__name__
        Us = list()
        Ss = list()
        if self.verbose > 1:
            print '[%s] Finished learning U_init' % self.__class__.__name__

        if mu == 'auto':
            mu = self.mem.cache(estimate_group_regularization,
                                ignore=['mem'])(data,
                                self.n_components, self.mem)
            self.mu_ = mu

        for Y_ in data:
            # Relearn U, V to have the right scaling on U
            Y__ = _load(Y_)
            U = linalg.lstsq(V_init.T, Y__.T)[0]
            del Y__
            S = np.sqrt((U ** 2).sum(axis=1))
            S[S == 0] = 1
            U /= S[:, np.newaxis]
            Ss.append(S)
            Us.append(U.T)
        V_init = (V_init * np.mean(Ss, axis=0)[:, np.newaxis])
        if self.verbose > 1:
            print '[%s] Finished learning V_init' % self.__class__.__name__

        _, maps_, group_maps_, E = group_dictionary(data,
                                    self.n_components,
                                    prox=self.prox,
                                    alpha=self.alpha, mu=mu,
                                    maxit=self.max_iter,
                                    Us_init=Us, V_init=V_init,
                                    n_shuffle_subjects=self.n_shuffle_subjects,
                                    n_jobs=self.n_jobs, tol=self.tol,
                                    verbose=self.verbose,
                                    copy=False,
                                    non_penalized=self.non_penalized,
                                    **params)
        # Remove any map with only zero values:
        mask = group_maps_.ptp(axis=1) != 0
        group_maps_ = group_maps_[mask]
        maps_ = [m[mask] for m in maps_]
        self.E = E
        del mask
        if not len(group_maps_):
            # All maps are zero
            self.cov_ = np.array([[]], dtype=np.float)
            self.maps_ = group_maps_
            self.subject_maps_ = maps_
            return
        self.maps_ = group_maps_
        self.subject_maps_ = maps_

        # Flip sign to always have positive features
        for index, map_ in enumerate(self.maps_):
            mask = map_ > 0
            if map_[mask].sum() < - map_[np.logical_not(mask)].sum():
                map_ *= -1
                for d in self.subject_maps_:
                    d[index] *= -1

        # Relearn U, V to have the right scaling on U
        residuals = list()
        covs = list()
        s = 0
        for Y_, m in zip(data, self.subject_maps_):
            # XXX: would like to parallelize this loop, but it might blow
            # memory, this could be done by returning only the scaling
            # the covariance and the residuals
            Y__ = _load(Y_)
            u, this_residuals = learn_time_series(m, Y__)
            del Y__
            n_samples = u.shape[1]
            this_s = np.sqrt((u ** 2).sum(axis=1) / n_samples)
            u /= this_s[:, np.newaxis]
            m *= this_s[:, np.newaxis]
            this_residuals.fill(np.sqrt(this_residuals.mean()))
            residuals.append(this_residuals)
            s += this_s
            covs.append(1. / n_samples * np.dot(u, u.T))
        s /= len(data)
        self.residuals_ = residuals
        del this_residuals, u, d, m
        self.maps_ *= s[:, np.newaxis]
        self.cov_ = np.mean(covs, axis=0)
        self.covs_ = covs

        return self
