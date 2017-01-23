"""
Base class for multi subject dictionary learning (MSDL)
"""
import time
import sys
from functools import partial

import numpy as np
from scipy import linalg

from joblib import Parallel, delayed, Memory

from sklearn.base import BaseEstimator
from sklearn.utils.extmath import randomized_svd

from sklearn.linear_model import ridge_regression
from sklearn.decomposition.dict_learning import _update_dict
from nilearn.decomposition import CanICA
from nilearn.masking import unmask


###############################################################################

def _learn_Us(Vs, Ys, alpha=.01):
    """Ridge regression to learn time series

    We do a ridge regression with a very small regularisation, this
    corresponds to a least square with control on the conditioning

    Parameters
    ----------
    Vs: array-like, shape = [n_maps, n_voxels]
        Subject maps.

    Ys: array-like, shape = [n_timepoints, n_voxels]
        Subject data

    alpha: float
        The amount of l2 penalization

    Returns
    -------
    Us: array-like, shape = [n_maps, n_timepoints]
        Map-specific time series.
    """
    Us = ridge_regression(Vs.T, Ys.T, alpha).T
    residuals = np.dot(Us.T, Vs)
    residuals -= Ys
    residuals **= 2
    residuals = np.mean(residuals, axis=0)
    return Us, residuals


def _update_Vs(Ys_, Us_, V, mu, column_wise=False):
    n_samples, n_atoms = Us_.shape

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
    return V_group, V_norms, [((Vs_ - V_group) ** 2).sum() for Vs_ in Vs]


from joblib.memory import MemorizedResult


def _get(cached_data):
    if isinstance(cached_data, MemorizedResult):
        return cached_data.get(), True
    return cached_data, False


def _update_Vs_Us(Ys, Us, V, mu, verbose=0):
    tYs, is_shelved = _get(Ys)

    # Update Vs
    Vs_, residuals_dV_ = _update_Vs(tYs, Us, V, mu=mu)

    # Update Us
    Us_, r2_ = _update_dict(Us, tYs, Vs_, return_r2=True, verbose=verbose)

    # If subject data is shelved, we dispose of it
    if is_shelved:
        del tYs
    return Vs_, residuals_dV_, Us_, r2_


def group_dictionary(Ys, n_atoms, prox, alpha, mu, maxit=100, minit=3,
            tol=1e-8, n_jobs=1, Us_init=None, Vs_init=None, V_init=None,
            n_shuffle_subjects=None,
            callback=None, verbose=False, copy=True, non_penalized=None):

    if verbose > 1:
        print('[MSDL] Entering group dictionary estimation')

    # Avoid integer division problems
    alpha = float(alpha)
    mu = float(mu)
    n_group = len(Ys)

    t0 = 0

    if n_shuffle_subjects is not None:
        if n_shuffle_subjects * 2 > n_group:
            raise ValueError('Number of shuffled subjects must be lesser than '
                             'number of subjects div 2')

    def cost_function():
        # r2 is the sum of squares residuals
        return ((sum(r2) + mu * sum(residuals_dV)) / (n_group * 2.)
                + mu * alpha * np.sum(V_norms))

    # There are 6 possible configurations:
    # - nothing is given, Vs_init is computed using PCA
    # - Vs_init is given (or computed above), V is computed using PCA
    # - Vs_init and V_init are given or computed, then U_init is computed
    #   using least squares method
    # - Us_init and Vs_init are given, then V is computed during the first
    #   step of the algorithm (update_V)
    # - Us_init and V_init are given, Vs_init is computed thanks to update_Vs
    # - everything is given, there is nothing to do

    if Us_init is None:
        # Either Vs_init and V_init are set, or none of them
        if Vs_init is None:
            if verbose > 1:
                print('[MSDL] Initialization of subject maps (Vs_init) using PCA')
            Vs = []
            for Ys_ in Ys:
                Ys__ = Ys_
                Vs.append(linalg.svd(Ys__, full_matrices=False)[-1][:n_atoms])
                del Ys__
            V = linalg.svd(np.concatenate(Vs, axis=0),
                           full_matrices=False)[-1][:n_atoms]

        # Learn Us, and relearn V to have the right scaling on U
        if verbose > 1:
            print('[MSDL] Initialization of subject timeseries (Us)')
        Us = list()
        Ss = list()
        for Ys_, Vs_ in zip(Ys, Vs):
            Ys__ = Ys_
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
            if verbose > 1:
                print('[MSDL] Estimation of initialization subject maps')
            V = V_init
            Vs = Parallel(n_jobs=1)(
                    delayed(_update_Vs)(Ys_, Us_, V, mu=mu)
                    for Ys_, Us_ in zip(Ys, Us))
            Vs, residuals_dV = zip(*Vs)
        # V may be None here. We do not care because update_V is called right
        # after that.

    E = [np.inf]
    Vs = np.asarray(Vs)
    V_norms = np.zeros(n_atoms)
    r2 = np.zeros(n_group)
    #dVs = 1e-2
    # For stochastic mode: used to make a last iteration using all subjects
    last_iteration = False

    if n_shuffle_subjects is not None:
        # Create the set of unseen subjects. These will have to be treated
        # specifically for update_V
        unseen = np.ones(n_group, dtype=bool)
        selected = []

    for ii in range(maxit):

        if verbose > 2:
            prev_V_norms = sum(V_norms)
            prev_residuals_dV = sum(residuals_dV)
        t0_ = time.time()
        V, V_norms, residuals_dV = _update_V(Vs, prox, alpha,
                      non_penalized=non_penalized)
        t0 += time.time() - t0_
        dt = t0

        # Group subject updates
        current_cost = cost_function()
        if verbose > 1:
            print("Iteration % 3i.1 "
                "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)" %
                    (ii, dt, dt / 60, current_cost))
            if verbose > 2:
                print("Norms %7.3f -> %7.3f" % (prev_V_norms, sum(V_norms)))
                print("Residuals %7.3f -> %7.3f" % (prev_residuals_dV,
                        sum(residuals_dV)))
                print("Total %7.3f -> %7.3f" % (
                        prev_residuals_dV / (n_group * 2.)
                            + alpha * prev_V_norms,
                        sum(residuals_dV) / (n_group * 2.)
                            + alpha * np.sum(V_norms)))

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
                print("Subjects selected: ", str(selected))
            # We attribute group maps to unseen subjects
            Vs[unseen] = V
        else:
            ys = Ys
            us = Us

        t0_ = time.time()
        VsUs = Parallel(n_jobs=n_jobs)(delayed(_update_Vs_Us)
                (ys_, us_, V, mu, verbose=verbose)
                for (ys_, us_) in zip(ys, us))

        t0 += time.time() - t0_
        dt = t0

        Vs_, residuals_dV_, Us_, r2_ = zip(*VsUs)

        if n_shuffle_subjects is not None:
            #dVs = 0.
            for j, i in enumerate(selected):
                Us[i] = Us_[j]
                #dVs += np.sum(((Vs_[j] - Vs[i]) / (n_group * mu * 2.)) ** 2)
                Vs[i] = Vs_[j]
                residuals_dV[i] = residuals_dV_[j]
                r2[i] = r2_[j]
        else:
            Vs_ = np.asarray(Vs_)
            # Do that inplace to save memory
            #Vs -= Vs_
            #dVs = np.sum((Vs / (n_group * mu * 2.)) ** 2)
            Vs = Vs_
            Us = Us_
            residuals_dV = residuals_dV_
            r2 = r2_

        if verbose > 2:
            #print "Delta Vs: %f" % dVs
            pass

        current_cost = cost_function()
        E.append(current_cost)
        if verbose == 1:
            sys.stdout.write('.')
            sys.stdout.flush()
        elif verbose:
            print("Iteration % 3i.2 "
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
                print('Energy increased by %.2f* tol' % (-dE / (tol * E[-1])))
        elif last_iteration or dE < tol * abs(E[-1]):
            if n_shuffle_subjects is not None:
                last_iteration = True
                n_shuffle_subjects = None
                continue
            if verbose == 1:
                print("")
            elif verbose:
                print("--- Convergence reached after %d iterations" % ii)
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
class BaseMSDLModel(BaseEstimator):
    """ Learn our model using Multi Subject Dictionary Learning
    """

    def __init__(self, n_components, prox, mask, alpha=.5, mu='auto',
                        mem=Memory(cachedir=None),
                        non_penalized=None,
                        do_ica=True,
                        tol=1e-4,
                        max_iter=300,
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
        self.mask = mask

    def fit(self, data, V_init=None, **params):
        self.mask_ = self.mask.get_data().astype(np.bool)
        mu = self.mu
        if V_init is None:
            imgs = [unmask(d, self.mask) for d in data]
            if self.do_ica:
                init_model = CanICA(mask=self.mask,
                                n_components=self.n_components,
                                memory=self.mem,
                                n_jobs=self.n_jobs)
            else:
                raise NotImplementedError
                # We should use a multi-pca instead of the ICA
            init_model.fit(imgs)
            V_init = init_model.components_

            # Store for later inspection
            self.init_model_ = init_model
            del imgs
            # We may be imposing a non-negativity constraint, thus we
            # need to make sure that the initial components are as
            # positive as possible
            for component in V_init:
                # Note that the components are centered, thus if we
                # do not square, the two sums will be equal
                if (np.sum(component[component > 0] ** 2) <
                        np.sum(component[component < 0] ** 2)):
                    component[:] *= -1
            if self.verbose > 1:
                print('[%s] Computed V_init' % self.__class__.__name__)
        Us = list()
        Ss = list()

        if mu == 'auto':
            mu = self.mem.cache(estimate_group_regularization,
                                ignore=['mem'])(data,
                                self.n_components, self.mem)
            self.mu_ = mu

        for Y_ in data:
            # Relearn U, V to have the right scaling on U
            Y__ = Y_
            U = linalg.lstsq(V_init.T, Y__.T)[0].copy()
            del Y__
            S = np.sqrt((U ** 2).sum(axis=1))
            S[S == 0] = 1
            U /= S[:, np.newaxis]
            Ss.append(S)
            Us.append(U.T)
        V_init = (V_init * np.mean(Ss, axis=0)[:, np.newaxis])
        if self.verbose > 1:
            print('[%s] Finished learning U_init' % self.__class__.__name__)
        if self.verbose > 1:
            print('[%s] Finished learning V_init' % self.__class__.__name__)

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
            Y__ = Y_
            u, this_residuals = _learn_Us(m, Y__)
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
        del this_residuals, u, m
        self.maps_ *= s[:, np.newaxis]
        self.cov_ = np.mean(covs, axis=0)
        self.covs_ = covs

        return self
