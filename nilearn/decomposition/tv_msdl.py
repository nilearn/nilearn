"""
Sparse recovery of group maps using dictionary learning
"""
from joblib import Memory, Parallel, delayed
import numpy as np

from .base_msdl import BaseMSDLModel
from parietal.learn.proximal.prox_tv_l1 import prox_tv_l1, tv_l1_from_gradient, gradient_id


def _prox_tvl1(v_group, mask, l1_ratio, options):
    # Unmask the image
    img = np.zeros(mask.shape)
    img[mask] = v_group
    img = prox_tv_l1(img, l1_ratio=l1_ratio, **options)
    v_group[:] = img[mask]
    tv_l1_norm = tv_l1_from_gradient(gradient_id(img, l1_ratio))

    return v_group, tv_l1_norm


class ProxTV(object):

    def __init__(self, mask, alpha, l1_ratio, positive, adaptive_dgap,
                 n_jobs=1, verbose=0):
        self.mask = mask
        self.l1_ratio = l1_ratio
        options = {}
        options['weight'] = alpha
        if positive:
            options['val_min'] = 0
            options['val_max'] = np.nan
        self.options = options
        self.adaptive_dgap = adaptive_dgap
        self.n_jobs = n_jobs
        self.verbose = verbose

    def set_dVs(self, dVs):
        if self.adaptive_dgap:
            dgap = dVs / 3.
            self.options['dgap_tol'] = dgap

    def __call__(self, V_group, alpha):
        self.options['weight'] = alpha
        imgs_norms = Parallel(n_jobs=self.n_jobs)(
            delayed(_prox_tvl1)(
                v_group, self.mask, self.l1_ratio, self.options)
            for v_group in V_group)
        imgs, norms = zip(*imgs_norms)
        imgs = np.asarray(imgs)
        return imgs, norms


def sparse_tv_group_dictionary(Ys, mask, n_atoms, alpha, mu, l1_ratio=.5,
                               positive=True, adaptive_dgap=True,
                               maxit=100, minit=3, tol=1e-8,
                               n_jobs=1,
                               Us_init=None, Vs_init=None, V_init=None,
                               Vs_method='solve', Vs_warm_restart=False,
                               n_shuffle_subjects=None,
                               callback=None, verbose=False, copy=True,
                               non_penalized=None):

    prox = ProxTV(mask, alpha, l1_ratio, positive, adaptive_dgap,
                  n_jobs, verbose)

    return base_gspca.group_dictionary(Ys, n_atoms, prox, alpha, mu, maxit,
                                       minit, tol,
                                       n_jobs, Us_init, Vs_init, V_init,
                                       n_shuffle_subjects, callback, verbose,
                                       copy, non_penalized)


###############################################################################
class TVMSDL(BaseMSDLModel):
    """ Learn our model using TVMSDL
    """

    def __init__(self, mask, n_components, alpha=.5, mu='auto',
                 l1_ratio=.5, positive=True, adaptive_dgap=True,
                 mem=Memory(cachedir=None),
                 non_penalized=None,
                 do_ica=True,
                 tol=1e-4,
                 max_iter=300,
                 n_shuffle_subjects=None,
                 n_jobs=1,
                 verbose=False):

        # This has to be stored for sklearn.clone to work
        self.l1_ratio = l1_ratio
        self.positive = positive
        self.adaptive_dgap = adaptive_dgap

        # This alpha does not depend on the number of subjects.
        prox = ProxTV(mask.get_data().astype(np.bool), alpha,
                      l1_ratio, positive, adaptive_dgap,
                      n_jobs, verbose)

        super(TVMSDL, self).__init__(
            n_components,
            prox,
            mask,
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
