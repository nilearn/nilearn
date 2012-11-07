"""
Our model for brain activity.
"""
import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator
from sklearn.utils.extmath import fast_logdet


def log_likelihood(test_series, cov, maps, residues):
    """ Return the log likelihood of test_series under the model
        described by cov, maps, and residues.
    """
    # try:
    # This makes heavy use of the matrix inversion lemma
    #test_series = np.concatenate(test_series, axis=0)
    n_samples = test_series.shape[0]
    white_test_series = test_series / residues
    residues_fit = np.sum(white_test_series ** 2)
    white_test_series /= residues
    white_projection = np.dot(white_test_series, maps)
    del white_test_series
    prec_maps = linalg.inv(cov)
    prec_maps += np.dot(maps.T / residues ** 2, maps)
    residues_fit -= np.trace(
        np.dot(np.dot(white_projection.T, white_projection),
               linalg.inv(prec_maps)))
    del white_projection
    white_maps = maps / residues[:, np.newaxis]
    prec_maps += np.dot(white_maps.T, white_maps)
    del white_maps
    det = fast_logdet(prec_maps)
    del prec_maps
    return (-residues_fit / n_samples - fast_logdet(cov)
            - det - 2 * np.sum(np.log(residues)))
    #except linalg.LinAlgError:
    #    return -np.inf


def log_likelihood_full(test_series, full_cov):
    """ Return the log likelihood of test_series under the model
        described by cov, maps, and residues.
    """
    # Without the matrix inversion lemma
    n_samples = test_series.shape[0]
    return -fast_logdet(full_cov) - 1. / n_samples * \
        np.trace(np.dot(np.dot(test_series, linalg.inv(full_cov)),
                        test_series.T))


def learn_time_series(maps, subject_data):
    # For this, we do a ridge regression with a very small
    # regularisation, corresponds to a least square with control on
    # the conditioning
    maps_cov = np.dot(maps, maps.T)
    n_maps = len(maps_cov)
    maps_cov.flat[::n_maps + 1] += .01 * np.trace(maps_cov) / n_maps
    u = linalg.solve(maps_cov, np.dot(maps, subject_data.T),
                     sym_pos=True, overwrite_a=True, overwrite_b=True)
    residuals = np.dot(u.T, maps)
    residuals -= subject_data
    residuals **= 2
    residuals = np.mean(residuals, axis=0)
    return u, residuals


###############################################################################
# Base model
class DecompositionModel(BaseEstimator):

    def score(self, test_series):
        # XXX: might need to relearn maps
        if len(test_series[0].shape) == 2:
            for series in test_series:
                series -= series.mean(axis=0)
            # n_subjects = len(test_series)
            n_samples, n_voxels = test_series[0].shape
            test_series = np.concatenate(test_series, axis=0)
        else:
            test_series -= test_series.mean(axis=0)
        return log_likelihood(test_series, self.cov_, self.maps_.T,
                              self.residuals_)

    def get_full_cov(self):
        """ Return the full covariance for a model described by cov, maps and
            residues
        """
        return (np.dot(np.dot(self.maps_.T, self.cov_), self.maps_)
                + np.diag(self.residuals_ ** 2))

    def learn_time_series(self, data):
        return learn_time_series(self.maps_, data)

    def learn_from_maps(self, data):
        """ Learn time-series and covariance from the maps.
        """
        # Remove any map with only zero values:
        self.maps_ = self.maps_[self.maps_.ptp(axis=1) != 0]
        if not len(self.maps_):
            # All maps are zero
            self.cov_ = np.array([[]], dtype=np.float)
            return
        # Flip sign to always have positive features
        for map in self.maps_:
            mask = map > 0
            if map[mask].sum() > - map[np.logical_not(mask)].sum():
                map *= -1

        # Relearn U, V to have the right scaling on U
        residuals = None
        #residuals = 0
        U = list()
        for d in data:
            u, this_residuals = self.learn_time_series(d)
            U.append(u)
            #this_residuals = np.sqrt(np.mean(this_residuals))
            #residuals += this_residuals
            if residuals is None:
                residuals = this_residuals
            else:
                residuals += this_residuals
        residuals /= len(data)
        #self.residuals_ = np.atleast_1d(residuals)
        self.residuals_ = residuals  # = np.sqrt(residuals)
        self.residuals_.fill(np.sqrt(self.residuals_.mean()))
        del this_residuals, u, d
        U = np.concatenate(U, axis=1)
        n_samples = U.shape[1]
        S = np.sqrt((U ** 2).sum(axis=1) / n_samples)
        U /= S[:, np.newaxis]
        self.maps_ *= S[:, np.newaxis]
        self.cov_ = 1. / n_samples * np.dot(U, U.T)
        #self.cov_ =  np.eye(n_maps)
