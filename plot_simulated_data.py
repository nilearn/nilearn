"""
=================================================
Example of pattern recognition on simulated data
=================================================

This examples simulates data according to a very simple sketch of brain
imaging data and applies machine learing techniques to predict output
values.
"""

# Licence : BSD

print __doc__

from time import time

import numpy as np
import pylab as pl
from scipy import linalg, ndimage

from sklearn import linear_model, svm
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold
from sklearn.feature_selection import f_regression
from nisl import searchlight


###############################################################################
# Fonction to generate data
def create_simulation_data(snr=5, n_samples=2 * 100, size=12, random_state=0):
    generator = check_random_state(random_state)
    roi_size = 2  # size / 3
    smooth_X = 2
    ### Coefs
    w = np.zeros((size, size, size))
    w[0:roi_size, 0:roi_size, 0:roi_size] = -0.6
    w[-roi_size:, -roi_size:, 0:roi_size] = 0.5
    w[0:roi_size, -roi_size:, -roi_size:] = -0.6
    w[-roi_size:, 0:roi_size:, -roi_size:] = 0.5
    w[(size - roi_size) / 2:(size + roi_size) / 2,
      (size - roi_size) / 2:(size + roi_size) / 2,
      (size - roi_size) / 2:(size + roi_size) / 2] = 0.5
    w = w.ravel()
    ### Images
    XX = generator.randn(n_samples, size, size, size)
    X = []
    y = []
    for i in range(n_samples):
        Xi = ndimage.filters.gaussian_filter(XX[i, :, :, :], smooth_X)
        Xi = Xi.ravel()
        X.append(Xi)
        y.append(np.dot(Xi, w))
    X = np.array(X)
    y = np.array(y)
    norm_noise = linalg.norm(y, 2) / np.exp(snr / 20.)
    orig_noise = generator.randn(y.shape[0])
    noise_coef = norm_noise / linalg.norm(orig_noise, 2)
    # Add additive noise
    noise = noise_coef * orig_noise
    snr = 20 * np.log(linalg.norm(y, 2) / linalg.norm(noise, 2))
    print "SNR : %d " % snr
    y += noise

    X -= X.mean(axis=-1)[:, np.newaxis]
    X /= X.std(axis=-1)[:, np.newaxis]
    X_test = X[n_samples / 2:, :]
    X_train = X[:n_samples / 2, :]
    y_test = y[n_samples / 2:]
    y = y[:n_samples / 2]

    return X_train, X_test, y, y_test, snr, noise, w, size


def plot_slices(data, title=None):
    pl.figure(figsize=(5.5, 2.2))
    vmax = np.abs(data).max()
    for i in (0, 6, 11):
        pl.subplot(1, 3, i / 5 + 1)
        pl.imshow(data[:, :, i], vmin=-vmax, vmax=vmax,
                  interpolation="nearest", cmap=pl.cm.RdBu_r)
        pl.xticks(())
        pl.yticks(())
    pl.subplots_adjust(hspace=0.05, wspace=0.05, left=.03, right=.97)
    if title is not None:
        pl.suptitle(title)


###############################################################################
# Create data
X_train, X_test, y_train, y_test, snr, noise, coefs, size = \
    create_simulation_data(snr=10, n_samples=400, size=12)
mask = np.ones((size, size, size), np.bool)
process_mask = np.zeros((size, size, size), np.bool)
process_mask[:, :, 0] = True
process_mask[:, :, 6] = True
process_mask[:, :, 11] = True

coefs = np.reshape(coefs, [size, size, size])
plot_slices(coefs, title="Ground truth")

###############################################################################
# Compute the results and estimated coef maps for different estimators
classifiers = [
    ('bayesian_ridge', linear_model.BayesianRidge(normalize=True)),
    ('enet_cv', linear_model.ElasticNetCV(alphas=[5, 1, 0.5, 0.1], rho=0.05)),
    ('ridge_cv', linear_model.RidgeCV(alphas=[100, 10, 1, 0.1], cv=5)),
    ('svr', svm.SVR(kernel='linear', C=0.001)),
    ('searchlight', searchlight.SearchLight(
        mask=mask, process_mask=process_mask,
        masked_data=True,
        radius=4.,
        score_func=r2_score,
        cv=KFold(y_train.size, k=4)))
]

# Run the estimators
for name, classifier in classifiers:
    t1 = time()
    classifier.fit(X_train, y_train)
    elapsed_time = time() - t1

    if name != 'searchlight':
        coefs = classifier.coef_
        coefs = np.reshape(coefs, [size, size, size])
        score = classifier.score(X_test, y_test)
        title = '%s: prediction score %.3f, training time: %.2fs' % (
                classifier.__class__.__name__, score,
                elapsed_time)

    else:  # Searchlight
        coefs = classifier.scores_
        title = '%s: training time: %.2fs' % (
                classifier.__class__.__name__,
                elapsed_time)

    # We use the plot_slices function provided in the example to
    # plot the results
    plot_slices(coefs, title=title)

    print title

f_values, p_values = f_regression(X_train, y_train)
p_values = np.reshape(p_values, (size, size, size))
p_values = -np.log10(p_values)
p_values[np.isnan(p_values)] = 0
p_values[p_values > 10] = 10
plot_slices(p_values, title="f_regress")

pl.show()
