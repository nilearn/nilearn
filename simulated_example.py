"""
=============================
Supervised clustering example
=============================

Example of a supervised clustering on simulated data
"""

print __doc__


import numpy as np
import pylab as pl
from scipy import linalg, ndimage
from scikits.learn.utils import check_random_state
from scikits.learn.feature_extraction.image import grid_to_graph
from scikits.learn.linear_model import BayesianRidge
from scikits.learn.cross_val import KFold


import supervised_clustering


###############################################################################
# Fonction to generate data

def create_simulation_data(snr=5, n_samples=2*100, size=12, random_state=0):
    generator = check_random_state(random_state)
    roi_size = 2  # size / 3
    smooth_X = 2
    X, y ,z = np.indices((size, size, size))
    ### Coefs
    w = np.zeros((size,size,size))
    w[0:roi_size, 0:roi_size, 0:roi_size] = -0.5
    w[-roi_size:, -roi_size:, 0:roi_size] = 0.5
    w[0:roi_size, -roi_size:, -roi_size:] = -0.5
    w[-roi_size:, 0:roi_size:, -roi_size:] = 0.5
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
    norm_noise = linalg.norm(y, 2) / np.exp(snr/20.)
    generator = check_random_state(0)
    orig_noise = generator.randn(y.shape[0])
    noise_coef = norm_noise / linalg.norm(orig_noise, 2)
    # Add additive noise
    noise = noise_coef * orig_noise
    snr = 20 * np.log(linalg.norm(y, 2) / linalg.norm(noise, 2))
    print "SNR : ",snr
    y +=  noise

    X -= X.mean(axis=-1)[:, np.newaxis]
    X /= X.std(axis=-1)[:, np.newaxis]
    X_test = X[n_samples/2:, :]
    X_train = X[:n_samples/2, :]
    y_test = y[n_samples/2:]
    y = y[:n_samples/2]

    return X_train, X_test, y, y_test, snr, noise, w, size


###############################################################################
# Create data
size = 12
n_samples = 400
X_train, Xtest, y_train, y_test, snr, noise, coefs, size =\
        create_simulation_data(snr=5, n_samples=n_samples, size=size)


###############################################################################
# Compute the results

A = grid_to_graph(n_x=size, n_y=size, n_z=size)
clf = BayesianRidge(fit_intercept=True, normalize=True)

sc = supervised_clustering.SupervisedClusteringRegressor(clf, connectivity=A,
        n_iterations=30, verbose=0, n_jobs=8, cv=KFold(X_train.shape[0], 4))
sc.fit(X_train, y_train)

computed_coefs = sc.inverse_transform()
score =  sc.score(Xtest, y_test)


###############################################################################
# Plot the results

#sc.plot_scores()
#pl.show()

print "Score of the supervised_clustering: ", score
pl.close('all')
computed_coefs = np.reshape(computed_coefs, [size,size,size])
pl.figure(figsize=[2.5,4])
pl.subplots_adjust(left=0.,right=1.,bottom=0.05,top=0.8,
                wspace=0.05,hspace=0.05)
vminmax = np.max(np.abs(computed_coefs))
vmin = 0
vmin = -vminmax
vmax = +vminmax
for i in range(size):
    pl.subplot(4, 3, i+1)
    pl.imshow(computed_coefs[:, :, i], vmin=vmin, vmax=vmax,
            interpolation="nearest", cmap=pl.cm.RdBu_r)
    pl.xticks(())
    pl.yticks(())
pl.show()
