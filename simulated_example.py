"""
=============================
Supervised clustering example
=============================
"""

# Licence : BSD

print __doc__


import numpy as np
import pylab as pl
from scipy import linalg, ndimage
from scikits.learn.utils import check_random_state
from scikits.learn.feature_extraction.image import grid_to_graph
from scikits.learn.linear_model import BayesianRidge
from time import time

import supervised_clustering

###############################################################################
# Fonction to generate data
def create_simulation_data(snr=5, n_samples=2*100, size=12, random_state=0):
    generator = check_random_state(random_state)
    roi_size = 2  # size / 3
    smooth_X = 2
    ### Coefs
    w = np.zeros((size, size, size))
    w[0:roi_size, 0:roi_size, 0:roi_size] = -0.6
    w[-roi_size:, -roi_size:, 0:roi_size] = 0.5
    w[0:roi_size, -roi_size:, -roi_size:] = -0.6
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
    print "SNR : %d " % snr
    y += noise

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
X_train, X_test, y_train, y_test, snr, noise, coefs, size =\
        create_simulation_data(snr=10, n_samples=n_samples, size=size)


###############################################################################
# Compute the results for supervised clustering
A = grid_to_graph(n_x=size, n_y=size, n_z=size)
clf = BayesianRidge(fit_intercept=True, normalize=True, tol=1.e-3)
sc = supervised_clustering.SupervisedClusteringRegressor(estimator=clf,
        connectivity=A, n_iterations=30, cv=25, verbose=1, n_jobs=8)
#sc = supervised_clustering.SupervisedClusteringRegressor(clf, connectivity=A,
#        n_iterations=30, verbose=1, n_jobs=8,
#        cv=ShuffleSplit(X_train.shape[0], n_splits=10, test_fraction=0.6,
#            random_state=0))
t1 = time()
sc.fit(X_train, y_train)
sc_time = time() -t1
computed_coefs = sc.inverse_transform()
computed_coefs = np.reshape(computed_coefs, [size, size, size])
score = sc.score(X_test, y_test)


###############################################################################
# Compute the results for simple BayesianRidge
t1 = time()
clf.fit(X_train, y_train)
bayes_time = time() - t1
bayes_coefs = clf.coef_
bayes_score = clf.score(X_test, y_test)
bayes_coefs = bayes_coefs.reshape((size, size, size))


###############################################################################
# Plot the results

pl.close('all')
pl.figure()
pl.title('Scores of the supervised clustering')
pl.subplot(2, 1, 1)
pl.plot(np.arange(len(sc.scores_)), sc.scores_)
pl.xlabel('score')
pl.ylabel('iteration')
pl.title('Score of the best parcellation of each iteration')
pl.subplot(2, 1, 2)
pl.plot(np.arange(len(sc.delta_scores_)), sc.delta_scores_)
pl.xlabel('delta_score')
pl.ylabel('iteration')
pl.title('Delta_Score of the best parcellation of each iteration')



pl.figure(figsize=(3*2, 3*1.5))
vminmax = np.max(np.abs(computed_coefs))
vmin = 0
vmin = -vminmax
vmax = +vminmax
computed_coefs *= 3
pl.suptitle('Supervised Clustering VS simple estimator,\nSNR = %d' % snr,
        size=27)

coefs = coefs.reshape((size, size, size))
for i in [0, 6, 11]:
    pl.subplot(3, 3, i/5+4)
    pl.imshow(coefs[:, :, i], vmin=vmin, vmax=vmax,
            interpolation="nearest", cmap=pl.cm.RdBu_r)
    if i == 0:
        pl.ylabel('real coefs', size=18)
    pl.xticks(())
    pl.yticks(())

for i in [0, 6, 11]:
    pl.subplot(3, 3, i/5+1)
    pl.imshow(computed_coefs[:, :, i], vmin=vmin, vmax=vmax,
            interpolation="nearest", cmap=pl.cm.RdBu_r)
    if i==0:
        pl.ylabel('Supervised Clustering coefs,\n\
                score = %f,\n %d parcels,\n\
                execution time : %f' % (score, len(np.unique(sc.labels_)),
                    sc_time), size=18)
    pl.xticks(())
    pl.yticks(())

# Plotting the BayesianRidge's coefs
vminmax = np.max(np.abs(bayes_coefs))
vmin = 0
vmin = -vminmax
vmax = +vminmax

bayes_coefs *= 3

for i in [0, 6, 11]:
    pl.subplot(3, 3, i/5+7)
    pl.imshow(bayes_coefs[:, :, i], vmin=vmin, vmax=vmax,
            interpolation="nearest", cmap=pl.cm.RdBu_r)
    if i == 0:
        pl.ylabel('simple BayesianRidge,\n score = %f,\n execution time = %f'\
                % (bayes_score, bayes_time), size=19)
    pl.xticks(())
    pl.yticks(())

pl.subplots_adjust(hspace=0.05, wspace=0.05)
pl.show()
