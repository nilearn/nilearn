"""
==============================
Supervised clustering VS Anova
==============================
"""

# Licence : BSD

print __doc__


import numpy as np
import pylab as pl
from scipy import linalg, ndimage
from scikits.learn.utils import check_random_state
from scikits.learn.feature_extraction.image import grid_to_graph
from scikits.learn.svm import SVC
from scikits.learn.feature_selection import SelectKBest, f_classif
from scikits.learn.pipeline import Pipeline

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
y_test, y_train = np.sign(y_test), np.sign(y_train)


##############################################################################
# Computing the score with Anova
# Here we use a Support Vector Classification, with a linear kernel and
# C=1
clf = SVC(kernel='linear', C=1.)

### Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. We set the number of features to be selected to 500
feature_selection = SelectKBest(f_classif, k=500)

### We combine the dimension reduction and the prediction function
anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])

# Computing the score with anova
anova_svc.fit(X_train, y_train)
anova_score = anova_svc.score(X_test, y_test)
clf.fit(feature_selection.transform(X_train), y_train)
anova_coefs = clf.coef_

###############################################################################
# Computing the score with supervised clustering
A = grid_to_graph(n_x=size, n_y=size, n_z=size)
estimator = SVC(kernel='linear', C=1.)
sc = supervised_clustering.SupervisedClusteringClassifier(estimator=estimator,
        connectivity=A, n_iterations=25, verbose=1, n_jobs=8, n_folds=9)
sc.fit(X_train, y_train)
computed_coefs = sc.inverse_transform()
sc_score = sc.score(X_test, y_test)


# Printing the scores
print "Score of the supervised_clustering: %f" % sc_score
print "Number of parcels : %d" % len(np.unique(sc.labels_))

print "Score of Anova: %f" % anova_score
print "n_parcels : %d" % len(clf.coef_[0])

###############################################################################
# Plot the results

pl.close('all')
pl.figure()
pl.title('Scores of the supervised clustering')
pl.subplot(2, 1, 1)
pl.bar(np.arange(len(sc.scores_)), sc.scores_)
pl.xlabel('score')
pl.ylabel('iteration')
pl.title('Score of the best parcellation of each iteration')
pl.subplot(2, 1, 2)
pl.bar(np.arange(len(sc.delta_scores_)), sc.delta_scores_)
pl.xlabel('delta_score')
pl.ylabel('iteration')
pl.title('Delta_Score of the best parcellation of each iteration')


computed_coefs = np.reshape(computed_coefs, [size, size, size])
pl.figure(figsize=(3*2, 3*1.5))
vminmax = np.max(np.abs(computed_coefs))
vmin = 0
vmin = -vminmax
vmax = +vminmax
computed_coefs *= 3
pl.suptitle('Supervised Clustering VS simple estimator', size=27)

for i in [0, 6, 11]:
    pl.subplot(3, 3, i/5+1)
    pl.imshow(computed_coefs[:, :, i], vmin=vmin, vmax=vmax,
            interpolation="nearest", cmap=pl.cm.RdBu_r)
    if i==0:
        pl.ylabel('computed coefs', size=19)
    pl.xticks(())
    pl.yticks(())

coefs = coefs.reshape((size, size, size))
for i in [0, 6, 11]:
    pl.subplot(3, 3, i/5+4)
    pl.imshow(coefs[:, :, i], vmin=vmin, vmax=vmax,
            interpolation="nearest", cmap=pl.cm.RdBu_r)
    if i == 0:
        pl.ylabel('real coefs', size=19)
    pl.xticks(())
    pl.yticks(())

anova_coefs = feature_selection.inverse_transform(anova_coefs).reshape((size, size, size))
vminmax = np.max(np.abs(anova_coefs))
vmin = 0
vmin = -vminmax
vmax = +vminmax

anova_coefs *= 3

for i in [0, 6, 11]:
    pl.subplot(3, 3, i/5+7)
    pl.imshow(anova_coefs[:, :, i], vmin=vmin, vmax=vmax,
            interpolation="nearest", cmap=pl.cm.RdBu_r)
    if i == 0:
        pl.ylabel('Anova', size=19)
    pl.xticks(())
    pl.yticks(())

pl.subplots_adjust(hspace=0.05, wspace=0.05)
pl.show()
