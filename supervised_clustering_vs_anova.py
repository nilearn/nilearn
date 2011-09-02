"""
==============================
Supervised clustering VS Anova
==============================

Chance level = 0.5
"""

# Licence : BSD

print __doc__

import numpy as np
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

###############################################################################
# Computing the score with supervised clustering
A = grid_to_graph(n_x=size, n_y=size, n_z=size)
estimator = SVC(kernel='linear', C=1.)
sc = supervised_clustering.SupervisedClusteringClassifier(estimator=estimator,
        connectivity=A, n_iterations=15, cv=9, verbose=0, n_jobs=-1)
sc.fit(X_train, y_train)
sc_score = sc.score(X_test, y_test)


# Printing the scores
print "\n============================================"
print "Score of the supervised_clustering with SVC: %f" % sc_score
print "( %d parcels)" % len(np.unique(sc.labels_))
print "============================================\n"
print "========================"
print "Score of Anova with SVC: %f" % anova_score
print "========================\n\n"
