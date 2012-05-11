"""
The Kamitani paper: reconstruction of visual stimuli
======================================================

"""

from matplotlib import pyplot as plt
### Load Kamitani dataset #####################################################
from nisl import datasets
dataset = datasets.fetch_kamitani_data()
X_random = dataset.data_random
X_figure = dataset.data_figure
y_random = dataset.target_random
y_figure = dataset.target_figure

n_features = 100

### Preprocess data ###########################################################
import numpy as np
from scipy import signal

# Detrend data on each session independently
print "detrending data"
for s in range(20):
    X_random[s] = signal.detrend(X_random[s], axis=0)

for s in range(12):
    X_figure[s] = signal.detrend(X_figure[s], axis=0)

X_train = np.hstack(X_random)
y_train = np.hstack(y_random)
X_test = np.hstack(X_figure)
y_test = np.hstack(y_figure)

# Remove rest period
X_train = X_train[:, y_train[0, :] != -1]
y_train = y_train[:, y_train[0, :] != -1]
X_test = X_test[:, y_test[0, :] != -1]
y_test = y_test[:, y_test[0, :] != -1]

X_train = X_train.T
X_test = X_test.T

# Take only the foveal part (radius is custom)
from numpy import linalg
c = (4.5, 4.5)
radius = 2.5
y_mask = np.ones(100, dtype='bool')
for i in range(10):
    for j in range(10):
        y_mask[i * 10 + j] = (linalg.norm((c[0] - i, c[1] - j)) <= radius)

n_features = y_mask.sum()

# Show the mask
# plt.imshow(np.reshape(y_mask, [10, 10]), cmap=plt.cm.gray,
#         interpolation='nearest')
# plt.show()

y_train = y_train[y_mask]
y_test = y_test[y_mask]

# Keep V1 only
"""
X_mask = np.ones(18064, dtype='bool')
rv = dataset.roi_volInd
v1_volInd = np.unique(np.concatenate((rv[0, 2], rv[0, 3], rv[4, 2], rv[4, 3])))
v1_ind = []
for i in v1_volInd:
    v1_ind.append(np.where(dataset.volInd == i)[0])

v1_ind = np.array([y for x in v1_ind for y in x])
X_mask[-v1_ind] = False

X_train = X_train[:, X_mask]
X_test = X_test[:, X_mask]
"""
# Feature selection analysis


def roi_stat(indices):
    # get ROI names
    names = dataset.roi_name[:, 2:4].flatten()
    roi_indices = dataset.roi_volInd[:, 2:4].flatten()
    data_indices = []
    for i, roi_ind in enumerate(roi_indices):
        roi_ind = roi_ind.squeeze()
        data_ind = []
        for p in roi_ind:
            data_ind.append(np.where(dataset.volInd == p)[0])
        data_indices.append(np.array([y for x in data_ind for y in x]))

    count = np.zeros(names.shape)
    for ind in indices:
        for i, data_ind in enumerate(data_indices):
            count[i] += (np.where(np.unique(data_ind) == ind)[0].size != 0)
    return (names, count)

# n,c = roi_stat(np.where(feature_selection.get_support())[0])
# for i, nn in enumerate(n):
#     print nn[0] + " : %d" % c[i]



### Prediction function #######################################################

# from sklearn.svm.sparse import SVC
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

pipelines = []
"""
    f_classif 100 + SVC classique : 65%
"""

for i in range(n_features):
    print "Count %d of %d" % ((i + 1), n_features)
    clf = SVC(kernel='linear', C=1.)
    feature_selection = SelectKBest(f_classif, k=100)
    anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])
    anova_svc.fit(X_train, y_train[i, :])
    pipelines.append(anova_svc)

"""
    f_classif 100 + Ridge

from sklearn.linear_model import Ridge

clf = Ridge(alpha=1.)

pvals = np.zeros(X_train.shape[1])
for i in range(n_features):
    F, pval = f_classif(X_train, y_train[i])
    pvals += pval

mask = np.zeros(X_train.shape[1], dtype=bool)
mask[np.argsort(pvals)[:100]] = 1

X = X_train[:, mask]
X_target = X_test[:, mask]

clf.fit(X, y_train.T)

acc = []

for i, y in enumerate(y_test):
    pred = clf.predict(X_target)
    acc.append((pred.squeeze() == y_test[:, i]).sum())
"""

"""
    Sparse SVC (tres long, 3 heures) :

from sklearn.svm.sparse import SVC
acc = np.zeros(n_features)

for i in range(100):
    print "Count %d of %d" % ((i + 1), n_features)
    clf = SVC(kernel='linear', C=1.)
    clf.fit(X_train, y_train[i, :])
    score = 0
    for j, y in enumerate(y_test.T):
        score += clf.predict(X_test[j, :]) == y[i]
    acc[i] = score

"""


"""
    Sparse SVC sur V1 :
    -------------------



from sklearn.svm.sparse import SVC
acc = []
clfs = []

for i in range(n_features):
    print "Count %d of %d" % ((i + 1), n_features)
    clf = SVC(kernel='linear', C=1.)
    clf.fit(X_train_v1, y_train[i, :])
    clfs.append(clf)

"""

"""
    f_classif 100 + sparse SVC
    --------------------------

from sklearn.svm.sparse import SVC

for i in range(n_features):
    print "Count %d of %d" % ((i + 1), n_features)
    clf = SVC(kernel='linear', C=1.)
    feature_selection = SelectKBest(f_classif, k=100)
    anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])
    anova_svc.fit(X_train, y_train[i, :])
    pipelines.append(anova_svc)

acc = []

for i, y in enumerate(y_test):
    pred = []
    for p in pipelines:
        pred.append(p.predict(X_test[i, :]))
    pred = np.array(pred)
    acc.append((pred.squeeze() == y_test[:, i]).sum())

"""

"""
    f_classif 100 + Ridge
    --------------------------

from sklearn.linear_model import RidgeClassifier

for i in range(_features):
    print "Count %d of %d" % ((i + 1), n_features)
    clf = RidgeClassifier(alpha=10.)
    feature_selection = SelectKBest(f_classif, k=100)
    anova_clf = Pipeline([('anova', feature_selection), ('clf', clf)])
    anova_clf.fit(X_train, y_train[i, :])
    pipelines.append(anova_clf)

acc = []

for i, y in enumerate(y_test):
    pred = []
    for p in pipelines:
        pred.append(p.predict(X_test[i, :]))
    pred = np.array(pred)
    acc.append((pred.squeeze() == y_test[:, i]).sum())
"""

acc = []

for p in pipelines:
    pred = []
    for i, y in enumerate(y_test):
        pred.append(p.predict(X_test[i, :]))
    pred = np.array(pred)
    acc.append(pred.squeeze().mean())

# Visualize results
y_pred = np.zeros(y_mask.shape)
y_pred[y_mask] = acc
plt.imshow(np.reshape(y_pred, [10, 10]), cmap=plt.cm.gray,
        interpolation='nearest')
plt.show()

#print "Result : %d" % np.mean(acc)
