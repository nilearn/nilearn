### Load Kamitani dataset #####################################################
from nisl import datasets
dataset = datasets.fetch_kamitani_data()
X_random = dataset.data_random
X_figure = dataset.data_figure
y_random = dataset.target_random
y_figure = dataset.target_figure

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
### Prediction function #######################################################

# from sklearn.svm.sparse import SVC
# from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

pipelines = []
"""
    f_classif 100 + SVC classique : 65%

for i in range(100):
    print "Count %d of 100" % (i + 1)
    clf = SVC(kernel='linear', C=1.)
    feature_selection = SelectKBest(f_classif, k=500)
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

from sklearn.linear_model import Ridge

clf = Ridge(alpha=1.)

pvals = np.zeros(X_train.shape[1])
for i in range(100):
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

acc = np.zeros(100)

for i in range(100):
    print "Count %d of 100" % (i + 1)
    clf = SVC(kernel='linear', C=1.)
    clf.fit(X_train, y_train[i, :])
    score = 0
    for j, y in enumerate(y_test.T):
        score += clf.predict(X_test[j, :]) == y[i]
    acc[i] = score
"""

"""
    f_classif 100 + sparse SVC
    --------------------------
"""

from sklearn.svm.sparse import SVC

for i in range(100):
    print "Count %d of 100" % (i + 1)
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
    f_classif 100 + Ridge
    --------------------------

from sklearn.linear_model import RidgeClassifier

for i in range(100):
    print "Count %d of 100" % (i + 1)
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

print "Result : %d" % np.mean(acc)
