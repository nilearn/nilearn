"""SVM classification on star plus data
"""

# Author : Alexandre Gramfort alexandre.gramfort@inria.fr

from datasets import fetch_star_plus_data
from sklearn.cross_val import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


for k, data in enumerate(fetch_star_plus_data()):
    X = data.data
    y = data.target
    mask = data.mask
    img_shape = mask.shape
    X = X[:, mask != 0]

    clf = SVC(C=1, kernel='linear')
    cv = StratifiedKFold(y, 2)
    print "computing score"
    cv_scores = cross_val_score(clf, X, y, cv=cv, n_jobs=8, verbose=0)

    print "Subject %s scores : %s" % (k, cv_scores)
