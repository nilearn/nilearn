"""
The Kamitani paper: reconstruction of visual stimuli
======================================================

"""

# Options
# =======

remove_rest_period = True
foveal_focus_radius = None  # 2.5 is a good value if set
multi_scale = True          # Not compatible with foveal focal radius
voxel_restriction = 'roi'   # could also be 'V1' or None
generate_video = 'test.mp4'

###############################################################################
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
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

X_train = np.hstack(X_random).T
y_train = np.hstack(y_random).T
X_test = np.hstack(X_figure).T
y_test = np.hstack(y_figure).T


n_pixels = y_train.shape[1]
n_features = X_train.shape[1]


# Remove rest period
if remove_rest_period:
    X_train = X_train[y_train[:, 0] != -1]
    y_train = y_train[y_train[:, 0] != -1]
    X_test = X_test[y_test[:, 0] != -1]
    y_test = y_test[y_test[:, 0] != -1]


# Take only the foveal part (radius is custom)
if foveal_focus_radius:
    from numpy import linalg
    c = (4.5, 4.5)
    radius = foveal_focus_radius
    y_mask = np.ones(100, dtype='bool')
    for i in range(10):
        for j in range(10):
            y_mask[i * 10 + j] = (linalg.norm((c[0] - i, c[1] - j)) <= radius)
    n_pixels = y_mask.sum()
    y_train = y_train[:, y_mask]
    y_test = y_test[:, y_mask]
    # Show the mask
    # plt.imshow(np.reshape(y_mask, [10, 10]), cmap=plt.cm.gray,
    #         interpolation='nearest')
    # plt.show()

# Compute scaled images

if multi_scale:
    y_shape = (10, 10)
    y_rows, y_cols = y_shape
    height_transform = np.repeat((np.identity(y_cols / 2) * 0.5), 2, 1)
    width_transform = np.repeat((np.identity(y_rows / 2) * 0.5), 2, 0)

    y_train_tall = np.array(
        [np.dot(height_transform, np.reshape(m, y_shape)).flatten()
        for m in y_train])
    y_train_large = np.array(
        [np.dot(np.reshape(m, y_shape), width_transform).flatten()
        for m in y_train])
    y_train_big = np.array([np.dot(height_transform,
        np.dot(np.reshape(m, y_shape), width_transform)).flatten()
        for m in y_train])

    # We add them to original data
    y_train = np.concatenate((y_train, y_train_tall,
            y_train_large, y_train_big), axis=1)

# Keep V1 only
if voxel_restriction:
    X_mask = np.zeros(n_features, dtype='bool')
    rv = dataset.roi_volInd
    if voxel_restriction == 'V1':
        volInd = np.unique(
            np.concatenate((rv[0, 2], rv[0, 3], rv[4, 2], rv[4, 3])))
    elif voxel_restriction == 'roi':
        volInd = np.unique(np.array([x for y in rv.flatten() for x in y]))
    else:
        print 'Error: unknown voxel restriction'
        exit
    mask_ind = []
    for i in volInd:
        mask_ind.append(np.where(dataset.volInd == i)[0])
    mask_ind = np.array([y for x in mask_ind for y in x])
    X_mask[mask_ind] = True

    X_train = X_train[:, X_mask]
    X_test = X_test[:, X_mask]
    n_features = X_train.shape[1]

# Feature selection analysis

"""
def roi_stat(indices):
    # get ROI names
    names = dataset.roi_name[0:8, 2:4].flatten()
    roi_indices = dataset.roi_volInd[0:8, 2:4].flatten()
    names = dataset.roi_name[:, 0:2].flatten()
    roi_indices = dataset.roi_volInd[:, 0:2].flatten()
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


from sklearn.svm import SVC
feature_selection = SelectKBest(f_classif, k=100)

feature_selection = RFE(SVC(kernel='linear', C=1.), n_features_to_select=100)

roi_features = []
for i in range(n_features):
    feature_selection.fit(X_train, y_train[i])
    n, c = roi_stat(np.where(feature_selection.get_support())[0])
    roi_features.append(c)

rf = np.array(roi_features)
plt.figure(1)
for i, nn in enumerate(n):
    plt.subplot(6, 4, i + 1)
    plt.axis('off')
    plt.title(nn)
    plt.imshow(np.reshape(rf[:, i], [10, 10]), cmap=plt.cm.hot,
                      interpolation='nearest', vmin=0, vmax=100)

plt.show()

for i, nn in enumerate(n):
    print nn[0] + " : %d" % c[i]
"""

### Prediction function #######################################################

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import copy

"""
def make_pipelines(feature_selection, clf, X_train, y_train, n_features):
    pipelines = []
    pipeline_ref = Pipeline([('fs', feature_selection), ('clf', clf)])
    for i in range(n_features):
        print "Count %d of %d" % ((i + 1), n_features)
        pipeline = copy.deepcopy(pipeline_ref)
        pipeline.fit(X_train, y_train[i, :])
        pipelines.append(pipeline)
    return pipelines


def predict(pipelines, X_test):
    preds = []
    for i, x_test in enumerate(X_test):
        pred = []
        for p in pipelines:
            pred.append(p.predict(x_test))
        pred = np.array(pred)
        preds.append(pred.squeeze())
    return preds
"""

"""
    f_classif 100 + SVC classique : 65%


pipelines = []

for i in range(n_features):
    print "Count %d of %d" % ((i + 1), n_features)
    clf = SVC(kernel='linear', C=1.)
    feature_selection = SelectKBest(f_classif, k=50)
    anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])
    anova_svc.fit(X_train, y_train[i, :])
    pipelines.append(anova_svc)

"""
"""
    f_classif 100 + Ridge
"""

from sklearn.linear_model import OrthogonalMatchingPursuit as OMP

clf = OMP(n_nonzero_coefs=20)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

"""
clf.fit(X_train, y_train_tall.T)
y_pred_tall = clf.predict(X_test)

clf.fit(X_train, y_train_large.T)
y_pred_large = clf.predict(X_test)

clf.fit(X_train, y_train_big.T)
y_pred_big = clf.predict(X_test)
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

"""

"""
    f_classif 100 + Ridge
    --------------------------

from sklearn.linear_model import RidgeClassifier

for i in range(n_features):
    print "Count %d of %d" % ((i + 1), n_features)
    clf = RidgeClassifier(alpha=10.)
    feature_selection = SelectKBest(f_classif, k=100)
    anova_clf = Pipeline([('anova', feature_selection), ('clf', clf)])
    anova_clf.fit(X_train, y_train[i, :])
    pipelines.append(anova_clf)
"""

"""
# Learn
pipes = make_pipelines(SelectKBest(f_classif, k=100),
        SVC(kernel='linear', C=1.), X_train, y_train, n_features)

# Predict
y_pred = predict(pipes, X_test)
"""

### Prediction ################################################################

# Revert scaled images if needed
if multi_scale:
    y_preds = np.split(y_pred, [n_pixels, 1.5 * n_pixels,
            2 * n_pixels, 2.25 * n_pixels], axis=1)

    y_pred = y_preds[0]
    y_pred_tall = np.array([np.dot(height_transform.T * 2,
        np.reshape(m, (5, 10))).flatten() for m in y_preds[1]])
    y_pred_large = np.array([np.dot(np.reshape(m, (10, 5)),
        width_transform.T * 2).flatten() for m in y_preds[2]])
    y_pred_big = [np.dot(height_transform.T * 2, np.reshape(m, (5, 5)))
        for m in y_preds[3]]
    y_pred_big = np.array([np.dot(np.reshape(m, (10, 5)),
        width_transform.T * 2).flatten() for m in y_pred_big])

    y_pred = (.25 * y_pred + .25 * y_pred_tall + .25 * y_pred_large
        + .25 * y_pred_big)

"""
preds = []
for i, x_test in enumerate(X_test):
    pred = []
    for p in pipelines:
        pred.append(p.predict(x_test))
    pred = np.array(pred)
    preds.append(pred.squeeze())


# Visualize results
y_pred = np.zeros(y_mask.shape)
y_pred[y_mask] = acc
plt.imshow(np.reshape(y_pred, [10, 10]), cmap=plt.cm.gray,
        interpolation='nearest')
plt.show()

print "Result : %d" % np.mean(acc)
"""

if generate_video:
    from matplotlib import animation
    fig = plt.figure()
    sp1 = plt.subplot(121)
    sp1.axis('off')
    sp2 = plt.subplot(122)
    sp2.axis('off')
    ims = []
    for i, t in enumerate(y_pred):
        ims.append((
            sp1.imshow(np.reshape(y_test[i], (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest'),
            sp2.imshow(np.reshape(t, (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest')))

    im_ani = animation.ArtistAnimation(fig, ims, interval=1000)
    im_ani.save(generate_video)
