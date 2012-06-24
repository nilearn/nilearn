# Options
# =======

# Preprocessing
remove_rest_period = True
foveal_focus_radius = None  # 2.5 is a good value if set
multi_scale = False          # Not compatible with foveal focal radius
voxel_restriction = 'roi'   # could also be 'V1' or None
preprocessing = 'detrend'

# Learning
learn_fusion_params = False  # Learn fusion params with LinearRegression
# Available classifiers :
# - anova_svc: f_classif 50 features + linear SVC (C = 1.)
# - ridge: ridge regression
# - omp: Orthgonal Matching Pursuit (n_nonzero_coefs=20)
# - anova_ridge: f_classif 50 features + ridge regression
# - lassolars
# - bayesianridge: Bayesian Ridge
classifier = 'lassolars'

# Output
generate_video = 'testLL.mp4'

### Imports ###################################################################

from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline


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

print "Preprocessing data"

# Detrend data on each session independently
if preprocessing == "detrend":
    for x in X_random:
        x[:] = signal.detrend(x, axis=1)
    for x in X_figure:
        x[:] = signal.detrend(x, axis=1)
elif preprocessing == "normalize":
    for x in X_random:
        x[:] = signal.detrend(x, axis=1)
    for x in X_figure:
        x[:] = signal.detrend(x, axis=1)


#for s in range(20):
#    X_random[s] -= X_random[s].mean()

#for s in range(12):
#    X_figure[s] -= X_figure[s].mean()

X_train = np.hstack(X_random).T
y_train = np.hstack(y_random).astype(np.float).T
X_test = np.hstack(X_figure).T
y_test = np.hstack(y_figure).astype(np.float).T

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
    height_transform = np.zeros((y_rows - 1, y_rows))
    for i in range(y_rows - 1):
        height_transform[i, i] = .5
        height_transform[i, i + 1] = .5
    width_transform = np.zeros((y_cols, y_cols - 1))
    for i in range(y_cols - 1):
        width_transform[i, i] = .5
        width_transform[i + 1, i] = .5

    width_backward = width_transform.T
    width_backward[0, 0] = 1
    width_backward[y_cols - 2, y_cols - 1] = 1

    height_backward = height_transform.T
    height_backward[0, 0] = 1
    height_backward[y_rows - 1, y_rows - 2] = 1

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


# Filter voxels to take all ROIs or just V1
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


# Try to learn coefficients to merge images
if multi_scale and learn_fusion_params:
    from sklearn.linear_model import LinearRegression
    clf = LinearRegression()
    clf.fit(X_train, y_train)

    t_pred = clf.predict(X_train)
    t_preds = np.split(t_pred, [n_pixels, 2 * n_pixels - y_rows,
        3 * n_pixels - y_rows - y_cols, 4 * n_pixels - y_rows - y_cols + 1],
        axis=1)

    t_pred = t_preds[0]
    t_pred_tall = np.array([np.dot(height_backward,
        np.reshape(m, (9, 10))).flatten() for m in t_preds[1]])
    t_pred_large = np.array([np.dot(np.reshape(m, (10, 9)),
        width_backward).flatten() for m in t_preds[2]])
    t_pred_big = [np.dot(height_backward, np.reshape(m, (9, 9)))
        for m in t_preds[3]]
    t_pred_big = np.array([np.dot(np.reshape(m, (10, 9)),
        width_backward).flatten() for m in t_pred_big])

    fusions = []
    from sklearn.linear_model import LinearRegression
    for i, t in enumerate(t_pred.T):
        tX = np.column_stack((t_pred[:, i], t_pred_tall[:, i],
            t_pred_large[:, i], t_pred_big[:, i]))
        f = LinearRegression()
        f.fit(tX, y_train[:, i])
        fusions.append(f.coef_)

    fusions = np.array(fusions)
    fusions = (fusions.T / np.sum(fusions, axis=1)).T

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

"""

### Prediction function #######################################################

print "Learning"
"""
import copy

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

# f_classif + SVC classique : 65%
if classifier == "anova_svc":
    unique_classifier = False

    from sklearn.svm import SVC
    clfs = []

    for i, pixel_time_serie in enumerate(y_train.T):
        print "Count %d of %d" % ((i + 1), y_train.shape[1])
        clf = SVC(kernel='linear', C=1.)
        feature_selection = SelectKBest(f_classif, k=50)
        anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])
        anova_svc.fit(X_train, pixel_time_serie)
        clfs.append(anova_svc)

# Ridge
elif classifier == "ridge":
    unique_classifier = True
    from sklearn.linear_model import RidgeCV
    clf = RidgeCV()
    clf.fit(X_train, y_train)

# f_classif + Ridge
elif classifier == 'anova_ridge':
    unique_classifier = False
    from sklearn.linear_model import RidgeCV
    clfs = []
    for i, pixel_time_serie in enumerate(y_train.T):
        print "Count %d of %d" % ((i + 1), y_train.shape[1])
        clf = RidgeCV()
        feature_selection = SelectKBest(f_classif, k=50)
        anova_clf = Pipeline([('anova', feature_selection), ('clf', clf)])
        anova_clf.fit(X_train, pixel_time_serie)
        clfs.append(anova_clf)

# OMP
elif classifier == "omp":
    unique_classifier = True
    from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
    clf = OMP(n_nonzero_coefs=20)
    clf.fit(X_train, y_train)

# LassoLars
elif classifier == "lassolars":
    unique_classifier = False
    from sklearn.linear_model import LassoLarsCV

    clfs = []
    for i, pixel_time_serie in enumerate(y_train.T):
        print "Count %d of %d" % ((i + 1), y_train.shape[1])
        clf = LassoLarsCV()
        clf.fit(X_train, pixel_time_serie)
        clfs.append(clf)

# f_classif 100 + sparse SVC
elif classifier == "anova_sparsesvc":
    unique_classifier = False
    from sklearn.svm.sparse import SVC
    clfs = []

    for i, pixel_time_serie in enumerate(y_train.T):
        print "Count %d of %d" % ((i + 1), y_train.shape[1])
        clf = SVC(kernel='linear', C=1.)
        feature_selection = SelectKBest(f_classif, k=50)
        anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])
        anova_svc.fit(X_train, pixel_time_serie)
        clfs.append(anova_svc)

# Bayesian Ridge
elif classifier == "bayesianridge":
    unique_classifier = False
    from sklearn.linear_model import BayesianRidge
    clfs = []
    for i, pixel_time_serie in enumerate(y_train.T):
        print "Count %d of %d" % ((i + 1), y_train.shape[1])
        clf = BayesianRidge(normalize=False, n_iter=1000)
        feature_selection = SelectKBest(f_classif, k=500)
        anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])
        anova_svc.fit(X_train, pixel_time_serie)
        clfs.append(anova_svc)

"""
# Learn
pipes = make_pipelines(SelectKBest(f_classif, k=100),
        SVC(kernel='linear', C=1.), X_train, y_train, n_features)

# Predict
y_pred = predict(pipes, X_test)
"""

### Prediction ################################################################

print "Calculating scores and outputs"

# Different processing for algorithms handling multiple outputs and those who
# do not
if unique_classifier:
    y_pred = clf.predict(X_test)
else:
    y_pred = []
    for i, x_test in enumerate(X_test):
        pred = []
        for c in clfs:
            pred.append(c.predict(x_test))
        pred = np.array(pred)
        y_pred.append(pred.squeeze())
    y_pred = np.array(y_pred)


# Revert scaled images if needed
if multi_scale:
    y_preds = np.split(y_pred, [n_pixels, 2 * n_pixels - y_rows,
        3 * n_pixels - y_rows - y_cols,
        4 * n_pixels - 2 * y_rows - 2 * y_cols + 1], axis=1)

    y_pred = y_preds[0]
    y_pred_tall = np.array([np.dot(height_backward,
        np.reshape(m, (9, 10))).flatten() for m in y_preds[1]])
    y_pred_large = np.array([np.dot(np.reshape(m, (10, 9)),
        width_backward).flatten() for m in y_preds[2]])
    y_pred_big = [np.dot(height_backward, np.reshape(m, (9, 9)))
        for m in y_preds[3]]
    y_pred_big = np.array([np.dot(np.reshape(m, (10, 9)),
        width_backward).flatten() for m in y_pred_big])

    if learn_fusion_params:
        y_pred = np.array([y_pred.T, y_pred_tall.T, y_pred_large.T,
            y_pred_big.T])
        y_pred = np.sum(y_pred.T * fusions, axis=2)
    else:
        y_pred = (.25 * y_pred + .25 * y_pred_tall + .25 * y_pred_large
            + .25 * y_pred_big)


"""


# Visualize results
y_pred = np.zeros(y_mask.shape)
y_pred[y_mask] = acc
plt.imshow(np.reshape(y_pred, [10, 10]), cmap=plt.cm.gray,
        interpolation='nearest')
plt.show()

print "Result : %d" % np.mean(acc)
"""

"""
Show brains !
"""

if generate_video:
    from matplotlib import animation
    fig = plt.figure()
    sp1 = plt.subplot(131)
    sp1.axis('off')
    sp2 = plt.subplot(132)
    sp2.axis('off')
    sp3 = plt.subplot(133)
    sp3.axis('off')
    ims = []
    for i, t in enumerate(y_pred):
        ims.append((
            sp1.imshow(np.reshape(y_test[i], (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest'),
            sp2.imshow(np.reshape(t, (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest'),
            sp3.imshow(np.reshape(t > 0.5, (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest')))

    im_ani = animation.ArtistAnimation(fig, ims, interval=1000)
    im_ani.save(generate_video)
