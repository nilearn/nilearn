"""
Voxel-Based Morphometry on Oasis dataset with Space-Net prior
=============================================================

"""
# Authors: DOHMATOB Elvis
#          FRITSCH Virgile

n_subjects = 100  # more subjects requires more memory
n_subjects_train = 100


### Load Oasis dataset ########################################################
import numpy as np
from nilearn import datasets
dataset_files = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
age = np.array(age)
perm = np.argsort(age)[::-1]
age = age[perm]
X = np.array(dataset_files.gray_matter_maps)[perm]
X_train = X[:n_subjects_train]
y_train = age[:n_subjects_train]
# X_test = X[n_subjects_train:]
# y_test = age[n_subjects_train:]
X_test = X_train.copy()
y_test = y_train.copy()


### Fit and predict ###########################################################
from nilearn.decoding import SpaceNetRegressor
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
for penalty in ['tv-l1', 'smooth-lasso']:
    decoder = SpaceNetRegressor(memory="cache", penalty=penalty, verbose=2,
                                n_jobs=20, standardize=False)
    decoder.fit(X_train, y_train)  # fit
    coef_img = decoder.coef_img_
    y_pred = decoder.predict(X_test).ravel()  # predict

    ### Visualization #########################################################
    # weights map
    background_img = X[0]
    plot_stat_map(coef_img, background_img, title="%s weights" % penalty,
                  display_mode="z")

    # quality of predictions
    plt.figure()
    plt.suptitle(penalty)
    linewidth = 3
    ax1 = plt.subplot('211')
    ax1.plot(y_test, label="True age", linewidth=linewidth)
    ax1.plot(y_pred, '--', c="g", label="Fitted age", linewidth=linewidth)
    ax1.set_ylabel("age")
    plt.legend(loc="best")
    ax2 = plt.subplot("212")
    ax2.plot(y_test - y_pred, label="True age - fitted age",
             linewidth=linewidth)
    ax2.set_xlabel("subject")
    plt.legend(loc="best")

plt.show()
