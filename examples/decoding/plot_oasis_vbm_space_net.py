"""
Voxel-Based Morphometry on Oasis dataset with Space-Net prior
=============================================================

"""
# Authors: DOHMATOB Elvis
#          FRITSCH Virgile


### Load Oasis dataset ########################################################
import numpy as np
from nilearn import datasets
n_subjects = None  # more subjects requires more memory
dataset_files = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)
age = np.array(age)
X = np.array(dataset_files.gray_matter_maps)

### Split data into training set and test set
from sklearn.utils import check_random_state
from sklearn.cross_validation import train_test_split
rng = check_random_state(42)
X_train, X_test, y_train, y_test = train_test_split(X, age, train_size=.7,
                                                    random_state=rng)

### Sort test data for better visualization (trend, etc.)
perm = np.argsort(y_test)[::-1]
y_test = y_test[perm]
X_test = X_test[perm]


### Fit and predict ###########################################################
from nilearn.decoding import SpaceNetRegressor
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
for penalty in ['tv-l1', 'smooth-lasso']:
    decoder = SpaceNetRegressor(memory="cache", penalty=penalty, verbose=2)
    decoder.fit(X_train, y_train)  # fit
    coef_img = decoder.coef_img_
    y_pred = decoder.predict(X_test).ravel()  # predict
    mse = np.mean(np.abs(y_test - y_pred))

    ### Visualization #########################################################
    # weights map
    background_img = X[0]
    plot_stat_map(coef_img, background_img, title="%s weights" % penalty,
                  display_mode="z")

    # quality of predictions
    plt.figure()
    plt.suptitle("%s: Mean Absolute Error %.2f years" % (penalty, mse))
    linewidth = 3
    ax1 = plt.subplot('211')
    ax1.plot(y_test, label="True age", linewidth=linewidth)
    ax1.plot(y_pred, '--', c="g", label="Predicted age", linewidth=linewidth)
    ax1.set_ylabel("age")
    plt.legend(loc="best")
    ax2 = plt.subplot("212")
    ax2.plot(y_test - y_pred, label="True age - predicted age",
             linewidth=linewidth)
    ax2.set_xlabel("subject")
    plt.legend(loc="best")

plt.show()
