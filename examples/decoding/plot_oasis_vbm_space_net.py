"""
Voxel-Based Morphometry on Oasis dataset with Space-Net prior
=============================================================

Predicting age from gray-matter concentration maps from OASIS dataset
"""
# Authors: DOHMATOB Elvis
#          FRITSCH Virgile


### Load Oasis dataset ########################################################
import numpy as np
from nilearn import datasets
dataset_files = datasets.fetch_oasis_vbm()
age = dataset_files.ext_vars['age'].astype(float)
age = np.array(age)
gm_imgs = np.array(dataset_files.gray_matter_maps)

### Split data into training set and test set
from sklearn.utils import check_random_state
from sklearn.cross_validation import train_test_split
rng = check_random_state(42)
gm_imgs_train, gm_imgs_test, age_train, age_test = train_test_split(
    gm_imgs, age, train_size=.6, random_state=rng)

### Sort test data for better visualization (trend, etc.)
perm = np.argsort(age_test)[::-1]
age_test = age_test[perm]
gm_imgs_test = gm_imgs_test[perm]


### Fit and predict ###########################################################
from nilearn.decoding import SpaceNetRegressor
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
for penalty in ['tv-l1', 'graph-net']:
    # To save time (because these are anat images with many voxels), we include
    # only the 5-percent voxels most correlated with the age variable to fit.
    # Also, we set memory_level=2 so that more of the intermediate computations
    # are cached. Also, you may pass and n_jobs=<some_high_value> to the
    # SpaceNetRegressor class, to take advantage of a multi-core system.
    decoder = SpaceNetRegressor(memory="cache", penalty=penalty,
                                screening_percentile=5, memory_level=2)
    decoder.fit(gm_imgs_train, age_train)  # fit
    coef_img = decoder.coef_img_
    y_pred = decoder.predict(gm_imgs_test).ravel()  # predict
    mse = np.mean(np.abs(age_test - y_pred))

    ### Visualization #########################################################
    # weights map
    background_img = gm_imgs[0]
    plot_stat_map(coef_img, background_img, title="%s weights" % penalty,
                  display_mode="z", cut_coords=1)

    # quality of predictions
    plt.figure()
    plt.suptitle("%s: Mean Absolute Error %.2f years" % (penalty, mse))
    linewidth = 3
    ax1 = plt.subplot('211')
    ax1.plot(age_test, label="True age", linewidth=linewidth)
    ax1.plot(y_pred, '--', c="g", label="Predicted age", linewidth=linewidth)
    ax1.set_ylabel("age")
    plt.legend(loc="best")
    ax2 = plt.subplot("212")
    ax2.plot(age_test - y_pred, label="True age - predicted age",
             linewidth=linewidth)
    ax2.set_xlabel("subject")
    plt.legend(loc="best")

plt.show()
