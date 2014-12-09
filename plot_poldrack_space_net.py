"""
SpaceNet learner on Poldrack's (Jimura) mixed gambles dataset.

"""
# author: DOHMATOB Elvis Dopgima,
#         GRAMFORT Alexandre


### Load data ################################################################
from nilearn.datasets import fetch_mixed_gambles
data = fetch_mixed_gambles(n_subjects=16, make_Xy=True)
X, y, mask_img = data.X, data.y, data.mask_img

# prepare input data for learner
n_samples = len(X)
n_samples_train = n_samples * 8 / 10
X_train = X[:n_samples_train]
y_train = y[:n_samples_train]
X_test = X[n_samples_train:]
y_test = y[n_samples_train:]


### Fit and predict ##########################################################
from nilearn.decoding import SpaceNetRegressor
penalties = ["smooth-lasso", "TV-L1"]
decoders = {}
for penalty in penalties:
    decoder = SpaceNetRegressor(mask=mask_img, penalty=penalty, verbose=2,
                                n_jobs=20)
    decoder.fit(X_train, y_train)  # fit
    decoders[penalty] = decoder


### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map
background_img = mean_img(X_train)
for penalty, decoder in decoders.iteritems():
    plot_stat_map(mean_img(decoder.coef_img_), background_img, title=penalty,
                  display_mode="yz", cut_coords=[20, -2])
plt.show()
