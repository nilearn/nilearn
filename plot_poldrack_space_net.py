"""
SpaceNet learner on Jimura "mixed gambles" dataset.

"""
# author: DOHMATOB Elvis Dopgima,
#         GRAMFORT Alexandre


### Load data ################################################################
from nilearn.datasets import fetch_mixed_gambles
data = fetch_mixed_gambles(n_subjects=16, make_Xy=True)
X, y, mask_img = data.X, data.y, data.mask_img


### Fit and predict ##########################################################
from nilearn.decoding import SpaceNetRegressor
penalties = ["smooth-lasso", "tv-l1"]
decoders = {}
for penalty in penalties:
    decoder = SpaceNetRegressor(mask=mask_img, penalty=penalty,
                                eps=1e-1,  # prefer large alphas
                                memory="cache", verbose=2)
    decoder.fit(X, y)  # fit
    decoders[penalty] = decoder


### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import mean_img
background_img = mean_img(X)
for penalty, decoder in decoders.iteritems():
    plot_stat_map(mean_img(decoder.coef_img_), background_img, title=penalty,
                  display_mode="yz", cut_coords=[20, -2])
    for f, ((best_alpha, best_l1_ratio), best_w) in enumerate(
        zip(decoder.best_model_params_, decoder.all_coef_[0])):
        plot_stat_map(decoder.masker_.inverse_transform(best_w),
                      background_img,
                      title=("%s: fold=%i, best alpha: %g, best "
                             "l1_ratio: %g" % (penalty, f, best_alpha,
                                               best_l1_ratio)),
                      display_mode="yz", cut_coords=[20, -2])

plt.show()
