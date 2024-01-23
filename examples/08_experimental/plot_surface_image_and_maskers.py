"""
A short demo of the surface images & maskers
============================================

copied from the nilearn sandbox discussion, to be transformed into tests &
examples

.. note::

    this example is meant to support discussion around a tentative API for
    surface images in nilearn. This functionality is provided by the
    nilearn.experimental.surface module; it is still incomplete and subject to
    change without a deprecation cycle. Please participate in the discussion on
    GitHub!

"""
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

# %%

from nilearn.experimental import plotting, surface
from nilearn.plotting import plot_matrix

img = surface.fetch_nki()[0]
print(f"NKI image: {img}")

masker = surface.SurfaceMasker()
masked_data = masker.fit_transform(img)
print(f"Masked data shape: {masked_data.shape}")

mean_data = masked_data.mean(axis=0)
mean_img = masker.inverse_transform(mean_data)
print(f"Image mean: {mean_img}")

plotting.plot_surf(mean_img)
plt.show()

# %%
# Connectivity with a surface atlas and `SurfaceLabelsMasker`
# -----------------------------------------------------------
from nilearn import connectome

img = surface.fetch_nki()[0]
print(f"NKI image: {img}")

labels_img, label_names = surface.fetch_destrieux()
print(f"Destrieux image: {labels_img}")
plotting.plot_surf(
    labels_img,
    views=["lateral", "medial"],
    cmap="gist_ncar",
    avg_method="median",
)

labels_masker = surface.SurfaceLabelsMasker(labels_img, label_names).fit()
masked_data = labels_masker.transform(img)
print(f"Masked data shape: {masked_data.shape}")

connectome = (
    connectome.ConnectivityMeasure(kind="correlation").fit([masked_data]).mean_
)
plot_matrix(connectome, labels=labels_masker.label_names_)

plt.show()


# %%
# Using the `Decoder`
# -------------------
import numpy as np

from nilearn import decoding
from nilearn._utils import param_validation

# %%
# The following is just disabling a couple of checks performed by the decoder
# that would force us to use a `NiftiMasker`.


def monkeypatch_masker_checks():
    def adjust_screening_percentile(screening_percentile, *args, **kwargs):
        return screening_percentile

    param_validation.adjust_screening_percentile = adjust_screening_percentile


monkeypatch_masker_checks()

# %%
# Now using the appropriate masker we can use a `Decoder` on surface data just
# as we do for volume images.

img = surface.fetch_nki()[0]
y = np.random.RandomState(0).choice([0, 1], replace=True, size=img.shape[0])

decoder = decoding.Decoder(
    mask=surface.SurfaceMasker(),
    param_grid={"C": [0.01, 0.1]},
    cv=3,
    screening_percentile=1,
)
decoder.fit(img, y)
print("CV scores:", decoder.cv_scores_)

plotting.plot_surf(decoder.coef_img_[0], threshold=1e-6)
plt.show()

# %%
# Decoding with a scikit-learn `Pipeline`
# ---------------------------------------
import numpy as np
from sklearn import feature_selection, linear_model, pipeline, preprocessing

img = surface.fetch_nki()[0]
y = np.random.RandomState(0).normal(size=img.shape[0])

decoder = pipeline.make_pipeline(
    surface.SurfaceMasker(),
    preprocessing.StandardScaler(),
    feature_selection.SelectKBest(
        score_func=feature_selection.f_regression, k=500
    ),
    linear_model.Ridge(),
)
decoder.fit(img, y)

coef_img = decoder[:-1].inverse_transform(np.atleast_2d(decoder[-1].coef_))


vmax = max([np.absolute(dp).max() for dp in coef_img.data.values()])
plotting.plot_surf(
    coef_img,
    cmap="cold_hot",
    vmin=-vmax,
    vmax=vmax,
    threshold=1e-6,
)
plt.show()
