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
import numpy as np

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

# let's create a figure with all the views for both hemispheres
views = ["lateral", "medial", "dorsal", "ventral", "anterior", "posterior"]
hemispheres = ["left", "right"]

fig, axes = plt.subplots(
    len(views),
    len(hemispheres),
    subplot_kw={"projection": "3d"},
    figsize=(4 * len(hemispheres), 4),
)
axes = np.atleast_2d(axes)

for view, ax_row in zip(views, axes):
    for ax, hemi in zip(ax_row, hemispheres):
        plotting.plot_surf(
            mean_img,
            part=hemi,
            view=view,
            figure=fig,
            axes=ax,
            title=f"mean image - {hemi} - {view}",
            colorbar=False,
            cmap="bwr",
            symmetric_cmap=True,
        )
fig.set_size_inches(6, 8)

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
# Now using the appropriate masker we can use a `Decoder` on surface data just
# as we do for volume images.

import numpy as np

from nilearn import decoding

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


vmax = max([np.absolute(dp).max() for dp in coef_img.data.parts.values()])
plotting.plot_surf(
    coef_img,
    cmap="cold_hot",
    vmin=-vmax,
    vmax=vmax,
    threshold=1e-6,
)
plt.show()
