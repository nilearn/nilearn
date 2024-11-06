"""
A short demo of the surface images & maskers
============================================
"""

from nilearn._utils.helpers import check_matplotlib

check_matplotlib()

# %%
import matplotlib.pyplot as plt
import numpy as np

from nilearn.experimental.surface import SurfaceMasker, fetch_nki
from nilearn.plotting import plot_matrix, show, plot_surf, plot_surf_roi

img = fetch_nki()[0]
print(f"NKI image: {img}")

masker = SurfaceMasker()
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
        plot_surf(
            surf_map=mean_img,
            hemi=hemi,
            view=view,
            figure=fig,
            axes=ax,
            title=f"mean image - {hemi} - {view}",
            colorbar=False,
            cmap="bwr",
            symmetric_cmap=True,
            bg_on_data=True,
        )
fig.set_size_inches(6, 8)

show()

# %%
# Connectivity with a surface atlas and `SurfaceLabelsMasker`
# -----------------------------------------------------------
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_atlas_surf_destrieux
from nilearn.experimental.surface import (
    SurfaceImage,
    SurfaceLabelsMasker,
    fetch_destrieux,
    load_fsaverage,
    load_fsaverage_data,
)

fsaverage = load_fsaverage("fsaverage5")
destrieux = fetch_atlas_surf_destrieux()
labels_img = SurfaceImage(
    mesh=fsaverage["pial"],
    data={
        "left": destrieux["map_left"],
        "right": destrieux["map_right"],
    },
)
label_names = [x.decode("utf-8") for x in destrieux.labels]

# for our plots we will be using the fsaverage sulcal data as background map
fsaverage_sulcal = load_fsaverage_data(data_type="sulcal")

img = fetch_nki()[0]
print(f"NKI image: {img}")

labels_img, label_names = fetch_destrieux()
print(f"Destrieux image: {labels_img}")
plot_surf_roi(
    roi_map=labels_img,
    avg_method="median",
    view="lateral",
    bg_on_data=True,
    bg_map=fsaverage_sulcal,
    darkness=0.5,
    title="Destrieux atlas",
)

labels_masker = SurfaceLabelsMasker(labels_img, label_names).fit()

report = labels_masker.generate_report()
# This report can be viewed in a notebook
report

# We have several ways to access the report:
# report.open_in_browser()

masked_data = labels_masker.transform(img)
print(f"Masked data shape: {masked_data.shape}")

# or we can save as an html file
from pathlib import Path

output_dir = Path.cwd() / "results" / "plot_surface_image_and_maskers"
output_dir.mkdir(exist_ok=True, parents=True)
report.save_as_html(output_dir / "report.html")

# %%
connectome = ConnectivityMeasure(kind="correlation").fit([masked_data]).mean_
plot_matrix(connectome, labels=labels_masker.label_names_)

show()


# %%
# Using the `Decoder`
# -------------------
# Now using the appropriate masker we can use a `Decoder` on surface data just
# as we do for volume images.
import numpy as np

from nilearn.decoding import Decoder

img = fetch_nki()[0]

rng = np.random.RandomState(0)

y = rng.choice([0, 1], replace=True, size=img.shape[0])

decoder = Decoder(
    mask=SurfaceMasker(),
    param_grid={"C": [0.01, 0.1]},
    cv=3,
    screening_percentile=1,
)
decoder.fit(img, y)
print("CV scores:", decoder.cv_scores_)

plot_surf(
    decoder.coef_img_[0],
    threshold=1e-6,
    bg_map=fsaverage_sulcal,
    bg_on_data=True,
    colorbar=True,
    cmap="black_red",
    vmin=0,
)
show()

# %%
# Decoding with a scikit-learn `Pipeline`
# ---------------------------------------
from sklearn import feature_selection, linear_model, pipeline, preprocessing

img = fetch_nki()[0]
y = rng.normal(size=img.shape[0])

decoder = pipeline.make_pipeline(
    SurfaceMasker(),
    preprocessing.StandardScaler(),
    feature_selection.SelectKBest(
        score_func=feature_selection.f_regression, k=500
    ),
    linear_model.Ridge(),
)
decoder.fit(img, y)

coef_img = decoder[:-1].inverse_transform(np.atleast_2d(decoder[-1].coef_))

vmax = max([np.absolute(dp).max() for dp in coef_img.data.parts.values()])
plot_surf(
    coef_img,
    cmap="cold_hot",
    vmin=-vmax,
    vmax=vmax,
    threshold=1e-6,
    bg_map=fsaverage_sulcal,
    bg_on_data=True,
    colorbar=True,
)
show()
