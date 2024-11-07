"""
Cortical surface-based searchlight decoding
===========================================

This is a demo for surface-based searchlight decoding,
as described in :footcite:t:`Chen2011`.
"""

# %%
# Load Haxby dataset
# ------------------
import pandas as pd

from nilearn import datasets

# We fetch 2nd subject from haxby datasets (which is default)
haxby_dataset = datasets.fetch_haxby()

fmri_filename = haxby_dataset.func[0]
labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
y = labels["labels"]
run = labels["chunks"]

# %%
# Restrict to faces and houses
# ----------------------------
from nilearn.image import index_img

condition_mask = y.isin(["face", "house"])

fmri_img = index_img(fmri_filename, condition_mask)
y, run = y[condition_mask], run[condition_mask]

# %%
# Surface :term:`BOLD` response
# -----------------------------
# Fetch a coarse surface of the left hemisphere only for speed
# and average voxels 5 mm close to the 3d pial surface.

from sklearn import neighbors

from nilearn import datasets
from nilearn.experimental.surface import (
    SurfaceImage,
    load_fsaverage,
)

fsaverage = load_fsaverage()
fmri_img_surf = SurfaceImage.from_volume(
    mesh=fsaverage["pial"], volume_img=fmri_img, radius=5
)


# %%
# Searchlight computation
# -----------------------
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nilearn.decoding.searchlight import search_light

scores = {}

for hemi in ["left", "right"]:
    # To define the :term:`BOLD responses
    # to be included within each searchlight "sphere"
    # we define an adjacency matrix
    # based on the inflated surface vertices
    # such that nearby surfaces are concatenated
    # within the same searchlight.
    coordinates = fsaverage["inflated"].parts[hemi].coordinates
    nn = neighbors.NearestNeighbors(radius=3)
    adjacency = nn.fit(coordinates).radius_neighbors_graph(coordinates).tolil()

    # Simple linear estimator preceded by a normalization step
    estimator = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))

    # Define cross-validation scheme
    cv = KFold(n_splits=3, shuffle=False)

    X = fmri_img_surf.data.parts["hemi"]

    # Cross-validated search light
    scores[hemi] = search_light(X, y, estimator, adjacency, cv=cv, n_jobs=2)

# %%
# Visualization
# -------------
from nilearn import plotting
from nilearn.experimental.surface import load_fsaverage_data

score_img = SurfaceImage(mesh=fsaverage["inflated"], data=scores)

chance = 0.5
for hemi in ["left", "right"]:
    score_img.data.parts[hemi] = score_img.data.parts[hemi] - 0.5

fsaverage_data = load_fsaverage_data(mesh_type="inflated", data_type="sulcal")

plotting.plot_surf_stat_map(
    stat_map=score_img,
    view="medial",
    hemi="left",
    colorbar=True,
    threshold=0.6,
    bg_map=fsaverage_data,
    title="Accuracy map, left hemisphere",
)
plotting.show()

# %%
# References
# ----------
#
#  .. footbibliography::
