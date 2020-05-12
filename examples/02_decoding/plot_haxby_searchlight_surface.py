"""
Cortical surface-based searchlight decoding
===========================================

This is a demo for surface-based searchlight decoding, as described in:
Chen, Y., Namburi, P., Elliott, L.T., Heinzle, J., Soon, C.S.,
Chee, M.W.L., and Haynes, J.-D. (2011). Cortical surface-based
searchlight decoding. NeuroImage 56, 582–592.

"""

#########################################################################
# Load Haxby dataset
# -------------------
import pandas as pd
from nilearn import datasets

# We fetch 2nd subject from haxby datasets (which is default)
haxby_dataset = datasets.fetch_haxby()

fmri_filename = haxby_dataset.func[0]
labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
y = labels['labels']
session = labels['chunks']

#########################################################################
# Restrict to faces and houses
# ------------------------------
from nilearn.image import index_img
condition_mask = y.isin(['face', 'house'])

fmri_img = index_img(fmri_filename, condition_mask)
y, session = y[condition_mask], session[condition_mask]

#########################################################################
# Surface bold response
# ----------------------
from nilearn import datasets, surface

# Fetch a coarse surface for speed
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
mesh_data = surface.load_surf_mesh(fsaverage.pial_left)

X = surface.vol_to_surf(fmri_img, fsaverage.pial_left)

#########################################################################
# Searchlight computation
# -----------------------
from nilearn import decoding
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

# Define cross-validation scheme
cv = KFold(n_splits=3, shuffle=False)

# Simple linear estimator preceeded by a normalization step
estimator = make_pipeline(StandardScaler(),
                          RidgeClassifier(alpha=10.))

# Define the searchlight "estimator"
searchlight = decoding.SurfSearchLight(mesh_data, radius=3, verbose=1,
                                       estimator=estimator, n_jobs=1, cv=cv)

# Run the searchlight decoding
# this can take time, depending mostly on the size of the mesh, the number
# of cross-validation splits and the radius
searchlight.fit(X, y)

#########################################################################
# Visualization
# -------------
from nilearn import plotting
plotting.plot_surf_stat_map(fsaverage.infl_left, searchlight.scores_,
                            hemi='right', colorbar=True, threshold=0.6,
                            bg_map=fsaverage.sulc_right,
                            title='Accuracy map, left hemisphere')
plotting.show()
