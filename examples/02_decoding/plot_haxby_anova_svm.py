"""
Decoding with ANOVA + SVM: face vs house in the Haxby dataset
=============================================================

This example does a simple but efficient decoding on the Haxby dataset:
using a feature selection, followed by an SVM.

"""

# %%
# Retrieve the files of the Haxby dataset
# ---------------------------------------
from nilearn import datasets

# By default 2nd subject will be fetched
haxby_dataset = datasets.fetch_haxby()
func_img = haxby_dataset.func[0]
# print basic information on the dataset
print(f"Mask nifti image (3D) is located at: {haxby_dataset.mask}")
print(f"Functional nifti image (4D) is located at: {func_img}")

# %%
# Load the behavioral data
# ------------------------
import pandas as pd

# Load target information as string and give a numerical identifier to each
behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
conditions = behavioral["labels"]

# Restrict the analysis to faces and places
from nilearn.image import index_img

condition_mask = behavioral["labels"].isin(["face", "house"])
conditions = conditions[condition_mask]
func_img = index_img(func_img, condition_mask)

# Confirm that we now have 2 conditions
print(conditions.unique())

# The number of the run is stored in the CSV file giving the behavioral data.
# We have to apply our run mask, to select only faces and houses.
run_label = behavioral["chunks"][condition_mask]

# %%
# :term:`ANOVA` pipeline with :class:`~nilearn.decoding.Decoder` object
# ---------------------------------------------------------------------
#
# Nilearn Decoder object aims to provide smooth user experience by acting as a
# pipeline of several tasks: preprocessing with NiftiMasker, reducing dimension
# by selecting only relevant features with :term:`ANOVA`
# -- a classical univariate feature selection based on F-test,
# and then decoding with different types of estimators
# (in this example is Support Vector Machine with a linear kernel)
# on nested cross-validation.
from nilearn.decoding import Decoder

# Here screening_percentile is set to 5 percent
mask_img = haxby_dataset.mask
decoder = Decoder(
    estimator="svc",
    mask=mask_img,
    smoothing_fwhm=4,
    standardize="zscore_sample",
    screening_percentile=5,
    scoring="accuracy",
)

# %%
# Fit the decoder and predict
# ---------------------------
decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)

# %%
# Obtain prediction scores via cross validation
# ---------------------------------------------
# Define the cross-validation scheme used for validation. Here we use a
# LeaveOneGroupOut cross-validation on the run group which corresponds to a
# leave a run out scheme, then pass the cross-validator object
# to the cv parameter of decoder.leave-one-session-out.
# For more details please take a look at:
# :ref:`sphx_glr_auto_examples_00_tutorials_plot_decoding_tutorial.py`.
from sklearn.model_selection import LeaveOneGroupOut

cv = LeaveOneGroupOut()

decoder = Decoder(
    estimator="svc",
    mask=mask_img,
    standardize="zscore_sample",
    screening_percentile=5,
    scoring="accuracy",
    cv=cv,
)
# Compute the prediction accuracy for the different folds (i.e. run)
decoder.fit(func_img, conditions, groups=run_label)

# Print the CV scores
print(decoder.cv_scores_["face"])

# %%
# Visualize the results
# ---------------------
# Look at the SVC's discriminating weights using
# :class:`~nilearn.plotting.plot_stat_map`
weight_img = decoder.coef_img_["face"]
from nilearn.plotting import plot_stat_map, show

plot_stat_map(weight_img, bg_img=haxby_dataset.anat[0], title="SVM weights")

show()

# %%
# Or we can plot the weights using :class:`~nilearn.plotting.view_img` as a
# dynamic html viewer
from nilearn.plotting import view_img

view_img(weight_img, bg_img=haxby_dataset.anat[0], title="SVM weights", dim=-1)

# %%
# Saving the results as a Nifti file may also be important
from pathlib import Path

output_dir = Path.cwd() / "results" / "plot_haxby_anova_svm"
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output will be saved to: {output_dir}")
weight_img.to_filename(output_dir / "haxby_face_vs_house.nii")

# sphinx_gallery_dummy_images=1
