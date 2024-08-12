"""
Simple example of two-runs fMRI surface based model fitting
===========================================================

.. warning::

    This is an adaption of
    :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_two_runs_model.py`
    to use make it work with the new experimental surface API.

Here, we will go through a full step-by-step example
of fitting a surface based GLM
to experimental data and visualizing the results.
This is done on two runs of one subject of the FIAC dataset.

For more details on the data,
please see experiment 2 in :footcite:t:`Dehaene2006`.

Here are the steps we will go through:

1. Set up the GLM
2. Compute a range of contrasts across both runs
3. Generate a report

Technically, this example shows how to handle two runs
that contain the same experimental conditions.
The model directly returns a fixed effect
of the statistics across the two runs.
"""

# %%
# Create an output ``results`` in the current working directory.
from pathlib import Path

output_dir = Path.cwd() / "results" / "plot_two_runs_model_experimental"
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output will be saved to: {output_dir}")


# %%
# Set up the GLM
# --------------
# Inspecting 'data', we note that there are two runs.
# We will retain those two runs in a list of 4D img objects.
from nilearn.datasets import func

data = func.fetch_fiac_first_level()
fmri_imgs = [data["func1"], data["func2"]]

# %%
# project data to surface
from nilearn import surface
from nilearn.experimental.surface import SurfaceImage, load_fsaverage

fsaverage5 = load_fsaverage()

run_imgs = []
for volume_image in [data["func1"], data["func2"]]:
    texture_left = surface.vol_to_surf(
        volume_image, fsaverage5["pial"].parts["left"]
    )
    texture_right = surface.vol_to_surf(
        volume_image, fsaverage5["pial"].parts["right"]
    )
    run_imgs.append(
        SurfaceImage(
            mesh=fsaverage5["pial"],
            data={
                "left": texture_left.T,
                "right": texture_right.T,
            },
        )
    )


import numpy as np

# %%
# The design matrices were pre-computed,
# we simply put them in a list of DataFrames.
import pandas as pd

design_files = [data["design_matrix1"], data["design_matrix2"]]
design_matrices = [pd.DataFrame(np.load(df)["X"]) for df in design_files]


# %%
# Initialize and run the GLM
# --------------------------
# First, we need to specify the model before fitting it to the data.
# Note that a brain mask was provided in the dataset,
# so that is what we will use.
from nilearn.glm.first_level import FirstLevelModel

fmri_glm = FirstLevelModel(
    minimize_memory=True,
)

# %%
# Compute the fixed effects statistics
# using the preprocessed data of both runs.
#
# Since we can assume that the design matrices of both runs
# have the same columns, in the same order,
# we can again reuse the first run's contrast vector.
fmri_glm_multirun = fmri_glm.fit(run_imgs, design_matrices=design_matrices)


# %%
# You may note that the results are the same as the first fixed effects
# analysis, but with a lot less code.

# %%
# Compute a range of contrasts across both runs
# ---------------------------------------------
# It may be useful to investigate a number of contrasts.
# Therefore, we will move beyond the original contrast of interest
# and both define and compute several.

# %%
# Contrast specification
n_columns = design_matrices[0].shape[1]
contrasts = {
    "SStSSp_minus_DStDSp": np.array([[1, 0, 0, -1]]),
    "DStDSp_minus_SStSSp": np.array([[-1, 0, 0, 1]]),
    "DSt_minus_SSt": np.array([[-1, -1, 1, 1]]),
    "DSp_minus_SSp": np.array([[-1, 1, -1, 1]]),
    "DSt_minus_SSt_for_DSp": np.array([[0, -1, 0, 1]]),
    "DSp_minus_SSp_for_DSt": np.array([[0, 0, -1, 1]]),
    "Deactivation": np.array([[-1, -1, -1, -1, 4]]),
    "Effects_of_interest": np.eye(n_columns)[:5, :],  # An F-contrast
}


# %%
# Generating a report
# -------------------
# Since we have already computed the FirstLevelModel
# and have a number of contrasts,
# we can quickly create a summary report.

from nilearn.experimental.surface import load_fsaverage_data

fsaverage_sulcal = load_fsaverage_data(
    data_type="sulcal", mesh_type="inflated"
)

report = fmri_glm_multirun.generate_report(
    contrasts, title="two-runs fMRI model fitting", bg_img=fsaverage_sulcal
)

# %%
# We have several ways to access the report:

# report  # This report can be viewed in a notebook
# report.open_in_browser()

# or we can save as an html file
report.save_as_html(output_dir / "report.html")
report

# %%
# References
# ----------
#
#  .. footbibliography::
