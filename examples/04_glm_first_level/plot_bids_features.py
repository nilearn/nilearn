"""
First level analysis of a complete BIDS dataset from openneuro
==============================================================


Full step-by-step example of fitting a GLM to perform a first level analysis
in an openneuro :term:`BIDS` dataset. We demonstrate how :term:`BIDS`
derivatives can be exploited to perform a simple one subject analysis with
minimal code. Details about the :term:`BIDS` standard are available at
`http://bids.neuroimaging.io/ <http://bids.neuroimaging.io/>`_.
We also demonstrate how to download individual groups of files from the
Openneuro s3 bucket.

More specifically:

1. Download an :term:`fMRI` :term:`BIDS` dataset
   with derivatives from openneuro.
2. Extract first level model objects automatically
   from the :term:`BIDS` dataset.
3. Demonstrate Quality assurance of Nistats estimation against available FSL.
   estimation in the openneuro dataset.
4. Display contrast plot and uncorrected first level statistics table report.
"""


##############################################################################
# Fetch openneuro BIDS dataset
# ----------------------------
# We download one subject from the stopsignal task
# in the ds000030 V4 :term:`BIDS` dataset available in openneuro.
# This dataset contains the necessary information to run a statistical analysis
# using Nilearn. The dataset also contains statistical results from a previous
# FSL analysis that we can employ for comparison with the Nilearn estimation.
from nilearn.datasets import (
    fetch_ds000030_urls,
    fetch_openneuro_dataset,
    select_from_index,
)

_, urls = fetch_ds000030_urls()

exclusion_patterns = [
    "*group*",
    "*phenotype*",
    "*mriqc*",
    "*parameter_plots*",
    "*physio_plots*",
    "*space-fsaverage*",
    "*space-T1w*",
    "*dwi*",
    "*beh*",
    "*task-bart*",
    "*task-rest*",
    "*task-scap*",
    "*task-task*",
]
urls = select_from_index(
    urls, exclusion_filters=exclusion_patterns, n_subjects=1
)

data_dir, _ = fetch_openneuro_dataset(urls=urls)

##############################################################################
# Obtain FirstLevelModel objects automatically and fit arguments
# --------------------------------------------------------------
# From the dataset directory we automatically obtain FirstLevelModel objects
# with their subject_id filled from the :term:`BIDS` dataset.
# Moreover we obtain,
# for each model, the list of run images and their respective events and
# confound regressors. Those are inferred from the confounds.tsv files
# available in the :term:`BIDS` dataset.
# To get the first level models we have to specify the dataset directory,
# the task_label and the space_label as specified in the file names.
# We also have to provide the folder with the desired derivatives, that in this
# case were produced by the :term:`fMRIPrep` :term:`BIDS` app.
from nilearn.glm.first_level import first_level_from_bids

task_label = "stopsignal"
space_label = "MNI152NLin2009cAsym"
derivatives_folder = "derivatives/fmriprep"
(
    models,
    models_run_imgs,
    models_events,
    models_confounds,
) = first_level_from_bids(
    data_dir,
    task_label,
    space_label,
    smoothing_fwhm=5.0,
    derivatives_folder=derivatives_folder,
)

#############################################################################
# Access the model and model arguments of the subject and process events.

model, imgs, events, confounds = (
    models[0],
    models_run_imgs[0],
    models_events[0],
    models_confounds[0],
)
subject = f"sub-{model.subject_label}"
model.minimize_memory = False  # override default

import os

from nilearn.interfaces.fsl import get_design_from_fslmat

fsl_design_matrix_path = os.path.join(
    data_dir, "derivatives", "task", subject, "stopsignal.feat", "design.mat"
)
design_matrix = get_design_from_fslmat(
    fsl_design_matrix_path, column_names=None
)

#############################################################################
# We identify the columns of the Go and StopSuccess conditions of the
# design matrix inferred from the FSL file, to use them later for contrast
# definition.
design_columns = [
    f"cond_{int(i):02}" for i in range(len(design_matrix.columns))
]
design_columns[0] = "Go"
design_columns[4] = "StopSuccess"
design_matrix.columns = design_columns

############################################################################
# First level model estimation (one subject)
# ------------------------------------------
# We fit the first level model for one subject.
model.fit(imgs, design_matrices=[design_matrix])

#############################################################################
# Then we compute the StopSuccess - Go contrast. We can use the column names
# of the design matrix.
z_map = model.compute_contrast("StopSuccess - Go")

#############################################################################
# We show the agreement between the Nilearn estimation and the FSL estimation
# available in the dataset.
import nibabel as nib

fsl_z_map = nib.load(
    os.path.join(
        data_dir,
        "derivatives",
        "task",
        subject,
        "stopsignal.feat",
        "stats",
        "zstat12.nii.gz",
    )
)

import matplotlib.pyplot as plt
from nilearn import plotting
from scipy.stats import norm

plotting.plot_glass_brain(
    z_map,
    colorbar=True,
    threshold=norm.isf(0.001),
    title='Nilearn Z map of "StopSuccess - Go" (unc p<0.001)',
    plot_abs=False,
    display_mode="ortho",
)
plotting.plot_glass_brain(
    fsl_z_map,
    colorbar=True,
    threshold=norm.isf(0.001),
    title='FSL Z map of "StopSuccess - Go" (unc p<0.001)',
    plot_abs=False,
    display_mode="ortho",
)
plt.show()

from nilearn.plotting import plot_img_comparison

plot_img_comparison(
    [z_map], [fsl_z_map], model.masker_, ref_label="Nilearn", src_label="FSL"
)
plt.show()

#############################################################################
# Simple statistical report of thresholded contrast
# -------------------------------------------------
# We display the contrast plot and table with cluster information
from nilearn.plotting import plot_contrast_matrix

plot_contrast_matrix("StopSuccess - Go", design_matrix)
plotting.plot_glass_brain(
    z_map,
    colorbar=True,
    threshold=norm.isf(0.001),
    plot_abs=False,
    display_mode="z",
    figure=plt.figure(figsize=(4, 4)),
)
plt.show()

###############################################################################
# We can get a latex table from a Pandas Dataframe for display and publication
# purposes
from nilearn.reporting import get_clusters_table

table = get_clusters_table(z_map, norm.isf(0.001), 10)
print(table.to_latex())

#########################################################################
# Generating a report
# -------------------
# Using the computed FirstLevelModel and contrast information,
# we can quickly create a summary report.
from nilearn.reporting import make_glm_report

report = make_glm_report(
    model=model,
    contrasts="StopSuccess - Go",
)

#########################################################################
# We have several ways to access the report:

# report  # This report can be viewed in a notebook
# report.save_as_html('report.html')
# report.open_in_browser()

#########################################################################
# Saving model outputs to disk
# ----------------------------
from nilearn.interfaces.bids import save_glm_to_bids

save_glm_to_bids(
    model,
    contrasts="StopSuccess - Go",
    contrast_types={"StopSuccess - Go": "t"},
    out_dir="derivatives/nilearn_glm/",
    prefix=f"{subject}_task-stopsignal",
)

#########################################################################
# View the generated files
from glob import glob

print("\n".join(sorted(glob("derivatives/nilearn_glm/*"))))
