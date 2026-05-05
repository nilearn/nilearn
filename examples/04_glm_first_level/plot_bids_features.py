"""
First level analysis of a complete BIDS dataset from openneuro
==============================================================

Full step-by-step example of fitting a :term:`GLM`
to perform a first level analysis in an openneuro :term:`BIDS` dataset.
We demonstrate how :term:`BIDS`
derivatives can be exploited to perform a simple one subject analysis with
minimal code. Details about the :term:`BIDS` standard are available at
`https://bids.neuroimaging.io/ <https://bids.neuroimaging.io/>`_.
We also demonstrate how to download individual groups of files from the
Openneuro s3 bucket.

More specifically:

1. Download an :term:`fMRI` :term:`BIDS` dataset
   with derivatives from openneuro.
2. Extract first level model objects automatically
   from the :term:`BIDS` dataset.
3. Demonstrate Quality assurance of Nilearn estimation against available FSL.
   estimation in the openneuro dataset.
4. Display contrast plot and uncorrected first level statistics table report.
"""

# %%
# Fetch openneuro :term:`BIDS` dataset
# ------------------------------------
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

# %%
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
    n_jobs=2,
    verbose=1,
)

# %%
# Access the model and model arguments of the subject and process events.

model, imgs, events, confounds = (
    models[0],
    models_run_imgs[0],
    models_events[0],
    models_confounds[0],
)
subject = f"sub-{model.subject_label}"
model.minimize_memory = False  # override default

from pathlib import Path

from nilearn.interfaces.fsl import get_design_from_fslmat

fsl_design_matrix_path = (
    Path(data_dir)
    / "derivatives"
    / "task"
    / subject
    / "stopsignal.feat"
    / "design.mat"
)
design_matrix = get_design_from_fslmat(
    fsl_design_matrix_path, column_names=None
)

# %%
# We identify the columns of the Go and StopSuccess conditions of the
# design matrix inferred from the FSL file, to use them later for contrast
# definition.
design_columns = [
    f"cond_{int(i):02}" for i in range(len(design_matrix.columns))
]
design_columns[0] = "Go"
design_columns[4] = "StopSuccess"
design_matrix.columns = design_columns

# %%
# First level model estimation (one subject)
# ------------------------------------------
# We fit the first level model for one subject.
model.fit(imgs, design_matrices=[design_matrix])

# %%
# Then we compute the StopSuccess - Go contrast. We can use the column names
# of the design matrix.
z_map = model.compute_contrast("StopSuccess - Go")

# %%
# Visualize results
# -----------------
# Let's have a look at the Nilearn estimation
# and the FSL estimation available in the dataset.
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.stats import norm

from nilearn.plotting import plot_glass_brain, show

fsl_z_map = nib.load(
    Path(data_dir)
    / "derivatives"
    / "task"
    / subject
    / "stopsignal.feat"
    / "stats"
    / "zstat12.nii.gz"
)

plot_glass_brain(
    z_map,
    threshold=norm.isf(0.001),
    title='Nilearn Z map of "StopSuccess - Go" (unc p<0.001)',
    plot_abs=False,
    display_mode="ortho",
)
plot_glass_brain(
    fsl_z_map,
    threshold=norm.isf(0.001),
    title='FSL Z map of "StopSuccess - Go" (unc p<0.001)',
    plot_abs=False,
    display_mode="ortho",
)

# %%
# We show the agreement between the 2 estimations.

from nilearn.plotting import plot_bland_altman, plot_img_comparison

plot_img_comparison(
    z_map, fsl_z_map, model.masker_, ref_label="Nilearn", src_label="FSL"
)

plot_bland_altman(
    z_map, fsl_z_map, model.masker_, ref_label="Nilearn", src_label="FSL"
)

show()

# %%
# Saving model outputs to disk
# ----------------------------
#
# It can be useful to quickly generate a portable, ready-to-view report with
# most of the pertinent information.
# We can do this by saving the output of the GLM to disk
# including an HTML report.
# This is easy to do if you have a fitted model and the list of contrasts,
# which we do here.
#
from nilearn.glm import save_glm_to_bids

output_dir = Path.cwd() / "results" / "plot_bids_features"
output_dir.mkdir(exist_ok=True, parents=True)

stat_threshold = norm.isf(0.001)

save_glm_to_bids(
    model,
    contrasts="StopSuccess - Go",
    contrast_types={"StopSuccess - Go": "t"},
    out_dir=output_dir / "derivatives" / "nilearn_glm",
    threshold=stat_threshold,
    cluster_threshold=10,
)

# %%
# View the generated files
files = sorted((output_dir / "derivatives" / "nilearn_glm").glob("**/*"))
print("\n".join([str(x.relative_to(output_dir)) for x in files]))

# %%
# Simple statistical report of thresholded contrast
# -------------------------------------------------
# We display the :term:`contrast` plot and table with cluster information.
#
# Here we will the image directly from the results saved to disk.
#
from nilearn.plotting import plot_contrast_matrix

plot_contrast_matrix("StopSuccess - Go", design_matrix)

z_map = (
    output_dir
    / "derivatives"
    / "nilearn_glm"
    / "sub-10159"
    / (
        "sub-10159_task-stopsignal_space-MNI152NLin2009cAsym_"
        "contrast-stopsuccessMinusGo_stat-z_statmap.nii.gz"
    )
)

plot_glass_brain(
    z_map,
    threshold=stat_threshold,
    plot_abs=False,
    display_mode="z",
    figure=plt.figure(figsize=(4, 4)),
)
show()

# %%
# The saved results include a table of activated clusters.
#
# .. note::
#
#     This table can also be generated by using the function
#     :func:`nilearn.reporting.get_clusters_table`.
#
#     .. code-block:: python
#
#        from nilearn.reporting import get_clusters_table
#
#        table = get_clusters_table(
#            z_map,
#            stat_threshold=norm.isf(0.001),
#            cluster_threshold=10
#        )
#
# .. seealso::
#
#     The restults saved to disk and the output of get_clusters_table
#     do not contain the anatomical location of the clusters.
#     To get the names of the location of the clusters
#     according to one or several atlases,
#     we recommend using
#     the `atlasreader package <https://github.com/miykael/atlasreader>`_.
#
import pandas as pd

table_file = (
    output_dir
    / "derivatives"
    / "nilearn_glm"
    / "sub-10159"
    / (
        "sub-10159_task-stopsignal_space-MNI152NLin2009cAsym_"
        "contrast-stopsuccessMinusGo_clusters.tsv"
    )
)

table = pd.read_csv(table_file, sep="\t")

table

# %%
# We can get a latex table from a Pandas Dataframe
# for display and publication purposes.
#
# .. note::
#
#   This requires to have jinja2 installed:
#
#     .. code-block:: bash
#
#        pip install jinja2
#
print(table.to_latex())

# %%
# You can also print the output to markdown,
# if you have the `tabulate` dependencies installed.
#
# .. code-block:: bash
#
#    pip install tabulate
#
# .. code-block:: python
#
#    table.to_markdown()
#
