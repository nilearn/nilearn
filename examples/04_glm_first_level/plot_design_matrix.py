"""
Examples of design matrices
===========================

Three examples of design matrices specification and computation for first-level
:term:`fMRI` data analysis (event-related design, block design,
:term:`FIR` design).

This examples requires matplotlib.

"""

# %%
# Define parameters
# -----------------
# At first, we define parameters related to the images acquisition.
import numpy as np

from nilearn.plotting import plot_design_matrix

tr = 1.0  # repetition time is 1 second
n_scans = 128  # the acquisition comprises 128 scans
frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times

# %%
# Then we define parameters related to the experimental design.

# these are the types of the different trials
conditions = ["c0", "c0", "c0", "c1", "c1", "c1", "c3", "c3", "c3"]
duration = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# these are the corresponding onset times
onsets = [30.0, 70.0, 100.0, 10.0, 30.0, 90.0, 30.0, 40.0, 60.0]
# Next, we simulate 6 motion parameters jointly observed with fMRI acquisitions
motion = np.cumsum(np.random.randn(n_scans, 6), 0)
# The 6 parameters correspond to three translations and three
# rotations describing rigid body motion
add_reg_names = ["tx", "ty", "tz", "rx", "ry", "rz"]

# %%
# Create design matrices
# ----------------------
# The same parameters allow us to obtain a variety of design matrices.
# We first create an events object.
import pandas as pd

events = pd.DataFrame(
    {"trial_type": conditions, "onset": onsets, "duration": duration}
)

# %%
# We sample the events into a design matrix, also including additional
# regressors.
from nilearn.glm.first_level import make_first_level_design_matrix

hrf_model = "glover"
X1 = make_first_level_design_matrix(
    frame_times,
    events,
    drift_model="polynomial",
    drift_order=3,
    add_regs=motion,
    add_reg_names=add_reg_names,
    hrf_model=hrf_model,
)

# %%
# Now we compute a block design matrix. We add duration to create the blocks.
# For this we first define an event structure that includes the duration
# parameter.

duration = 7.0 * np.ones(len(conditions))
events = pd.DataFrame(
    {"trial_type": conditions, "onset": onsets, "duration": duration}
)

# %%
# Then we sample the design matrix.

X2 = make_first_level_design_matrix(
    frame_times,
    events,
    drift_model="polynomial",
    drift_order=3,
    hrf_model=hrf_model,
)

# %%
# Finally we compute a :term:`FIR` model

events = pd.DataFrame(
    {"trial_type": conditions, "onset": onsets, "duration": duration}
)
hrf_model = "FIR"
X3 = make_first_level_design_matrix(
    frame_times,
    events,
    hrf_model="fir",
    drift_model="polynomial",
    drift_order=3,
    fir_delays=np.arange(1, 6),
)

# %%
# Here are the three designs side by side.
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 6), nrows=1, ncols=3)
plot_design_matrix(X1, ax=ax1)
ax1.set_title("Event-related design matrix", fontsize=12)
plot_design_matrix(X2, ax=ax2)
ax2.set_title("Block design matrix", fontsize=12)
plot_design_matrix(X3, ax=ax3)
ax3.set_title("FIR design matrix", fontsize=12)
plt.show()


# %%
# Correlation between regressors
# ------------------------------
from matplotlib.colorbar import make_axes


def plot_corr_matrix_mpl(
    des_mat, partial="upper", cmap="bwr", ax=None, fig=None
):

    if not isinstance(des_mat, pd.DataFrame):
        raise TypeError(
            f"'des_mat' must be a andas dataframe instance.\n"
            f"Got: {type(des_mat)}"
        )

    ALLOWED_CMAP = ["RdBu_r", "bwr", "seismic_r"]
    if cmap not in ALLOWED_CMAP:
        raise ValueError(f"cmap must be one of {ALLOWED_CMAP}")

    ALLOWED_PARTIALS = ["upper", "lower", None]
    if partial not in ALLOWED_PARTIALS:
        raise ValueError(f"partial must be one of {ALLOWED_PARTIALS}")

    columns_to_drop = ["intercept", "constant"]
    columns_to_drop.extend(
        col for col in des_mat.columns if col.startswith("drift_")
    )
    des_mat = des_mat.drop(columns=columns_to_drop, errors="ignore")

    mat = des_mat.corr()

    # For a heatmap mask, 0 = show, 1 = hide
    if partial:
        mask = np.ones_like(mat, dtype=bool)
        if partial == "upper":
            mask[np.triu_indices_from(mask)] = False
        elif partial == "lower":
            mask[np.tril_indices_from(mask)] = False
        mat[mask] = 0

    # find the second-largest value in each row
    # to omit values on the diagonal that will always be == 1
    second_largest = np.partition(mat.to_numpy(), -2, axis=1)[:, -2]
    vmax = max(abs(mat.min().min()), max(second_largest))

    if ax is None:
        if fig is None:
            fig = plt.figure()
        ax = fig.add_axes(111)
    elif fig is None:
        fig = ax.get_figure()

    im = ax.imshow(mat, cmap=cmap, vmax=vmax, vmin=vmax * -1)

    cax, _ = make_axes(
        ax, location="right", fraction=0.18, shrink=0.75, pad=0.02, aspect=15.0
    )
    fig.colorbar(im, cax=cax, spacing="proportional", orientation="vertical")

    col_labels = des_mat.columns
    ax.set_xticks(
        np.arange(mat.shape[1]), labels=col_labels, rotation=60, ha="left"
    )
    ax.set_yticks(np.arange(mat.shape[0]), labels=col_labels)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return ax


fig2, (ax1, ax2, ax3) = plt.subplots(figsize=(100, 6), nrows=1, ncols=3)
plot_corr_matrix_mpl(X1, ax=ax1)
ax1.set_title("Event-related correlation matrix", fontsize=12)
plot_corr_matrix_mpl(X2, ax=ax2, partial="lower")
ax2.set_title("Block correlation matrix", fontsize=12)
plot_corr_matrix_mpl(X3, ax=ax3, partial=None)
ax3.set_title("FIR correlation matrix", fontsize=12)
plt.show()


# %%
# Parametric modulation
# ---------------------
# By default, the fMRI GLM will expect that all events
# for a given condition have a BOLD
# response with the same amplitude.
# Sometimes, we may have specific expectations
# about how strong the BOLD response
# will be on a given event.
# This can be incorporated into the model by using **parametric modulation**,
# wherein each event has a predicted amplitude.
# This can be used both to improve model fit and to test hypotheses regarding
# how the BOLD response scales with important features of events,
# such as trial intensity or response time.
#
# Here we will assume that when a trial
# is the same condition as the previous one,
# it will elicit a less intense response.

conditions = ["c0", "c0", "c0", "c1", "c1", "c1", "c3", "c3", "c3"]
modulation = [1.0, 0.5, 0.25, 1.0, 0.5, 0.25, 1.0, 0.5, 0.25]
modulated_events = pd.DataFrame(
    {
        "trial_type": conditions,
        "onset": onsets,
        "duration": duration,
        "modulation": modulation,
    }
)

hrf_model = "glover"
X4 = make_first_level_design_matrix(
    frame_times,
    modulated_events,
    drift_model="polynomial",
    drift_order=3,
    hrf_model=hrf_model,
)

# Let's compare it to the unmodulated block design
fig, (ax1, ax2) = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
plot_design_matrix(X2, ax=ax1)
ax1.set_title("Block design matrix", fontsize=12)
plot_design_matrix(X4, ax=ax2)
ax2.set_title("Modulated block design matrix", fontsize=12)
plt.show()
