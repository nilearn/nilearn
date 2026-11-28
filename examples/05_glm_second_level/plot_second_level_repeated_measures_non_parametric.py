"""
Non-parametric group-level inference for repeated measures
===========================================================

When doing a group analysis,
we can often choose between:

- computing contrast between conditions at the subect level
  and passing that contrast image at the group level
  where we end up with one image per subject,

- passing one image per condition from the subject-level
  to estimate a group-level model
  and perform the contrast between conditions at the group level.

But in the latter case, we need to take into account for the fact
that data coming from the same-subject will be correlated:
when doing parametric inference this is the classic case of
t-test versus paired t-test or ANOVA versus repeated-measure ANOVA.
When doing non-parametric inference based on permutation testing
this means that data coming
from the same subject must be permuted 'together'.
To do this we rely on defining ``exchangeability_blocks``
to define the rows of a design matrix that belong together.

So in this example we compare
three group-level tests of a vertical-versus-horizontal checkerboard contrast:

1. unrestricted:
   permuting condition labels without accounting for repeated observations.

2. block-restricted: permuting condition labels within each subject using
   ``exchangeability_blocks``.

3. direct-contrast:
   testing each subject's vertical-versus-horizontal contrast map
   with a one-sample sign-flipping test.

The block-restricted and direct-contrast analyses should give similar results.
The unrestricted analysis ignores the pairing between each subject's two maps
and should therefore have a higher false positive rate
as it assumes that each row comes from independent samples.

.. admonition:: dataset

    This example uses
    the :ref:`Brainomics Localizer dataset <brainomics_maps>`
"""

# %%
# Fetch checkerboard contrast maps for the same subjects. The dataset fetcher
# returns subjects in the same order for each requested contrast.
from nilearn.datasets import fetch_localizer_contrasts

n_subjects = 16
horizontal = fetch_localizer_contrasts(
    ["horizontal checkerboard"], n_subjects=n_subjects
)
vertical = fetch_localizer_contrasts(
    ["vertical checkerboard"], n_subjects=n_subjects
)
within_subject_contrast = fetch_localizer_contrasts(
    ["vertical vs horizontal checkerboard"], n_subjects=n_subjects
)

# %%
# Set up the repeated-measures design.
# The rows are ordered by condition,
# so each subject label appears once in the horizontal maps
# and once in the vertical maps.
import numpy as np
import pandas as pd

second_level_input = horizontal.cmaps + vertical.cmaps
condition_effect = np.hstack((np.zeros(n_subjects), np.ones(n_subjects)))
subject_effect = np.vstack((np.eye(n_subjects), np.eye(n_subjects)))
subject_labels = [f"subject_{i:02d}" for i in range(n_subjects)]
design_matrix = pd.DataFrame(
    np.column_stack((condition_effect, subject_effect)),
    columns=["vertical vs horizontal", *subject_labels],
)
exchangeability_blocks = np.tile(np.arange(n_subjects), 2)

# %%
# Run the condition-level analysis without exchangeability blocks,
# then with blocks that keep the two observations from each subject together.
from nilearn.glm.second_level import non_parametric_inference

common_kwargs = {
    "design_matrix": design_matrix,
    "second_level_contrast": "vertical vs horizontal",
    "model_intercept": False,
    "n_perm": 1000,
    "two_sided_test": True,
    "random_state": 0,
}

unrestricted_result = non_parametric_inference(
    second_level_input,
    **common_kwargs,
)
blocked_result = non_parametric_inference(
    second_level_input,
    exchangeability_blocks=exchangeability_blocks,
    **common_kwargs,
)

# %%
# As a reference, run the one-sample test
# on each subject's directly estimated vertical-versus-horizontal contrast.
# Each row now represents one subject,
# so the default sign-flipping test can be used without blocks.
contrast_design_matrix = pd.DataFrame({"intercept": np.ones(n_subjects)})
reference_result = non_parametric_inference(
    within_subject_contrast.cmaps,
    design_matrix=contrast_design_matrix,
    second_level_contrast="intercept",
    n_perm=1000,
    two_sided_test=True,
    random_state=0,
)

# %%
# Visualization
# -------------
# First let's have a look at the different design matrix
# and highlight the exchangeability_blocks:
# the colored column on the left shows which rows belong to the same block,
# with one color per subject.
import matplotlib.pyplot as plt

from nilearn.plotting import plot_design_matrix, show

fig, (ax_blocks, ax_design) = plt.subplots(
    1,
    2,
    figsize=(6, 6),
    sharey=True,
    gridspec_kw={"width_ratios": [1, 17]},
    layout="constrained",
)
ax_blocks.imshow(
    exchangeability_blocks[:, np.newaxis],
    cmap="tab20",
    aspect="auto",
    interpolation="nearest",
)
ax_blocks.xaxis.tick_top()
ax_blocks.set_xticks([0])
ax_blocks.set_xticklabels(["blocks"], rotation=60, ha="left")
plot_design_matrix(design_matrix, axes=ax_design)
ax_design.set_ylabel("")
ax_blocks.set_ylabel("scan number")

plot_design_matrix(contrast_design_matrix)

show()

# %%
# Then we can compare the results
# using :func:`~nilearn.plotting.plot_img_comparison`
# and :func:`~nilearn.plotting.plot_bland_altman`
# that allow to compare entire images
# without thresholding effects
# or having to select specific cut coordinates.
from nilearn.plotting import plot_bland_altman, plot_img_comparison

common_plotting_params = {
    "cmap": "plasma",
    "gridsize": 25,
}

# %%
# First we can check that doing non-parametric second-level inference
# using block-restricted and direct-contrast give the same results.
#
plot_bland_altman(
    reference_result,
    blocked_result,
    ref_label="direct-contrast",
    src_label="block-restricted",
    **common_plotting_params,
)
plot_img_comparison(
    reference_result,
    blocked_result,
    ref_label="direct-contrast",
    src_label="block-restricted",
    **common_plotting_params,
)

show()

# %%
# Then we can check that doing non-parametric second-level inference
# using unrestricted and direct-contrast give different results.
#
plot_bland_altman(
    reference_result,
    unrestricted_result,
    ref_label="direct-contrast",
    src_label="unrestricted",
    **common_plotting_params,
)
plot_img_comparison(
    reference_result,
    unrestricted_result,
    ref_label="direct-contrast",
    src_label="unrestricted",
    **common_plotting_params,
)

show()
