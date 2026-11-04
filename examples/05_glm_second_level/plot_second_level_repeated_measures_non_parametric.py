"""
Non-parametric second-level inference for repeated measures
===========================================================

This example compares three group-level tests of a vertical-versus-horizontal
checkerboard contrast:

1. Permuting condition labels without accounting for repeated observations.
2. Permuting condition labels within each subject using
   ``exchangeability_blocks``.
3. Testing each subject's vertical-versus-horizontal contrast map with a
   one-sample sign-flipping test.

The block-restricted and direct-contrast analyses should give similar results.
The unrestricted analysis ignores the pairing between each subject's two maps.
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
# Set up the repeated-measures design. The rows are ordered by condition, so
# each subject label appears once in the horizontal maps and once in the
# vertical maps.
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
# Run the condition-level analysis without exchangeability blocks, then with
# blocks that keep the two observations from each subject together.
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
# As a reference, run the one-sample test on each subject's directly estimated
# vertical-versus-horizontal contrast. Each row now represents one subject, so
# the default sign-flipping test can be used without blocks.
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
# Display the family-wise-error-corrected negative log10 p-values. The blocked
# result and the direct-contrast reference should be similar; finite
# permutation counts can make their maps differ slightly.
import matplotlib.pyplot as plt

from nilearn.plotting import plot_glass_brain, show

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
results = [unrestricted_result, blocked_result, reference_result]
titles = [
    "Unrestricted row permutations",
    "Permutations within subjects",
    "Direct within-subject contrast",
]
for result, title, ax in zip(results, titles, axes, strict=True):
    plot_glass_brain(
        result,
        axes=ax,
        figure=fig,
        display_mode="z",
        threshold=-np.log10(0.05),
        vmin=-np.log10(0.05),
        vmax=3.0,
        title=title,
        colorbar=False,
    )
fig.suptitle("Non-parametric repeated-measures inference")
show()
