"""
The haxby dataset: different multi-class strategies
===================================================

In this example,
we compare one vs all and one vs one multi-class strategies:
the overall cross-validated accuracy and the confusion matrix.

"""

import numpy as np
import pandas as pd

from nilearn import datasets

# %%
# Load the Haxby data dataset
# ---------------------------

haxby_dataset = datasets.fetch_haxby()

func_filename = haxby_dataset.func[0]
mask_filename = haxby_dataset.mask

print(f"Mask nifti images are located at: {mask_filename}")
print(f"Functional nifti images are located at: {func_filename}")


# %%
# We load the behavioral data that we will predict and
# remove the rest condition, as it is of no interest to us.
labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")

y = labels["labels"]
run = labels["chunks"]
n_runs = len(np.unique(run))

non_rest = y != "rest"
y = y[non_rest]

# %%
# We get the labels of the numerical conditions represented by the vector y
# and we sort the conditions by the order of appearance.
unique_conditions, order = np.unique(y, return_index=True)
unique_conditions = unique_conditions[np.argsort(order)]

# %%
# Prepare the :term:`fMRI` data
# -----------------------------
# We extract the data with a NiftiMasker.
# For decoding, standardizing is often very important,
# so we set ``standardize="zscore_sample"``.

from nilearn.maskers import NiftiMasker

nifti_masker = NiftiMasker(
    mask_img=mask_filename,
    runs=run,
    smoothing_fwhm=4,
    memory="nilearn_cache",
    standardize="zscore_sample",
    memory_level=1,
    verbose=1,
)
X = nifti_masker.fit_transform(func_filename)

# %%
# Remove the "rest" condition
X = X[non_rest]
run = run[non_rest]

# %%
# Build the decoders, using scikit-learn
# --------------------------------------
# Nilearn does not have dedicate multiclass estimators,
# so we rely on those from sklearn.
# Here we use a Support Vector Classification, with a linear kernel,
# and a simple feature selection step.

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# %%
svc_ovo = OneVsOneClassifier(
    Pipeline(
        [
            ("anova", SelectKBest(f_classif, k=500)),
            ("svc", SVC(kernel="linear")),
        ]
    )
)

# %%
svc_ova = OneVsRestClassifier(
    Pipeline(
        [
            ("anova", SelectKBest(f_classif, k=500)),
            ("svc", SVC(kernel="linear")),
        ]
    )
)

# %%
# Now we compute cross-validation scores
# --------------------------------------
# The :term:`fMRI` data is acquired by runs,
# and the noise is autocorrelated in a given run.
# Hence, it is better to predict across runs when doing cross-validation.
# To leave a run out, pass the cross-validator object
# to the cv parameter of decoder.

from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

cv = LeaveOneGroupOut()

# %%
cv_scores_ovo = cross_val_score(svc_ovo, X, y, cv=cv, verbose=1, groups=run)

# %%
cv_scores_ova = cross_val_score(svc_ova, X, y, cv=cv, verbose=1, groups=run)

# %%
print("OvO:", cv_scores_ovo.mean().round(decimals=3))
print("OvA:", cv_scores_ova.mean().round(decimals=3))

# %%
# Plot barplots of the prediction scores
# --------------------------------------

from matplotlib import pyplot as plt

from nilearn.plotting import show

plt.figure(figsize=(4, 3))
plt.boxplot([cv_scores_ova, cv_scores_ovo])
plt.xticks([1, 2], ["One vs All", "One vs One"])
plt.title("Prediction: accuracy score")

show()

# %%
# Plot the confusion matrices
# ---------------------------
# We fit on the first 10 runs and plot a confusion matrix on the last 2 runs.

from sklearn.metrics import confusion_matrix

from nilearn.plotting import plot_matrix

svc_ovo.fit(X[run < 10], y[run < 10])
y_pred_ovo = svc_ovo.predict(X[run >= 10])

im = plot_matrix(
    confusion_matrix(y_pred_ovo, y[run >= 10]),
    labels=unique_conditions,
    title="Confusion matrix: One vs One",
    cmap="inferno",
    figure=(6, 5),
    auto_fit=False,
)
ax = im.axes
ax.set_ylabel("True label")
ax.set_xlabel("Predicted label")

svc_ova.fit(X[run < 10], y[run < 10])
y_pred_ova = svc_ova.predict(X[run >= 10])

im = plot_matrix(
    confusion_matrix(y_pred_ova, y[run >= 10]),
    labels=unique_conditions,
    title="Confusion matrix: One vs All",
    cmap="inferno",
    figure=(6, 5),
    auto_fit=False,
)
ax = im.axes
ax.set_ylabel("True label")
ax.set_xlabel("Predicted label")

show()

# sphinx_gallery_dummy_images=3
