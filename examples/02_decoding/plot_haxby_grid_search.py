"""Setting a parameter by cross-validation
=======================================================

Here we set the number of features selected in an Anova-SVC approach to
maximize the cross-validation score.

After separating 2 sessions for validation, we vary that parameter and
measure the cross-validation score. We also measure the prediction score
on the left-out validation data. As we can see, the two scores vary by a
significant amount: this is due to sampling noise in cross validation,
and choosing the parameter k to maximize the cross-validation score,
might not maximize the score on left-out data.

Thus using data to maximize a cross-validation score computed on that
same data is likely to optimistic and lead to an overfit.

The proper approach is known as a "nested cross-validation". It consists
in doing cross-validation loops to set the model parameters inside the
cross-validation loop used to judge the prediction performance: the
parameters are set separately on each fold, never using the data used to
measure performance.

For decoding task, in nilearn, this can be done using the
:class:`nilearn.decoding.Decoder` object, that will automatically select
the best parameters of an estimator from a grid of parameter values.

One difficulty is that the Decoder object is a composite estimator: a
pipeline of feature selection followed by Support Vector Machine. Tuning
the SVM's parameters is already done automatically inside the Decoder, but
performing cross-validation for the feature selection must be done
manually.

"""

###########################################################################
# Load the Haxby dataset
# -----------------------
from nilearn import datasets

# by default 2nd subject data will be fetched on which we run our analysis
haxby_dataset = datasets.fetch_haxby()
fmri_img = haxby_dataset.func[0]
mask_img = haxby_dataset.mask

# print basic information on the dataset
print(f"Mask nifti image (3D) is located at: {haxby_dataset.mask}")
print(f"Functional nifti image (4D) are located at: {haxby_dataset.func[0]}")

# Load the behavioral data
import pandas as pd

labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
y = labels["labels"]


# Keep only data corresponding to shoes or bottles
from nilearn.image import index_img

condition_mask = y.isin(["shoe", "bottle"])

fmri_niimgs = index_img(fmri_img, condition_mask)
y = y[condition_mask]
session = labels["chunks"][condition_mask]

###########################################################################
# ANOVA pipeline with :class:`nilearn.decoding.Decoder` object
# ------------------------------------------------------------
#
# Nilearn Decoder object aims to provide smooth user experience by acting as a
# pipeline of several tasks: preprocessing with NiftiMasker, reducing dimension
# by selecting only relevant features with ANOVA -- a classical univariate
# feature selection based on F-test, and then decoding with different types of
# estimators (in this example is Support Vector Machine with a linear kernel)
# on nested cross-validation.
from nilearn.decoding import Decoder

# Here screening_percentile is set to 2 percent, meaning around 800
# features will be selected with ANOVA.
decoder = Decoder(
    estimator="svc",
    cv=5,
    mask=mask_img,
    smoothing_fwhm=4,
    standardize=True,
    screening_percentile=2,
)

###########################################################################
# Fit the Decoder and predict the responses
# -------------------------------------------------
# As a complete pipeline by itself, decoder will perform cross-validation
# for the estimator, in this case Support Vector Machine. We can output the
# best parameters selected for each cross-validation fold. See
# https://scikit-learn.org/stable/modules/cross_validation.html for an
# excellent explanation of how cross-validation works.
#
# First we fit the Decoder
decoder.fit(fmri_niimgs, y)
for i, (param, cv_score) in enumerate(
    zip(decoder.cv_params_["shoe"]["C"], decoder.cv_scores_["shoe"])
):
    print(
        "Fold %d | Best SVM parameter: %.1f with score: %.3f"
        % (i + 1, param, cv_score)
    )
# Output the prediction with Decoder
y_pred = decoder.predict(fmri_niimgs)

###########################################################################
# Compute prediction scores with different values of screening percentile
# -----------------------------------------------------------------------
import numpy as np

screening_percentile_range = [2, 4, 8, 16, 32, 64]
cv_scores = []
val_scores = []

for sp in screening_percentile_range:
    decoder = Decoder(
        estimator="svc",
        mask=mask_img,
        smoothing_fwhm=4,
        cv=3,
        standardize=True,
        screening_percentile=sp,
    )
    decoder.fit(index_img(fmri_niimgs, session < 10), y[session < 10])
    cv_scores.append(np.mean(decoder.cv_scores_["bottle"]))
    print(f"Sreening Percentile: {sp:.3f}")
    print(f"Mean CV score: {cv_scores[-1]:.4f}")

    y_pred = decoder.predict(index_img(fmri_niimgs, session == 10))
    val_scores.append(np.mean(y_pred == y[session == 10]))
    print(f"Validation score: {val_scores[-1]:.4f}")

###########################################################################
# Nested cross-validation
# -----------------------
# We are going to tune the parameter 'screening_percentile' in the
# pipeline.
from sklearn.model_selection import KFold

cv = KFold(n_splits=3)
nested_cv_scores = []

for train, test in cv.split(session):
    y_train = np.array(y)[train]
    y_test = np.array(y)[test]
    val_scores = []

    for sp in screening_percentile_range:
        decoder = Decoder(
            estimator="svc",
            mask=mask_img,
            smoothing_fwhm=4,
            cv=3,
            standardize=True,
            screening_percentile=sp,
        )
        decoder.fit(index_img(fmri_niimgs, train), y_train)
        y_pred = decoder.predict(index_img(fmri_niimgs, test))
        val_scores.append(np.mean(y_pred == y_test))

    nested_cv_scores.append(np.max(val_scores))

print(f"Nested CV score: {np.mean(nested_cv_scores):.4f}")

###########################################################################
# Plot the prediction scores using matplotlib
# ---------------------------------------------
from matplotlib import pyplot as plt
from nilearn.plotting import show

plt.figure(figsize=(6, 4))
plt.plot(cv_scores, label="Cross validation scores")
plt.plot(val_scores, label="Left-out validation data scores")
plt.xticks(
    np.arange(len(screening_percentile_range)), screening_percentile_range
)
plt.axis("tight")
plt.xlabel("ANOVA screening percentile")

plt.axhline(
    np.mean(nested_cv_scores), label="Nested cross-validation", color="r"
)

plt.legend(loc="best", frameon=False)
show()
