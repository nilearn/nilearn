"""
Setting a parameter by cross-validation
=======================================================
TODO: edit this
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

In scikit-learn, this can be done using the GridSearchCV object, that
will automatically select the best parameters of an estimator from a
grid of parameter values.

One difficulty here is that we are working with a composite estimator: a
pipeline of feature selection followed by SVC. Thus to give the name
of the parameter that we want to tune we need to give the name of the
step in the pipeline, followed by the name of the parameter, with '__' as
a separator.

"""

###########################################################################
# Load the Haxby dataset
# -----------------------
from nilearn import datasets
# by default 2nd subject data will be fetched on which we run our analysis
haxby_dataset = datasets.fetch_haxby()
fmri_filename = haxby_dataset.func[0]
mask_filename = haxby_dataset.mask

# print basic information on the dataset
print('Mask nifti image (3D) is located at: %s' % haxby_dataset.mask)
print('Functional nifti image (4D) are located at: %s' % haxby_dataset.func[0])

# Load the behavioral data
import pandas as pd
labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
y = labels['labels']


# Keep only data corresponding to shoes or bottles
from nilearn.image import index_img
condition_mask = y.isin(['shoe', 'bottle'])

fmri_niimgs = index_img(fmri_filename, condition_mask)
y = y[condition_mask]
session = labels['chunks'][condition_mask]

###########################################################################
# ANOVA pipeline with :class:`nilearn.decoding.Decoder` object
# ------------------------------------------------------------
#
# Nilearn Decoder object aims to provide smooth user experience by acting
# as a pipeline of several tasks: preprocessing with NiftiMasker, decoding
# with different types of estimators (in this example is Support Vector
# Machine with a linear kernel) on nested cross-validation, selecting
# relevant features with ANOVA -- a classical univariate feature selection
# based on F-test.
from nilearn.decoding import Decoder
# Here screening_percentile is set to 1.25 percent, meaning around 500
# features will be selected with ANOVA.
decoder = Decoder(estimator='svc', mask=mask_filename, smoothing_fwhm=4,
                  standardize=True, screening_percentile=1.25)

###########################################################################
# Fit the Decoder and predict the reponses
# -------------------------------------------------
decoder.fit(fmri_niimgs, y)
y_pred = decoder.predict(fmri_niimgs)

###########################################################################
# Compute prediction scores with different values of screening percentile
# -----------------------------------------------------------------------
import numpy as np
sp_range = [1.25, 2.5, 3.75, 7.5, 12.5, 25.0]
cv_scores = []
val_scores = []

for sp in sp_range:
    decoder = Decoder(estimator='svc', mask=mask_filename,
                      smoothing_fwhm=4, cv=3, standardize=True,
                      screening_percentile=sp)
    decoder.fit(index_img(fmri_niimgs, session < 10),
                y[session < 10])
    cv_scores.append(np.mean(decoder.cv_scores_['bottle']))
    print("Sreening Percentile: %.3f" % sp)
    print("Mean CV score: %.4f" % cv_scores[-1])

    y_pred = decoder.predict(index_img(fmri_niimgs, session == 10))
    val_scores.append(np.mean(y_pred == y[session == 10]))
    print("Validation score: %.4f" % val_scores[-1])

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
    
    for sp in sp_range:
        decoder = Decoder(estimator='svc', mask=mask_filename,
                          smoothing_fwhm=4, cv=3, standardize=True,
                          screening_percentile=sp)
        decoder.fit(index_img(fmri_niimgs, train), y_train)
        y_pred = decoder.predict(index_img(fmri_niimgs, test))
        val_scores.append(np.mean(y_pred == y_test))

    nested_cv_scores.append(np.max(val_scores))

print("Nested CV score: %.4f" % np.mean(nested_cv_scores))
    
###########################################################################
# Plot the prediction scores using matplotlib
# ---------------------------------------------
from matplotlib import pyplot as plt
from nilearn.plotting import show

plt.figure(figsize=(6, 4))
plt.plot(cv_scores, label='Cross validation scores')
plt.plot(val_scores, label='Left-out validation data scores')
plt.xticks(np.arange(len(sp_range)), sp_range)
plt.axis('tight')
plt.xlabel('ANOVA screening percentile')

plt.axhline(np.mean(nested_cv_scores),
            label='Nested cross-validation',
            color='r')

plt.legend(loc='best', frameon=False)
show()
