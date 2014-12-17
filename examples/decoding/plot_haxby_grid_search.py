"""
Setting a parameter by cross-validation
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

The proper appraoch is known as a "nested cross-validation". It consists
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

### Load Haxby dataset ########################################################
from nilearn import datasets
import numpy as np
haxby_dataset = datasets.fetch_haxby_simple()

y, session = np.loadtxt(haxby_dataset.session_target).astype("int").T
conditions = np.recfromtxt(haxby_dataset.conditions_target)['f0']

### Preprocess data ###########################################################

# Keep only data corresponding to shoes or bottles
condition_mask = np.logical_or(conditions == 'shoe', conditions == 'bottle')
y = y[condition_mask]
conditions = conditions[condition_mask]

### Loading step ##############################################################
from nilearn.input_data import NiftiMasker

mask_filename = haxby_dataset.mask
# For decoding, standardizing is often very important
nifti_masker = NiftiMasker(mask_img=mask_filename, sessions=session,
                           smoothing_fwhm=4, standardize=True,
                           memory="nilearn_cache", memory_level=1)
func_filename = haxby_dataset.func
X = nifti_masker.fit_transform(func_filename)
# Restrict to non rest data
X = X[condition_mask]
session = session[condition_mask]

### Prediction function #######################################################

### Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel
from sklearn.svm import SVC
svc = SVC(kernel='linear')

### Dimension reduction #######################################################

from sklearn.feature_selection import SelectKBest, f_classif

### Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. We set the number of features to be selected to 500
feature_selection = SelectKBest(f_classif, k=500)

# We have our classifier (SVC), our feature selection (SelectKBest), and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])

### Cross validation ##########################################################

anova_svc.fit(X, y)
y_pred = anova_svc.predict(X)

from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score
cv = LeaveOneLabelOut(session[session < 10])

k_range = [10, 15, 30, 50, 150, 300, 500, 1000, 1500, 3000, 5000]
cv_scores = []
scores_validation = []

for k in k_range:
    feature_selection.k = k
    cv_scores.append(np.mean(
        cross_val_score(anova_svc, X[session < 10], y[session < 10])))
    print "CV score", cv_scores[-1]

    anova_svc.fit(X[session < 10], y[session < 10])
    y_pred = anova_svc.predict(X[session == 10])
    scores_validation.append(np.mean(y_pred == y[session == 10]))
    print "score validation", scores_validation[-1]


from matplotlib import pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(cv_scores, label='Cross validation scores')
plt.plot(scores_validation, label='Left-out validation data scores')
plt.xticks(np.arange(len(k_range)), k_range)
plt.axis('tight')
plt.xlabel('k')

### Nested cross-validation ###################################################
from sklearn.grid_search import GridSearchCV
# We are going to tune the parameter 'k' of the step called 'anova' in
# the pipeline. Thus we need to address it as 'anova__k'.

# Note that GridSearchCV takes an n_jobs argument that can make it go
# much faster
grid = GridSearchCV(anova_svc, param_grid={'anova__k': k_range}, verbose=True)
nested_cv_scores = cross_val_score(grid, X, y)

plt.axhline(np.mean(nested_cv_scores),
            label='Nested cross-validation',
            color='r')

plt.legend(loc='best', frameon=False)
plt.show()
