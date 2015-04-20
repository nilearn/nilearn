"""
The haxby dataset: different multi-class strategies
=======================================================

We compare one vs all and one vs one multi-class strategies: the overall
cross-validated accuracy and the confusion matrix.

"""
# Import matplotlib for plotting
from matplotlib import pyplot as plt

### Load Haxby dataset ########################################################
from nilearn import datasets
import numpy as np
haxby_dataset = datasets.fetch_haxby_simple()

# print basic information on the dataset
print('Mask nifti images are located at: %s' % haxby_dataset.mask)
print('Functional nifti images are located at: %s' % haxby_dataset.func)

func_filename = haxby_dataset.func
mask_filename = haxby_dataset.mask

y, session = np.loadtxt(haxby_dataset.session_target).astype('int').T
conditions = np.recfromtxt(haxby_dataset.conditions_target)['f0']

# Remove the rest condition, it is not very interesting
non_rest = conditions != b'rest'
conditions = conditions[non_rest]
y = y[non_rest]

# Get the labels of the numerical conditions represented by the vector y
unique_conditions, order = np.unique(conditions, return_index=True)
# Sort the conditions by the order of appearance
unique_conditions = unique_conditions[np.argsort(order)]

### Loading step ##############################################################
from nilearn.input_data import NiftiMasker
# For decoding, standardizing is often very important
nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                           sessions=session, smoothing_fwhm=4,
                           memory="nilearn_cache", memory_level=1)
X = nifti_masker.fit_transform(func_filename)
X = X[non_rest]
session = session[non_rest]

### Predictor #################################################################

### Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline

svc_ovo = OneVsOneClassifier(Pipeline([
                ('anova', SelectKBest(f_classif, k=500)),
                ('svc', SVC(kernel='linear'))
                ]))

svc_ova = OneVsRestClassifier(Pipeline([
                ('anova', SelectKBest(f_classif, k=500)),
                ('svc', SVC(kernel='linear'))
                ]))

### Cross-validation scores ###################################################
from sklearn.cross_validation import cross_val_score

cv_scores_ovo = cross_val_score(svc_ovo, X, y, cv=5, verbose=1)

cv_scores_ova = cross_val_score(svc_ova, X, y, cv=5, verbose=1)

print(79 * "_")
print('OvO', cv_scores_ovo.mean())
print('OvA', cv_scores_ova.mean())

plt.figure(figsize=(4, 3))
plt.boxplot([cv_scores_ova, cv_scores_ovo])
plt.xticks([1, 2], ['One vs All', 'One vs One'])
plt.title('Prediction: accuracy score')

### Plot a confusion matrix ###################################################
# Fit on the the first 10 sessions and plot a confusion matrix on the
# last 2 sessions
from sklearn.metrics import confusion_matrix

svc_ovo.fit(X[session < 10], y[session < 10])
y_pred_ovo = svc_ovo.predict(X[session >= 10])

plt.matshow(confusion_matrix(y_pred_ovo, y[session >= 10]))
plt.title('Confusion matrix: One vs One')
plt.xticks(np.arange(len(unique_conditions)), unique_conditions)
plt.yticks(np.arange(len(unique_conditions)), unique_conditions)

svc_ova.fit(X[session < 10], y[session < 10])
y_pred_ova = svc_ova.predict(X[session >= 10])

plt.matshow(confusion_matrix(y_pred_ova, y[session >= 10]))
plt.title('Confusion matrix: One vs All')
plt.xticks(np.arange(len(unique_conditions)), unique_conditions)
plt.yticks(np.arange(len(unique_conditions)), unique_conditions)

plt.show()
