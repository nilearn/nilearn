import numpy as np
import os
import os.path as op
import glob
import joblib
import pandas
import sys

import nibabel as nb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer

from nilearn.image import new_img_like, concat_imgs, index_img, smooth_img, mean_img
from nilearn.decoding import SearchLight
from nilearn.masking import intersect_masks

from sklearn.model_selection import LeaveOneGroupOut

"""
This file is a copy of /hpc/scalp/tchamina/HomoFaber/MVPY/mvpa_shifters_vs_fillers_sylvain_sourcecode/02_homofaber_searchlight_shifters_vs_fillers.py;
which has been modified to:
- change the directory structure of the outputs of the analysis
- add a more simple leave-one-run-out cross-validation scheme
Sylvain Takerkart 2017/09/19

The current version, as of 2018/06/14, is a copy of 
/hpc/scalp/tchamina/HomoFaber/MVPY/mvpa_shifters_vs_fillers_sylvain_sourcecode/03_homofaber_intersubject_searchlight.py
"""

def balanced_accuracy(y1,y2):
    return recall_score(y1, y2, pos_label=None, average='macro')

root_dir = '/hpc/scalp/tchamina/HomoFaber'
spm_modelname = 'Analysis_single_bayes_final_normalized'

subjects_list = ['homofaber_00', 'homofaber_01', 'homofaber_02', 'homofaber_03', 'homofaber_04',
                 'homofaber_05', 'homofaber_06', 'homofaber_07', 'homofaber_08', 'homofaber_09',
                 'homofaber_11', 'homofaber_12', 'homofaber_13', 'homofaber_14', 'homofaber_16']

mvpa_question = 'shifters_vs_fillers'

searchlight_radius = 6

n_jobs = 16

crossval_case = 'loso'

permutation_ind = 0


# defining all input directories
mvpa_subdir = 'mvpa_{}_analyses'.format(mvpa_question)
permutations_dir = op.join(root_dir, 
                           mvpa_subdir, 
                           'permutations', 
                           'within_sess_perms_correct_trials')
classif_metadata_dir = op.join(root_dir,
                               mvpa_subdir,
                               'mvpa_task_definition',
                               'within_sess_perms_correct_trials')
# defining all output directories
searchlight_res_dir = op.join(root_dir,
                              mvpa_subdir,
                              'intersubj_Cbeta_normalized_searchlight_res')
if not(op.exists(searchlight_res_dir)):
    os.makedirs(searchlight_res_dir)
    print('Creating new directory to save results: {}'.format(searchlight_res_dir))
single_split_res_dir = op.join(searchlight_res_dir,'single_split_maps')
if not(op.exists(single_split_res_dir)):
    os.makedirs(single_split_res_dir)
    print('Creating new directory to save single split searchlight maps: {}'.format(single_split_res_dir))


y = []
subj_vect = []
fmri_nii_list = []
subjmask_list = []
for subj_ind, subject in enumerate(subjects_list):
    classif_metadata_path = op.join(classif_metadata_dir,subject+'_classif_metadata.jl')
    [y_subj,session_xval,modality,beta_numbers] = joblib.load(classif_metadata_path)
    y.extend(y_subj)
    subj_vect.extend(subj_ind*np.ones(len(y_subj)))
        
    subject_dir = op.join(root_dir, subject)
    betas_dir = op.join(subject_dir, spm_modelname)

    for current_beta_ind in beta_numbers:
        beta_path = op.join(betas_dir, 'Cbeta_{:04d}.nii'.format(current_beta_ind))
        print(beta_path)
        beta_nii = nb.load(beta_path)
        #sbeta_nii = smooth_img(beta_nii, fwhm=fwhm)
        #sbeta_data = sbeta_nii.get_data()
        fmri_nii_list.append(beta_nii)

    subjmask_name = op.join(betas_dir,'mask.nii')
    subjmask_nii = nb.load(subjmask_name)
    subjmask_list.append(subjmask_nii)

subj_vect = np.array(subj_vect)

print("Intersecting the masks from all the subjects...")
mask_nii = intersect_masks(subjmask_list)
print("Concatenating the data from all the subjects...")
fmri_img = concat_imgs(fmri_nii_list)

loso = LeaveOneGroupOut()
n_splits = len(subjects_list)

y = np.array(y)
y_permuted = y
single_split_path_list = []
print("Launching cross-validation...")
for split_ind, (train_inds,test_inds) in enumerate(loso.split(subj_vect,subj_vect,subj_vect)):
    print("...split {:02d} of {:02d}".format(split_ind+1, n_splits))
    single_split = [(train_inds,test_inds)]
    y_train = y_permuted[train_inds]
    n_samples = len(y_train)
    class_labels = np.unique(y_train)
    n_classes = len(class_labels)
    weights_list = []
    for c in class_labels:
        weight = float(n_samples) / (n_classes * np.sum(y_train == c))
        weights_list.append(weight)
    class_weight = {class_labels[0]: weights_list[0], class_labels[1]: weights_list[1]}
    print('Class weights used for classifier estimation:', class_weight)
    # define our estimator (which uses the weights!)
    #weighted_clf = SVC(kernel="linear", class_weight=class_weight, C=0.001)
    weighted_clf = LogisticRegression(C=0.1, class_weight=class_weight)
    # now we can call the searchlight with all these options
    print("...preparing searchlight for this split")
    searchlight = SearchLight(mask_nii,
                              process_mask_img=mask_nii,
                              radius=searchlight_radius,
                              n_jobs=n_jobs,
                              verbose=1,
                              cv=single_split,
                              scoring=make_scorer(balanced_accuracy),
                              estimator=weighted_clf)
    print("...fitting searchlight for this split!")
    searchlight.fit(fmri_img, y_permuted)


    single_split_nii = new_img_like(mask_nii,searchlight.scores_)
    single_split_path = op.join(single_split_res_dir,'intersubj_balancedacc_rad{:05.2f}mm_{}_permut{:04d}_split{:1d}of{:1d}.nii.gz'.format(searchlight_radius,crossval_case,permutation_ind,split_ind+1,n_splits))
    print('Saving score map for {} and fold number {:02d}'.format(subject,split_ind+1))
    single_split_nii.to_filename(single_split_path)
    single_split_path_list.append(single_split_path)

mean_splits_nii = mean_img(single_split_path_list)
mean_splits_path = op.join(searchlight_res_dir,'intersubj_balancedacc_rad{:05.2f}mm_{}_permut{:04d}_mean.nii.gz'.format(searchlight_radius,crossval_case,permutation_ind))
print('Saving average score map: {}'.format(mean_splits_path))
mean_splits_nii.to_filename(mean_splits_path)




