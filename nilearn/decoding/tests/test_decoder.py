from nose.tools import (assert_equal, assert_true, assert_raises,
                        assert_is_instance)
import warnings
import os
import numpy as np
import nibabel
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectPercentile

from nilearn.input_data import NiftiMasker
from nilearn.decoding.decoder import (Decoder, MNI152_BRAIN_VOLUME,
                                      _check_estimator,
                                      _check_masking,
                                      _check_feature_screening,
                                      _get_mask_volume)



mni152_brain_mask = (
    "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz")

# Data used in almost all tests
# from .test_same_api import to_niimgs

# def test_decoder():
#     # Create a toy dataset to run decoder on
#     # Initialize with 4x4x4 scans of random values on 30 frames
#     rand = np.random.RandomState(0)
#     frames = 30
#     data = rand.rand(5, 5, 5, frames)
#     mask = np.ones((5, 5, 5), np.bool)
#     mask_img = nibabel.Nifti1Image(mask.astype(np.int),
#                                    np.eye(4))

#     n_jobs = 1
#     # Create a condition array
#     cond = np.arange(frames, dtype=int) > frames // 2

#     cond = cond.astype(np.int)

#     # Create an activation pixel.
#     data[2, 2, 2, :] = 0
#     data[2, 2, 2][cond.astype(np.bool)] = 2
#     data_img = nibabel.Nifti1Image(data, np.eye(4))

#     nifti_masker = NiftiMasker(mask_img=mask_img).fit()

#     sl = Decoder(estimator='svc_l2', mask=nifti_masker, n_jobs=n_jobs)

#     # sl.fit(data_img, cond)

#     # assert_equal(np.where(sl.scores_ == 1)[0].size, 1)
#     # assert_equal(sl.scores_[2, 2, 2], 1.)

#     pass



# def test_check_cv():
#     pass


def test_check_masking():

    smoothing_fwhm = 4
    target_affine = np.eye(4)
    target_shape = (5, 5, 5)
    standardize = True
    mask_strategy = 'epi'
    memory = None
    memory_level = 1
    smoothing_fwhm_test = 8

    masker_test = NiftiMasker(smoothing_fwhm=smoothing_fwhm_test,
                              target_affine=target_affine,
                              target_shape=target_shape,
                              standardize=standardize,
                              mask_strategy=mask_strategy, memory=memory,
                              memory_level=memory_level)
    masks = [None, masker_test]
    for mask in masks:
        masker = _check_masking(mask, smoothing_fwhm=smoothing_fwhm,
                                target_affine=target_affine,
                                target_shape=target_shape,
                                standardize=standardize,
                                mask_strategy=mask_strategy, memory=memory,
                                memory_level=memory_level)
        if mask is None:
            assert_is_instance(masker, BaseEstimator)
        else:
            assert_true(masker.smoothing_fwhm == smoothing_fwhm_test)



# def test_check_param_grid():
#     (estimator, X, y, param_grid)
#     pass



# def test_check_estimator():

#     regressor = LinearSVR()
#     classifier = LinearSVC()

#     estimators = [regressor, classifier]
#     for estimator in estimators:
#         print check_estimator(estimator)

#     # (estimator, y, pos_label)
#     pass



# def test_scorer():

#     regressor =  LinearSVR()
#     classifier = LinearSVC()

#     estimators = [regressor, classfier]
#     scoring = ['']
#     pos_label = ['']

#     y_regression = []
#     y_classification = []

#     for i, estimator in  estimators:
#         estimator, scoring, pos_label, y
#     pass


def test_feature_screening():

    for is_classif in [True, False]:
        for screening_percentile in [100, None, 20, 101, -1]:

            if screening_percentile == 100 or screening_percentile is None:
                assert_equal(_check_feature_screening(
                    screening_percentile, MNI152_BRAIN_VOLUME, is_classif),
                    None)
            elif screening_percentile == 101 or screening_percentile == -1:
                assert_raises(ValueError, _check_feature_screening,
                              screening_percentile, MNI152_BRAIN_VOLUME,
                              is_classif)
            elif screening_percentile == 20:
                assert_is_instance(_check_feature_screening(
                    screening_percentile, MNI152_BRAIN_VOLUME, is_classif),
                    BaseEstimator)


def test_get_mask_volume():
    if os.path.isfile(mni152_brain_mask):
        assert_equal(MNI152_BRAIN_VOLUME, _get_mask_volume(nibabel.load(
            mni152_brain_mask)))
    else:
        warnings.warn("Couldn't find %s (for testing)" % (
            mni152_brain_mask))


if __name__ == '__main__':

    test_check_masking()
    # test_get_mask_volume()
    # test_feature_screening()
    # test_decoder()
    # test_check_estimator()
