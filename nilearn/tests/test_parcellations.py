"""
Test the parcellations tools module
"""
import numpy as np
import nibabel

import sklearn
from distutils.version import LooseVersion
from nose.tools import assert_true, assert_equal
from nilearn.parcellations import (Parcellations,
                                   _check_parameters_transform)
from nilearn._utils.testing import assert_raises_regex


def test_errors_raised_in_check_parameters_fit():
    # Test whether an error is raised or not given
    # a false method type
    valid_methods = ['kmeans', 'ward', 'complete', 'average']
    data = np.zeros((6, 7, 8, 5))

    img = nibabel.Nifti1Image(data, affine=np.eye(4))

    method_raise1 = Parcellations(method=None)
    assert_raises_regex(ValueError,
                        "Parcellation method is specified as None. ",
                        method_raise1.fit, img)

    for invalid_method in ['kmens', 'avg', 'complte']:
        method_raise2 = Parcellations(method=invalid_method)
        msg = ("The method you have selected is not implemented "
               "'{0}'".format(invalid_method))
        assert_raises_regex(ValueError, msg, method_raise2.fit, img)

    if LooseVersion(sklearn.__version__) < LooseVersion('0.15'):
        for method in ['average', 'complete']:
            parcellator = Parcellations(method=method)
            np.testing.assert_raises(NotImplementedError, parcellator.fit, img)


def test_parcellations_fit_on_single_nifti_image():
    # Test return attributes for each method
    data = np.zeros((10, 11, 12, 5))
    data[9, 10, 2] = 1
    data[4, 9, 3] = 2
    fmri_img = nibabel.Nifti1Image(data, affine=np.eye(4))

    if LooseVersion(sklearn.__version__) >= LooseVersion('0.15'):
        methods = ['kmeans', 'ward', 'complete', 'average']
        n_parcels = [5, 10, 15]
        for n_parcel, method in zip(n_parcels, methods):
            parcellator = Parcellations(method=method, n_parcels=n_parcel)
            parcellator.fit(fmri_img)
            # Test that object returns attribute labels_
            assert_true(parcellator.labels_ is not None)
            # Test object returns attribute masker_
            assert_true(parcellator.masker_ is not None)
            assert_true(parcellator.mask_img_ is not None)
            if method != 'kmeans':
                # Test that object returns attribute connectivity_
                # only for AgglomerativeClustering methods
                assert_true(parcellator.connectivity_ is not None)
            masker = parcellator.masker_
            labels_img = masker.inverse_transform(parcellator.labels_)
            # After inverse_transform, shape must match with original input
            # data
            assert_true(labels_img.shape, (data.shape[0],
                                           data.shape[1],
                                           data.shape[2]))
    else:
        parcellator2 = Parcellations(method='ward', n_parcels=10)
        parcellator2.fit(fmri_img)
        assert_true(parcellator2.labels_ is not None)


def test_parcellations_fit_on_multi_nifti_images():
    data = np.zeros((10, 11, 12, 5))
    data[9, 10, 2] = 1
    data[4, 9, 3] = 2
    fmri_img = nibabel.Nifti1Image(data, affine=np.eye(4))
    # List of fmri images
    fmri_imgs = [fmri_img, fmri_img, fmri_img]

    parcellator = Parcellations(method='kmeans', n_parcels=5)
    parcellator.fit(fmri_imgs)
    assert_true(parcellator.labels_ is not None)

    parcellator = Parcellations(method='ward', n_parcels=5)
    parcellator.fit(fmri_imgs)
    assert_true(parcellator.labels_ is not None)

    # Smoke test with explicit mask image
    mask_img = np.ones((10, 11, 12))
    mask_img = nibabel.Nifti1Image(mask_img, np.eye(4))

    parcellator = Parcellations(method='kmeans', n_parcels=5,
                                mask=mask_img)
    parcellator.fit(fmri_imgs)

    parcellator = Parcellations(method='ward', n_parcels=5,
                                mask=mask_img)
    parcellator.fit(fmri_imgs)


def test_parcellations_transform_single_nifti_image():
    # Test with NiftiLabelsMasker extraction of timeseries data
    # after building a parcellations image

    # Here, data has ones. zeros will be considered as background labels
    # not foreground labels
    data = np.ones((10, 11, 12, 8))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3
    parcels = 5

    fmri_img = nibabel.Nifti1Image(data, affine=np.eye(4))

    if LooseVersion(sklearn.__version__) >= LooseVersion('0.15'):
        for method in ['kmeans', 'ward', 'complete', 'average']:
            parcellator = Parcellations(method=method, n_parcels=parcels)
            parcellator.fit(fmri_img)
            # transform to signals
            signals = parcellator.transform(fmri_img)
            # Test if the signals extracted are of same shape as inputs
            # Here, we take index 0 since we return list even for single
            # subject
            assert_equal(signals[0].shape, (fmri_img.shape[3], parcels))
    else:
        parcellator2 = Parcellations(method='ward', n_parcels=parcels)
        parcellator2.fit(fmri_img)
        signals2 = parcellator.transform(fmri_img)
        assert_equal(signals[0].shape, (fmri_img.shape[3], parcels))


def test_parcellations_transform_multi_nifti_images():
    data = np.ones((10, 11, 12, 10))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3
    parcels = 5

    fmri_img = nibabel.Nifti1Image(data, affine=np.eye(4))
    fmri_imgs = [fmri_img, fmri_img, fmri_img]

    if LooseVersion(sklearn.__version__) >= LooseVersion('0.15'):
        for method in ['kmeans', 'ward', 'complete', 'average']:
            parcellator = Parcellations(method=method, n_parcels=parcels)
            parcellator.fit(fmri_imgs)
            # transform multi images to signals. In return, we have length
            # equal to the number of images
            signals = parcellator.transform(fmri_imgs)
            assert_equal(signals[0].shape, (fmri_img.shape[3], parcels))
            assert_equal(signals[1].shape, (fmri_img.shape[3], parcels))
            assert_equal(signals[2].shape, (fmri_img.shape[3], parcels))

            assert_equal(len(signals), len(fmri_imgs))


def test_check_parameters_transform():
    rng = np.random.RandomState(0)
    data = np.ones((10, 11, 12, 10))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3

    # single image
    fmri_img = nibabel.Nifti1Image(data, affine=np.eye(4))
    # single confound
    confounds = rng.randn(*(10, 3))
    # Tests to check whether imgs, confounds returned are
    # list or not. Pre-check in parameters to work for list
    # of multi images and multi confounds
    imgs, confounds = _check_parameters_transform(fmri_img, confounds)
    assert_true(isinstance(imgs, (list, tuple)))
    assert_true(isinstance(confounds, (list, tuple)))

    # multi images
    fmri_imgs = [fmri_img, fmri_img, fmri_img]
    confounds_list = [confounds, confounds, confounds]
    imgs, confounds = _check_parameters_transform(fmri_imgs, confounds_list)
    assert_equal(imgs, fmri_imgs)
    assert_equal(confounds_list, confounds)

    # Test the error when length of images and confounds are not same
    msg = ("Number of confounds given does not match with the "
           "given number of images")
    not_match_confounds_list = [confounds, confounds]
    assert_raises_regex(ValueError, msg, _check_parameters_transform,
                        fmri_imgs, not_match_confounds_list)


def test_parcellations_transform_with_multi_confounds_multi_images():
    rng = np.random.RandomState(0)
    data = np.ones((10, 11, 12, 10))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3

    fmri_img = nibabel.Nifti1Image(data, affine=np.eye(4))
    fmri_imgs = [fmri_img, fmri_img, fmri_img]

    confounds = rng.randn(*(10, 3))
    confounds_list = (confounds, confounds, confounds)

    if LooseVersion(sklearn.__version__) >= LooseVersion('0.15'):
        for method in ['kmeans', 'ward', 'complete', 'average']:
            parcellator = Parcellations(method=method, n_parcels=5)
            parcellator.fit(fmri_imgs)

            signals = parcellator.transform(fmri_imgs,
                                            confounds=confounds_list)


def test_fit_transform():
    rng = np.random.RandomState(0)
    data = np.ones((10, 11, 12, 10))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3

    fmri_img = nibabel.Nifti1Image(data, affine=np.eye(4))
    fmri_imgs = [fmri_img, fmri_img, fmri_img]

    confounds = rng.randn(*(10, 3))
    confounds_list = [confounds, confounds, confounds]

    if LooseVersion(sklearn.__version__) >= LooseVersion('0.15'):
        for method in ['kmeans', 'ward', 'complete', 'average']:
            parcellator = Parcellations(method=method, n_parcels=5)
            signals = parcellator.fit_transform(fmri_imgs)
            assert_true(parcellator.labels_ is not None)
            if method != 'kmeans':
                assert_true(parcellator.connectivity_ is not None)
            assert_true(parcellator.masker_ is not None)
            # fit_transform with confounds
            signals = parcellator.fit_transform(fmri_imgs,
                                                confounds=confounds_list)
