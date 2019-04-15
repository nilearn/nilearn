"""
Test the parcellations tools module
"""
import numpy as np
import nibabel

from nose.tools import assert_true, assert_equal
from nilearn.regions.parcellations import (Parcellations,
                                           _check_parameters_transform)
from nilearn._utils.testing import assert_raises_regex


def test_errors_raised_in_check_parameters_fit():
    # Test whether an error is raised or not given
    # a false method type
    # valid_methods = ['kmeans', 'ward', 'complete', 'average']
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


def test_parcellations_fit_on_single_nifti_image():
    # Test return attributes for each method
    data = np.zeros((10, 11, 12, 5))
    data[9, 10, 2] = 1
    data[4, 9, 3] = 2
    fmri_img = nibabel.Nifti1Image(data, affine=np.eye(4))

    methods = ['kmeans', 'ward', 'complete', 'average']
    n_parcels = [5, 10, 15]
    for n_parcel, method in zip(n_parcels, methods):
        parcellator = Parcellations(method=method, n_parcels=n_parcel)
        parcellator.fit(fmri_img)
        # Test that object returns attribute labels_img_
        assert_true(parcellator.labels_img_ is not None)
        # Test object returns attribute masker_
        assert_true(parcellator.masker_ is not None)
        assert_true(parcellator.mask_img_ is not None)
        if method != 'kmeans':
            # Test that object returns attribute connectivity_
            # only for AgglomerativeClustering methods
            assert_true(parcellator.connectivity_ is not None)
            labels_img = parcellator.labels_img_
            assert_true(parcellator.labels_img_ is not None)
            # After inverse_transform, shape must match with original input
            # data
            assert_true(labels_img.shape, (data.shape[0],
                                           data.shape[1],
                                           data.shape[2]))


def test_parcellations_fit_on_multi_nifti_images():
    data = np.zeros((10, 11, 12, 5))
    data[9, 10, 2] = 1
    data[4, 9, 3] = 2
    fmri_img = nibabel.Nifti1Image(data, affine=np.eye(4))
    # List of fmri images
    fmri_imgs = [fmri_img, fmri_img, fmri_img]

    parcellator = Parcellations(method='kmeans', n_parcels=5)
    parcellator.fit(fmri_imgs)
    assert_true(parcellator.labels_img_ is not None)

    parcellator = Parcellations(method='ward', n_parcels=5)
    parcellator.fit(fmri_imgs)
    assert_true(parcellator.labels_img_ is not None)

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

    for method in ['kmeans', 'ward', 'complete', 'average']:
        parcellator = Parcellations(method=method, n_parcels=parcels)
        parcellator.fit(fmri_img)
        # transform to signals
        signals = parcellator.transform(fmri_img)
        # Test if the signals extracted are of same shape as inputs
        # Here, we simply return numpy array for single subject input
        assert_equal(signals.shape, (fmri_img.shape[3], parcels))

        # Test for single subject but in a list.
        signals = parcellator.transform([fmri_img])
        assert_equal(signals.shape, (fmri_img.shape[3], parcels))


def test_parcellations_transform_multi_nifti_images():
    data = np.ones((10, 11, 12, 10))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3
    parcels = 5

    fmri_img = nibabel.Nifti1Image(data, affine=np.eye(4))
    fmri_imgs = [fmri_img, fmri_img, fmri_img]

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
    imgs, confounds, single_subject = _check_parameters_transform(fmri_img,
                                                                  confounds)
    assert_true(isinstance(imgs, (list, tuple)))
    assert_true(isinstance(confounds, (list, tuple)))
    assert_true(single_subject, True)

    # multi images
    fmri_imgs = [fmri_img, fmri_img, fmri_img]
    confounds_list = [confounds, confounds, confounds]
    imgs, confounds, _ = _check_parameters_transform(fmri_imgs, confounds_list)
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

    for method in ['kmeans', 'ward', 'complete', 'average']:
        parcellator = Parcellations(method=method, n_parcels=5)
        parcellator.fit(fmri_imgs)

        signals = parcellator.transform(fmri_imgs,
                                        confounds=confounds_list)
        assert_true(isinstance(signals, list))
        # n_parcels=5, length of data=10
        assert_equal(signals[0].shape, (10, 5))


def test_fit_transform():
    rng = np.random.RandomState(0)
    data = np.ones((10, 11, 12, 10))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3

    fmri_img = nibabel.Nifti1Image(data, affine=np.eye(4))
    fmri_imgs = [fmri_img, fmri_img, fmri_img]

    confounds = rng.randn(*(10, 3))
    confounds_list = [confounds, confounds, confounds]

    for method in ['kmeans', 'ward', 'complete', 'average']:
        parcellator = Parcellations(method=method, n_parcels=5)
        signals = parcellator.fit_transform(fmri_imgs)
        assert_true(parcellator.labels_img_ is not None)
        if method != 'kmeans':
            assert_true(parcellator.connectivity_ is not None)
        assert_true(parcellator.masker_ is not None)
        # fit_transform with confounds
        signals = parcellator.fit_transform(fmri_imgs,
                                            confounds=confounds_list)
        assert_true(isinstance(signals, list))
        assert_equal(signals[0].shape, (10, 5))


def test_inverse_transform_single_nifti_image():
    data = np.ones((10, 11, 12, 10))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3

    fmri_img = nibabel.Nifti1Image(data, affine=np.eye(4))
    methods = ['kmeans', 'ward', 'complete', 'average']

    for method in methods:
        parcellate = Parcellations(method=method, n_parcels=5)
        # Fit
        parcellate.fit(fmri_img)
        assert_true(parcellate.labels_img_ is not None)
        # Transform
        fmri_reduced = parcellate.transform(fmri_img)
        assert_true(isinstance(fmri_reduced, np.ndarray))
        # Shape matching with (scans, regions)
        assert_true(fmri_reduced.shape, (10, 5))
        # Inverse transform
        fmri_compressed = parcellate.inverse_transform(fmri_reduced)
        # A single Nifti image for single subject input
        assert_true(isinstance(fmri_compressed, nibabel.Nifti1Image))
        # returns shape of fmri_img
        assert_true(fmri_compressed.shape, (10, 11, 12, 10))

        # fmri_reduced in a list
        fmri_compressed = parcellate.inverse_transform([fmri_reduced])
        # A single Nifti image for single subject input
        assert_true(isinstance(fmri_compressed, nibabel.Nifti1Image))
        # returns shape of fmri_img
        assert_true(fmri_compressed.shape, (10, 11, 12, 10))


def test_transform_3d_input_images():
    # test list of 3D images
    data = np.ones((10, 11, 12))
    data[6, 7, 8] = 2
    data[9, 10, 11] = 3
    img = nibabel.Nifti1Image(data, affine=np.eye(4))
    # list of 3
    imgs = [img, img, img]
    parcellate = Parcellations(method='ward', n_parcels=20)
    X = parcellate.fit_transform(imgs)
    assert_true(isinstance(X, list))
    # (number of samples, number of features)
    assert_equal(np.concatenate(X).shape, (3, 20))
    # inverse transform
    imgs_ = parcellate.inverse_transform(X)
    assert_true(isinstance(imgs_, list))
    # test single 3D image
    X = parcellate.fit_transform(imgs[0])
    assert_true(isinstance(X, np.ndarray))
    assert_equal(X.shape, (1, 20))
