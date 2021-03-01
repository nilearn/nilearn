import pytest
from nibabel import Nifti1Image
from collections import Counter
import numpy as np
from nilearn import input_data
from nilearn._utils import data_gen
from nilearn.image import get_data
from numpy.testing import assert_almost_equal

# Note: html output by nilearn view_* functions
# should validate as html5 using https://validator.w3.org/nu/ with no
# warnings


def _check_html(html_view):
    """ Check the presence of some expected code in the html viewer
    """
    assert "Parameters" in str(html_view)
    assert "data:image/svg+xml;base64," in str(html_view)
    assert html_view._repr_html_() == html_view.body


def test_3d_reports():
    # Dummy 3D data
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    data_img_3d = Nifti1Image(data, np.eye(4))

    # Dummy mask
    mask = np.zeros((9, 9, 9))
    mask[4:-4, 4:-4, 4:-4] = True
    mask_img_3d = Nifti1Image(data, np.eye(4))

    # test .fit method
    mask = input_data.NiftiMasker()
    mask.fit(data_img_3d)
    html = mask.generate_report()
    _check_html(html)

    # check providing mask to init
    masker = input_data.NiftiMasker(mask_img=mask_img_3d)
    masker.fit(data_img_3d)
    assert masker._report_content['warning_message'] is None
    html = masker.generate_report()
    _check_html(html)

    # check providing mask to init and no images to .fit
    masker = input_data.NiftiMasker(mask_img=mask_img_3d)
    assert masker._report_content['warning_message'] is None
    masker.fit()
    warn_message = ("No image provided to fit in NiftiMasker. "
                    "Setting image to mask for reporting.")
    with pytest.warns(UserWarning, match=warn_message):
        html = masker.generate_report()
    assert masker._report_content['warning_message'] == warn_message
    _check_html(html)


def test_nifti_labels_masker_report():
    shape = (13, 11, 12)
    affine = np.diag([2, 2, 2, 1])
    n_regions = 9
    labels = ['background'] + ['region_{}'.format(i) for i in range(1, n_regions+1)]
    length = 3
    EXPECTED_COLUMNS = ['label value',
                        'region name',
                        'size (in mm^3)',
                        'relative size (in %)']
    labels_img = data_gen.generate_labeled_regions(shape,
                                                   affine=affine,
                                                   n_regions=n_regions)
    # Check that providing incorrect labels raises an error
    masker = input_data.NiftiLabelsMasker(labels_img,
                                          labels = labels[:-1])
    masker.fit()
    with pytest.raises(ValueError,
                      match="Mismatch between the number of provided labels"):
        report = masker.generate_report()
    masker = input_data.NiftiLabelsMasker(labels_img,
                                          labels=labels)
    masker.fit()
    # Check that a warning is given when generating the report
    # since no image was provided to fit
    with pytest.warns(UserWarning,
                      match="No image provided to fit in NiftiLabelsMasker"):
        report = masker.generate_report()
    # Check that background label was left as default
    assert masker.background_label == 0
    assert masker._report_content['description'] == (
        'This reports shows the regions defined by the labels of the mask.')
    # Check that the number of regions is correct
    assert masker._report_content['number_of_regions'] == n_regions
    # Check that all expected columns are present with the right size
    for col in EXPECTED_COLUMNS:
        assert col in masker._report_content['summary']
        assert len(masker._report_content['summary'][col]) == n_regions
    # Check that labels match
    assert masker._report_content['summary']['region name'] == labels[1:]
    # Relative sizes of regions should sum to 100%
    assert_almost_equal(sum(masker._report_content['summary']['relative size (in %)']), 100)
    _check_html(report)
    assert "Regions summary" in str(report)
    # Check region sizes calculations
    expected_region_sizes = Counter(get_data(labels_img).ravel())
    for r in range(1, n_regions+1):
        assert_almost_equal(masker._report_content['summary']['size (in mm^3)'][r-1],
               expected_region_sizes[r] *
               np.abs(np.linalg.det(affine[:3, :3])))

    # Check that region labels are no displayed in the report
    # when they were not provided by the user.
    masker = input_data.NiftiLabelsMasker(labels_img)
    masker.fit()
    report = masker.generate_report()
    for col in EXPECTED_COLUMNS:
        if col == "region name":
            assert col not in masker._report_content["summary"]
        else:
            assert col in masker._report_content["summary"]
            assert len(masker._report_content['summary'][col]) == n_regions


def test_4d_reports():
    # Dummy 4D data
    data = np.zeros((10, 10, 10, 3), dtype=int)
    data[..., 0] = 1
    data[..., 1] = 2
    data[..., 2] = 3
    data_img_4d = Nifti1Image(data, np.eye(4))

    # Dummy mask
    mask = np.zeros((10, 10, 10), dtype=int)
    mask[3:7, 3:7, 3:7] = 1
    mask_img = Nifti1Image(mask, np.eye(4))

    # test .fit method
    mask = input_data.NiftiMasker(mask_strategy='epi')
    mask.fit(data_img_4d)
    assert mask._report_content['warning_message'] is None
    html = mask.generate_report()
    _check_html(html)

    # test .fit_transform method
    masker = input_data.NiftiMasker(mask_img=mask_img, standardize=True)
    masker.fit_transform(data_img_4d)
    assert mask._report_content['warning_message'] is None
    html = masker.generate_report()
    _check_html(html)


def _generate_empty_report():
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    data_img_3d = Nifti1Image(data, np.eye(4))

    # turn off reporting
    mask = input_data.NiftiMasker(reports=False)
    mask.fit(data_img_3d)
    mask.generate_report()


def test_empty_report():
    pytest.warns(UserWarning, _generate_empty_report)


def test_overlaid_report():
    pytest.importorskip('matplotlib')

    # Dummy 3D data
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    data_img_3d = Nifti1Image(data, np.eye(4))

    mask = input_data.NiftiMasker(target_affine=np.eye(3) * 8)
    html = mask.generate_report()
    assert "Please `fit` the object" in str(html)
    mask.fit(data_img_3d)
    html = mask.generate_report()
    assert '<div class="overlay">' in str(html)
