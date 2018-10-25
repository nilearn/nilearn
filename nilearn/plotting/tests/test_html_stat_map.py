import warnings
from io import BytesIO

import numpy as np
from numpy.testing import assert_warns, assert_raises
from nibabel import Nifti1Image
from nilearn import datasets, image
from nilearn.plotting import html_stat_map


def _check_html(html):
    """ Check the presence of some expected code in the html
    """
    assert isinstance(html, html_stat_map.StatMapView)
    assert "var brain =" in str(html)
    assert "overlayImg" in str(html)


def _assert_warnings_in(set1, set2):
    """ Check that warnings are inside a list
    """
    assert set1.issubset(set2), ("the following warnings were not "
                                 "expected: {}").format(set1.difference(set2))


def _simu_img():

    # Generate simple simulated data with one "spot"
    data = np.zeros([8, 8, 8])
    data[4, 4, 4] = 1
    affine = np.eye(4)
    img = Nifti1Image(data, affine)
    return img, data


def test_view_stat_map():
    mni = datasets.load_mni152_template()
    with warnings.catch_warnings(record=True) as w:
        # Create a fake functional image by resample the template
        img = image.resample_img(mni, target_affine=3 * np.eye(3))
        html = html_stat_map.view_stat_map(img)
        _check_html(html)
        html = html_stat_map.view_stat_map(img, threshold='95%')
        _check_html(html)
        html = html_stat_map.view_stat_map(img, bg_img=mni)
        _check_html(html)
        html = html_stat_map.view_stat_map(img, bg_img=None)
        _check_html(html)
        html = html_stat_map.view_stat_map(img, threshold=2., vmax=4.)
        _check_html(html)
        html = html_stat_map.view_stat_map(img, symmetric_cmap=False)
        img_4d = image.new_img_like(img, img.get_data()[:, :, :, np.newaxis])
        assert len(img_4d.shape) == 4
        html = html_stat_map.view_stat_map(img_4d, threshold=2., vmax=4.)
        _check_html(html)
    warning_categories = set(warning_.category for warning_ in w)
    expected_categories = set([FutureWarning, UserWarning,
                               DeprecationWarning])
    _assert_warnings_in(warning_categories, expected_categories)


def test_data2sprite():

    # Generate a simulated volume with a square inside
    data = np.zeros([8, 8, 8])
    data[2:6, 2:6, 2:6] = 1

    # turn that into a sprite and check it has the right shape
    sprite = html_stat_map._data2sprite(data)
    assert sprite.shape == (24, 24)

    # Generate ground truth for the sprite
    Z = np.zeros([8, 8])
    Zr = np.zeros([2, 8])
    Cr = np.tile(np.array([[0, 0, 1, 1, 1, 1, 0, 0]]), [4, 1])
    C = np.concatenate((Zr, Cr, Zr), axis=0)
    gtruth = np.concatenate((np.concatenate((Z, Z, C), axis=1),
                             np.concatenate((C, C, C), axis=1),
                             np.concatenate((Z, Z, Z), axis=1)),
                            axis=0)

    # Check that the sprite matches ground truth
    assert (sprite == gtruth).all()


def test_get_vmin_vmax():

    # Generate a simulated volume with a square inside
    data = np.zeros([8, 8, 8])
    data[2:6, 2:6, 2:6] = 1

    # Check default vmin, vmax
    vmin, vmax = html_stat_map._get_vmin_vmax(data)
    assert (vmin == 0) and (vmax == 1)

    # Force vmin and vmax
    vmin, vmax = html_stat_map._get_vmin_vmax(data, vmin=.5, vmax=.7)
    assert (vmin == .5) and (vmax == .7)

    # a warning should be issued if vmax or vmin is NaN
    assert_warns(UserWarning, html_stat_map._get_vmin_vmax,
                 data, vmin=.5, vmax=np.nan)
    assert_warns(UserWarning, html_stat_map._get_vmin_vmax,
                 data, vmin=np.nan, vmax=0.7)

    # an error should be raised if vmax is smaller than vmin
    assert_raises(ValueError, html_stat_map._get_vmin_vmax,
                  data, vmin=3, vmax=0.7)


def test_threshold_data():

    data = np.arange(-3, 4)

    # Check that an 'auto' threshold leaves at least one element
    data_t, thresh = html_stat_map._threshold_data(data, threshold='auto')
    gtruth = np.array([False, True, True, True, True, True, False])
    assert((data_t.mask == gtruth).all())

    # Check that threshold=None keeps everything
    data_t, thresh = html_stat_map._threshold_data(data, threshold=None)
    assert(~np.ma.is_masked(data_t))

    # Check positive threshold works
    data_t, thresh = html_stat_map._threshold_data(data, threshold=1)
    gtruth = np.array([False, False, True, True, True, False, False])
    assert((data_t.mask == gtruth).all())

    # Check 0 threshold works
    data_t, thresh = html_stat_map._threshold_data(data, threshold=0)
    gtruth = np.array([False, False, False, True, False, False, False])
    assert((data_t.mask == gtruth).all())


def test_save_sprite():
    """This test covers _save_sprite as well as _bytesIO_to_base64
    """

    # Generate a simulated volume with a square inside
    data = np.zeros([2, 1, 1])
    data[0, 0, 0] = 1
    mask = data > 0

    # Save the sprite using BytesIO
    sprite_io = BytesIO()
    html_stat_map._save_sprite(data, sprite_io, vmin=0, vmax=1,
                               mask=mask, format='raw')

    # Load the sprite back in base64
    sprite_base64 = html_stat_map._bytesIO_to_base64(sprite_io)

    # Check the sprite is correct
    assert(sprite_base64 == '////AP////8=\n')


def test_save_cmap():
    """This test covers _save_cmap as well as _bytesIO_to_base64
    """

    # Save the cmap using BytesIO
    cmap_io = BytesIO()
    html_stat_map._save_cm(cmap_io, 'cold_hot', format='raw', n_colors=2)

    # Load the colormap back in base64
    cmap_base64 = html_stat_map._bytesIO_to_base64(cmap_io)

    # Check the colormap is correct
    assert(cmap_base64 == '//////////8=\n')


def test_sanitize_cm():
    # Build a cold_hot colormap
    n_colors = 4
    cmap = html_stat_map._sanitize_cm('cold_hot', n_colors=n_colors)

    # Check that it has the right number of colors
    assert(cmap.N == n_colors)

    # Check that there are no duplicated colors
    cmaplist = [cmap(i) for i in range(cmap.N)]
    mask = [cmaplist.count(cmap(i)) == 1 for i in range(cmap.N)]
    assert(np.min(mask))


def test_mask_stat_map():

    # Generate simple simulated data with one "spot"
    img, data = _simu_img()

    # Try not to threshold anything
    mask_img, img, data_t, thre = html_stat_map._mask_stat_map(img,
                                                               threshold=None)
    assert(np.max(mask_img.get_data()) == 0)

    # Now threshold the zero
    mask_img, img, data_t, thre = html_stat_map._mask_stat_map(img,
                                                               threshold=0)
    assert(np.min((data == 0) == mask_img.get_data()))


def test_get_cut_slices():

    # Generate simple simulated data with one "spot"
    img, data = _simu_img()

    # Use automatic selection of coordinates
    cut_slices = html_stat_map._get_cut_slices(img, cut_coords=None,
                                               threshold=None)
    assert((cut_slices == [4, 4, 4]).all())

    # Check that using a single number for cut_coords raises an error
    assert_raises(ValueError, html_stat_map._get_cut_slices,
                  img, cut_coords=4, threshold=None)

    # Check that it is possible to manually specify coordinates
    cut_slices = html_stat_map._get_cut_slices(img, cut_coords=[2, 2, 2],
                                               threshold=None)
    assert((cut_slices == [2, 2, 2]).all())

    # Check that the affine does not change where the cut is done
    affine = 2 * np.eye(4)
    img = Nifti1Image(data, affine)
    cut_slices = html_stat_map._get_cut_slices(img, cut_coords=None,
                                               threshold=None)
    assert((cut_slices == [4, 4, 4]).all())
