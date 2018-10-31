import warnings
from io import BytesIO
import sys

import numpy as np
from numpy.testing import assert_warns, assert_raises

from nibabel import Nifti1Image

from nilearn import datasets, image
from nilearn.plotting import html_stat_map
from nilearn.image import new_img_like
from ..js_plotting_utils import colorscale


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


def _simu_img(affine=np.eye(4)):

    # Generate simple simulated data with one "spot"
    data = np.zeros([8, 8, 8])
    data[4, 4, 4] = 1
    img = Nifti1Image(data, affine)
    return img, data


def _check_affine(affine):
    # Check positive and isotropic diagonal
    assert(affine[0, 0] == affine[1, 1])
    assert(affine[2, 2] == affine[1, 1])
    assert(affine[0, 0] > 0)

    # Check near-diagonal affine
    A, b = image.resampling.to_matrix_vector(affine)
    assert np.all((np.abs(A) > 0.001).sum(axis=0) == 1), (
        "the affine transform was not near-diagonal")


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
    assert (data_t.mask == gtruth).all()

    # Check that threshold=None keeps everything
    data_t, thresh = html_stat_map._threshold_data(data, threshold=None)
    assert ~np.ma.is_masked(data_t)

    # Check positive threshold works
    data_t, thresh = html_stat_map._threshold_data(data, threshold=1)
    gtruth = np.array([False, False, True, True, True, False, False])
    assert (data_t.mask == gtruth).all()

    # Check 0 threshold works
    data_t, thresh = html_stat_map._threshold_data(data, threshold=0)
    gtruth = np.array([False, False, False, True, False, False, False])
    assert (data_t.mask == gtruth).all()


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
    assert sprite_base64 == '////AP////8=\n'


def test_save_cmap():
    """This test covers _save_cmap as well as _bytesIO_to_base64
    """

    # Save the cmap using BytesIO
    cmap_io = BytesIO()
    html_stat_map._save_cm(cmap_io, 'cold_hot', format='raw', n_colors=2)

    # Load the colormap back in base64
    cmap_base64 = html_stat_map._bytesIO_to_base64(cmap_io)

    # Check the colormap is correct
    assert cmap_base64 == '//////////8=\n'


def test_sanitize_cm():
    # Build a cold_hot colormap
    n_colors = 4
    cmap = html_stat_map._sanitize_cm('cold_hot', n_colors=n_colors)

    # Check that it has the right number of colors
    assert cmap.N == n_colors

    # Check that there are no duplicated colors
    cmaplist = [cmap(i) for i in range(cmap.N)]
    mask = [cmaplist.count(cmap(i)) == 1 for i in range(cmap.N)]
    assert np.min(mask)


def test_mask_stat_map():

    # Generate simple simulated data with one "spot"
    img, data = _simu_img()

    # Try not to threshold anything
    mask_img, img, data_t, thre = html_stat_map._mask_stat_map(img,
                                                               threshold=None)
    assert np.max(mask_img.get_data()) == 0

    # Now threshold at zero
    mask_img, img, data_t, thre = html_stat_map._mask_stat_map(img,
                                                               threshold=0)
    assert np.min((data == 0) == mask_img.get_data())


def test_load_bg_img():

    # Generate simple simulated data with non-diagonal affine
    affine = np.eye(4)
    affine[0, 0] = -1
    affine[0, 1] = 0.1
    img, data = _simu_img(affine)

    # use empty bg_img
    bg_img, bg_min, bg_max, black_bg = html_stat_map._load_bg_img(img,
                                                                  bg_img=None)
    # Check positive isotropic, near-diagonal affine
    _check_affine(bg_img.affine)

    # Try to load the default background
    bg_img, bg_min, bg_max, black_bg = html_stat_map._load_bg_img(img)

    # Check positive isotropic, near-diagonal affine
    _check_affine(bg_img.affine)


def test_resample_stat_map():

    # Start with simple simulated data
    bg_img, data = _simu_img()

    # Now double the voxel size and mess with the affine
    affine = 2 * np.eye(4)
    affine[3, 3] = 1
    affine[0, 1] = 0.1
    stat_map_img = Nifti1Image(data, affine)

    # Make a mask for the stat image
    mask_img = new_img_like(stat_map_img, data > 0, stat_map_img.affine)

    # Now run the resampling
    stat_map_img, mask_img = html_stat_map._resample_stat_map(
        stat_map_img, bg_img, mask_img, resampling_interpolation='nearest')

    # Check positive isotropic, near-diagonal affine
    _check_affine(stat_map_img.affine)
    _check_affine(mask_img.affine)

    # Check voxel size matches bg_img
    assert stat_map_img.affine[0, 0] == bg_img.affine[0, 0], (
        "stat_map_img was not resampled at the resolution of background")
    assert mask_img.affine[0, 0] == bg_img.affine[0, 0], (
        "mask_img was not resampled at the resolution of background")


def test_json_sprite():

    # Try to generate some sprite parameters
    sprite_params = html_stat_map._json_sprite(
        shape=[4, 4, 4], affine=np.eye(4), vmin=0, vmax=1,
        cut_slices=[1, 1, 1], black_bg=True, opacity=0.5, draw_cross=False,
        annotate=True, title="A test", colorbar=True)

    # Just check that a structure was generated,
    # and test a single parameter
    assert sprite_params['overlay']['opacity'] == 0.5


def test_get_size_sprite():

    # Build some minimal sprite Parameters
    sprite_params = {'nbSlice': {'X': 4, 'Y': 4, 'Z': 4}}
    width, height = html_stat_map._get_size_sprite(sprite_params)

    # Here is a simple case: height is 4 pixels, width 3 x 4 = 12 pixels
    # there is an additional 120% factor for annotations and margins
    ratio = 1.2 * 4 / 12

    # check we received the expected width and height
    width_exp = 600
    height_exp = np.ceil(ratio * 600)
    assert width == width_exp, "html viewer does not have expected width"
    assert height == height_exp, "html viewer does not have expected height"


def test_build_sprite_data():

    # simple simulated data for stat_img and background
    bg_img, data = _simu_img()
    stat_map_img, data = _simu_img()

    # make a mask
    mask_img = new_img_like(stat_map_img, data > 0, stat_map_img.affine)

    # Get color bar and data ranges
    colors = colorscale('cold_hot', data.ravel(), threshold=0,
                        symmetric_cmap=True, vmax=1)

    # Build a sprite
    sprite = html_stat_map._build_sprite_data(
        bg_img, stat_map_img, mask_img, bg_min=0, bg_max=1, colors=colors,
        cmap='cold_hot', colorbar=True)

    # Check the presence of critical fields
    if (sys.version_info > (3, 0)):
        assert isinstance(sprite['bg_base64'], str)
        assert isinstance(sprite['stat_map_base64'], str)
        assert isinstance(sprite['cm_base64'], str)
    else:
        assert isinstance(sprite['bg_base64'], basestring)
        assert isinstance(sprite['stat_map_base64'], basestring)
        assert isinstance(sprite['cm_base64'], basestring)

    return sprite, data


def test_html_brainsprite():

    # Re use the data simulated in another test
    sprite, data = test_build_sprite_data()

    # Add meta-data to the sprite object
    sprite['params'] = html_stat_map._json_sprite(
        data.shape, np.eye(4), vmin=0, vmax=1, cut_slices=[1, 1, 1],
        black_bg=True, opacity=1, draw_cross=True, annotate=False,
        title="test", colorbar=True)

    # Create a viewer
    html = html_stat_map._html_brainsprite(sprite)
    _check_html(html)


def test_get_cut_slices():

    # Generate simple simulated data with one "spot"
    img, data = _simu_img()

    # Use automatic selection of coordinates
    cut_slices = html_stat_map._get_cut_slices(img, cut_coords=None,
                                               threshold=None)
    assert (cut_slices == [4, 4, 4]).all()

    # Check that using a single number for cut_coords raises an error
    assert_raises(ValueError, html_stat_map._get_cut_slices,
                  img, cut_coords=4, threshold=None)

    # Check that it is possible to manually specify coordinates
    cut_slices = html_stat_map._get_cut_slices(img, cut_coords=[2, 2, 2],
                                               threshold=None)
    assert (cut_slices == [2, 2, 2]).all()

    # Check that the affine does not change where the cut is done
    affine = 2 * np.eye(4)
    img = Nifti1Image(data, affine)
    cut_slices = html_stat_map._get_cut_slices(img, cut_coords=None,
                                               threshold=None)
    assert (cut_slices == [4, 4, 4]).all()


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
