import base64
import warnings
from io import BytesIO

import numpy as np
import pytest
from matplotlib import pyplot as plt
from nibabel import Nifti1Image

from nilearn import datasets, image
from nilearn.image import get_data, new_img_like
from nilearn.plotting import html_stat_map
from nilearn.plotting.js_plotting_utils import colorscale


def _check_html(html_view, title=None):
    """Check the presence of some expected code in the html viewer."""
    assert isinstance(html_view, html_stat_map.StatMapView)
    assert "var brain =" in str(html_view)
    assert "overlayImg" in str(html_view)
    if title is not None:
        assert f"<title>{title}</title>" in str(html_view)


def _simulate_img(affine=None):
    """Simulate data with one "spot".

    Returns
    -------
    img

    data
    """
    if affine is None:
        affine = np.eye(4)
    data = np.zeros([8, 8, 8])
    data[4, 4, 4] = 1
    img = Nifti1Image(data, affine)
    return img, data


def _check_affine(affine):
    """Check positive, isotropic, near-diagonal affine."""
    assert affine[0, 0] == affine[1, 1]
    assert affine[2, 2] == affine[1, 1]
    assert affine[0, 0] > 0

    A, b = image.resampling.to_matrix_vector(affine)
    assert np.all(
        (np.abs(A) > 0.001).sum(axis=0) == 1
    ), "the affine transform was not near-diagonal"


def test_data_to_sprite():
    # Simulate data and turn into sprite
    data = np.zeros([8, 8, 8])
    data[2:6, 2:6, 2:6] = 1
    sprite = html_stat_map._data_to_sprite(data)

    # Generate ground truth for the sprite
    Z = np.zeros([8, 8])
    Zr = np.zeros([2, 8])
    Cr = np.tile(np.array([[0, 0, 1, 1, 1, 1, 0, 0]]), [4, 1])
    C = np.concatenate((Zr, Cr, Zr), axis=0)
    gtruth = np.concatenate(
        (
            np.concatenate((Z, Z, C), axis=1),
            np.concatenate((C, C, C), axis=1),
            np.concatenate((Z, Z, Z), axis=1),
        ),
        axis=0,
    )

    assert sprite.shape == gtruth.shape, "shape of sprite not as expected"
    assert (sprite == gtruth).all(), "simulated sprite not as expected"


def test_threshold_data():
    data = np.arange(-3, 4)

    # Check that an 'auto' threshold leaves at least one element
    data_t, mask, thresh = html_stat_map._threshold_data(
        data, threshold="auto"
    )
    gtruth_m = np.array([False, True, True, True, True, True, False])
    gtruth_d = np.array([-3, 0, 0, 0, 0, 0, 3])
    assert (mask == gtruth_m).all()
    assert (data_t == gtruth_d).all()

    # Check that threshold=None keeps everything
    data_t, mask, thresh = html_stat_map._threshold_data(data, threshold=None)
    assert np.all(np.logical_not(mask))
    assert np.all(data_t == data)

    # Check positive threshold works
    data_t, mask, thresh = html_stat_map._threshold_data(data, threshold=1)
    gtruth = np.array([False, False, True, True, True, False, False])
    assert (mask == gtruth).all()

    # Check 0 threshold works
    data_t, mask, thresh = html_stat_map._threshold_data(data, threshold=0)
    gtruth = np.array([False, False, False, True, False, False, False])
    assert (mask == gtruth).all()

    # Check that overly lenient threshold returns array
    data = np.arange(3, 10)
    data_t, mask, thresh = html_stat_map._threshold_data(data, threshold=2)
    gtruth = np.full(7, False)
    assert (mask == gtruth).all()


def test_save_sprite(rng):
    """Test covers _save_sprite as well as _bytes_io_to_base64."""
    # Generate a simulated volume with a square inside
    data = rng.uniform(size=140).reshape(7, 5, 4)
    mask = np.zeros((7, 5, 4), dtype=int)
    mask[1:-1, 1:-1, 1:-1] = 1
    # Save the sprite using BytesIO
    sprite_io = BytesIO()
    html_stat_map._save_sprite(
        data, sprite_io, vmin=0, vmax=1, mask=mask, format="png"
    )

    # Load the sprite back in base64
    sprite_base64 = html_stat_map._bytes_io_to_base64(sprite_io)

    decoded_io = BytesIO()
    decoded_io.write(base64.b64decode(sprite_base64))
    decoded_io.seek(0)
    img = plt.imread(decoded_io, format="png")
    correct_img = np.ma.array(
        html_stat_map._data_to_sprite(data),
        mask=html_stat_map._data_to_sprite(mask),
    )
    correct_img = plt.Normalize(0, 1)(correct_img)
    cmapped = plt.get_cmap("Greys")(correct_img)
    assert np.allclose(img, cmapped, atol=0.1)


@pytest.mark.parametrize("cmap", ["tab10", "cold_hot"])
@pytest.mark.parametrize("n_colors", [7, 20])
def test_save_cmap(cmap, n_colors):
    """Test covers _save_cmap as well as _bytes_io_to_base64."""
    # Save the cmap using BytesIO
    cmap_io = BytesIO()
    html_stat_map._save_cm(cmap_io, cmap, format="png", n_colors=n_colors)

    # Load the colormap back in base64
    cmap_base64 = html_stat_map._bytes_io_to_base64(cmap_io)

    decoded_io = BytesIO()
    decoded_io.write(base64.b64decode(cmap_base64))
    decoded_io.seek(0)
    img = plt.imread(decoded_io, format="png")
    expected = plt.get_cmap(cmap)(np.linspace(0, 1, n_colors))
    assert np.allclose(img, expected, atol=0.1)


def test_mask_stat_map():
    # Generate simple simulated data with one "spot"
    img, data = _simulate_img()

    # Try not to threshold anything
    mask_img, img, data_t, thresh = html_stat_map._mask_stat_map(
        img, threshold=None
    )
    assert np.max(get_data(mask_img)) == 0

    # Now threshold at zero
    mask_img, img, data_t, thresh = html_stat_map._mask_stat_map(
        img, threshold=0
    )
    assert np.min((data == 0) == get_data(mask_img))


def test_load_bg_img(affine_eye):
    # Generate simple simulated data with non-diagonal affine
    affine = affine_eye
    affine[0, 0] = -1
    affine[0, 1] = 0.1
    img, _ = _simulate_img(affine)

    # use empty bg_img
    bg_img, _, _, _ = html_stat_map._load_bg_img(img, bg_img=None)
    # Check positive isotropic, near-diagonal affine
    _check_affine(bg_img.affine)

    # Try to load the default background
    bg_img, _, _, _ = html_stat_map._load_bg_img(img)

    # Check positive isotropic, near-diagonal affine
    _check_affine(bg_img.affine)


def test_get_bg_mask_and_cmap():
    # non-regression test for issue #3120 (bg image was masked with mni
    # template mask)
    img, _ = _simulate_img()
    mask, cmap = html_stat_map._get_bg_mask_and_cmap(img, False)
    assert (mask == np.zeros(img.shape, dtype=bool)).all()


def test_resample_stat_map(affine_eye):
    # Start with simple simulated data
    bg_img, data = _simulate_img()

    # Now double the voxel size and mess with the affine
    affine = 2 * affine_eye
    affine[3, 3] = 1
    affine[0, 1] = 0.1
    stat_map_img = Nifti1Image(data, affine)

    # Make a mask for the stat image
    mask_img = new_img_like(stat_map_img, data > 0, stat_map_img.affine)

    # Now run the resampling
    stat_map_img, mask_img = html_stat_map._resample_stat_map(
        stat_map_img, bg_img, mask_img, resampling_interpolation="nearest"
    )

    # Check positive isotropic, near-diagonal affine
    _check_affine(stat_map_img.affine)
    _check_affine(mask_img.affine)

    # Check voxel size matches bg_img
    assert (
        stat_map_img.affine[0, 0] == bg_img.affine[0, 0]
    ), "stat_map_img was not resampled at the resolution of background"
    assert (
        mask_img.affine[0, 0] == bg_img.affine[0, 0]
    ), "mask_img was not resampled at the resolution of background"


def test_json_view_params(affine_eye):
    # Try to generate some sprite parameters
    params = html_stat_map._json_view_params(
        shape=[4, 4, 4],
        affine=affine_eye,
        vmin=0,
        vmax=1,
        cut_slices=[1, 1, 1],
        black_bg=True,
        opacity=0.5,
        draw_cross=False,
        annotate=True,
        title="A test",
        colorbar=True,
        value=True,
    )

    # Just check that a structure was generated,
    # and test a single parameter
    assert params["overlay"]["opacity"] == 0.5


def test_json_view_size():
    # Build some minimal sprite Parameters
    sprite_params = {"nbSlice": {"X": 4, "Y": 4, "Z": 4}}
    width, height = html_stat_map._json_view_size(sprite_params)

    # This is a simple case: height is 4 pixels, width 3 x 4 = 12 pixels
    # with an additional 120% height factor for annotations and margins
    ratio = 1.2 * 4 / 12

    # check we received the expected width and height
    width_exp = 600
    height_exp = np.ceil(ratio * 600)
    assert width == width_exp, "html viewer does not have expected width"
    assert height == height_exp, "html viewer does not have expected height"


def _get_data_and_json_view(black_bg, cbar):
    # simple simulated data for stat_img and background
    bg_img, data = _simulate_img()
    stat_map_img, data = _simulate_img()

    # make a mask
    mask_img = new_img_like(stat_map_img, data > 0, stat_map_img.affine)

    # Get color bar and data ranges
    colors = colorscale(
        "cold_hot", data.ravel(), threshold=0, symmetric_cmap=True, vmax=1
    )

    # Build a sprite
    json_view = html_stat_map._json_view_data(
        bg_img,
        stat_map_img,
        mask_img,
        bg_min=0,
        bg_max=1,
        black_bg=black_bg,
        colors=colors,
        cmap="cold_hot",
        colorbar=cbar,
    )
    return data, json_view


@pytest.mark.parametrize("black_bg", [True, False])
@pytest.mark.parametrize("cbar", [True, False])
def test_json_view_data(black_bg, cbar):
    _, json_view = _get_data_and_json_view(black_bg, cbar)
    # Check the presence of critical fields
    assert isinstance(json_view["bg_base64"], str)
    assert isinstance(json_view["stat_map_base64"], str)
    assert isinstance(json_view["cm_base64"], str)


@pytest.mark.parametrize("black_bg", [True, False])
@pytest.mark.parametrize("cbar", [True, False])
def test_json_view_to_html(affine_eye, black_bg, cbar):
    data, json_view = _get_data_and_json_view(black_bg, cbar)
    json_view["params"] = html_stat_map._json_view_params(
        data.shape,
        affine_eye,
        vmin=0,
        vmax=1,
        cut_slices=[1, 1, 1],
        black_bg=True,
        opacity=1,
        draw_cross=True,
        annotate=False,
        title="test",
        colorbar=True,
    )

    # Create a viewer
    html_view = html_stat_map._json_view_to_html(json_view)
    _check_html(html_view)


def test_get_cut_slices(affine_eye):
    # Generate simple simulated data with one "spot"
    img, data = _simulate_img()

    # Use automatic selection of coordinates
    cut_slices = html_stat_map._get_cut_slices(
        img, cut_coords=None, threshold=None
    )
    assert (cut_slices == [4, 4, 4]).all()

    # Check that using a single number for cut_coords raises an error
    with pytest.raises(ValueError):
        html_stat_map._get_cut_slices(img, cut_coords=4, threshold=None)

    # Check that it is possible to manually specify coordinates
    cut_slices = html_stat_map._get_cut_slices(
        img, cut_coords=[2, 2, 2], threshold=None
    )
    assert (cut_slices == [2, 2, 2]).all()

    # Check that the affine does not change where the cut is done
    affine = 2 * affine_eye
    img = Nifti1Image(data, affine)
    cut_slices = html_stat_map._get_cut_slices(
        img, cut_coords=None, threshold=None
    )
    assert (cut_slices == [4, 4, 4]).all()


def test_view_img():
    mni = datasets.load_mni152_template(resolution=2)
    with warnings.catch_warnings(record=True) as w:
        # Create a fake functional image by resample the template
        img = image.resample_img(
            mni,
            target_affine=3 * np.eye(3),
            copy_header=True,
            force_resample=True,
        )
        html_view = html_stat_map.view_img(img)
        _check_html(html_view, title="Slice viewer")
        html_view = html_stat_map.view_img(
            img, threshold="95%", title="SOME_TITLE"
        )
        _check_html(html_view, title="SOME_TITLE")
        html_view = html_stat_map.view_img(img, bg_img=mni)
        _check_html(html_view)
        html_view = html_stat_map.view_img(img, bg_img=None)
        _check_html(html_view)
        html_view = html_stat_map.view_img(img, threshold=2.0, vmax=4.0)
        _check_html(html_view)
        html_view = html_stat_map.view_img(img, symmetric_cmap=False)
        img_4d = image.new_img_like(img, get_data(img)[:, :, :, np.newaxis])
        assert len(img_4d.shape) == 4
        html_view = html_stat_map.view_img(img_4d, threshold=2.0, vmax=4.0)
        _check_html(html_view)
        html_view = html_stat_map.view_img(img_4d, threshold=1e6)
        _check_html(html_view)
        html_view = html_stat_map.view_img(img_4d, width_view=1000)
        _check_html(html_view)

    # Check that all warnings were expected
    warnings_set = {warning_.category for warning_ in w}
    expected_set = {FutureWarning, UserWarning, DeprecationWarning}
    assert warnings_set.issubset(expected_set), (
        "the following warnings were not expected: "
        f"{warnings_set.difference(expected_set)}"
    )
