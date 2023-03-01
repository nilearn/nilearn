# Tests for functions in surf_plotting.py
import numpy as np
import nibabel
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pytest
import re
import tempfile
import os
import unittest.mock as mock

from nilearn.plotting.img_plotting import MNI152TEMPLATE
from nilearn.plotting.surf_plotting import (plot_surf, plot_surf_stat_map,
                                            plot_surf_roi, plot_img_on_surf,
                                            plot_surf_contours,
                                            _get_ticks_matplotlib,
                                            _compute_facecolors_matplotlib)
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_data, load_surf_mesh
from nilearn.surface.testing_utils import generate_surf
from numpy.testing import assert_array_equal
from nilearn.plotting.surf_plotting import VALID_HEMISPHERES, VALID_VIEWS
from nilearn.plotting.displays import PlotlySurfaceFigure

try:
    import plotly.graph_objects as go  # noqa
except ImportError:
    PLOTLY_INSTALLED = False
else:
    PLOTLY_INSTALLED = True

try:
    import kaleido  # noqa
except ImportError:
    KALEIDO_INSTALLED = False
else:
    KALEIDO_INSTALLED = True

try:
    import IPython.display  # noqa
except ImportError:
    IPYTHON_INSTALLED = False
else:
    IPYTHON_INSTALLED = True

EXPECTED_CAMERAS_PLOTLY = {"left": {"anterior": "anterior",
                                    "posterior": "posterior",
                                    "medial": "right",
                                    "lateral": "left",
                                    "dorsal": "dorsal",
                                    "ventral": "ventral"},
                           "right": {"anterior": "anterior",
                                     "posterior": "posterior",
                                     "medial": "left",
                                     "lateral": "right",
                                     "dorsal": "dorsal",
                                     "ventral": "ventral"}}


EXPECTED_VIEW_MATPLOTLIB = {"left": {"anterior": (0, 90),
                                     "posterior": (0, 270),
                                     "medial": (0, 0),
                                     "lateral": (0, 180),
                                     "dorsal": (90, 0),
                                     "ventral": (270, 0)},
                            "right": {"anterior": (0, 90),
                                      "posterior": (0, 270),
                                      "medial": (0, 180),
                                      "lateral": (0, 0),
                                      "dorsal": (90, 0),
                                      "ventral": (270, 0)}}


@pytest.fixture
def expected_cameras_plotly(hemi, view):
    return EXPECTED_CAMERAS_PLOTLY[hemi][view]


@pytest.mark.parametrize("hemi", VALID_HEMISPHERES)
@pytest.mark.parametrize("view", VALID_VIEWS)
def test_set_view_plot_surf_plotly(hemi, view, expected_cameras_plotly):
    from nilearn.plotting.surf_plotting import _set_view_plot_surf_plotly
    assert _set_view_plot_surf_plotly(hemi, view) == expected_cameras_plotly


@pytest.fixture
def expected_view_matplotlib(hemi, view):
    return EXPECTED_VIEW_MATPLOTLIB[hemi][view]


@pytest.mark.parametrize("hemi", VALID_HEMISPHERES)
@pytest.mark.parametrize("view", VALID_VIEWS)
def test_set_view_plot_surf_matplotlib(hemi, view, expected_view_matplotlib):
    from nilearn.plotting.surf_plotting import _set_view_plot_surf_matplotlib
    assert(_set_view_plot_surf_matplotlib(hemi, view)
           == expected_view_matplotlib)


def test_surface_figure():
    from nilearn.plotting.displays import SurfaceFigure
    s = SurfaceFigure()
    assert s.output_file is None
    assert s.figure is None
    with pytest.raises(NotImplementedError):
        s.show()
    with pytest.raises(ValueError, match="You must provide an output file"):
        s._check_output_file()
    s._check_output_file("foo.png")
    assert s.output_file == "foo.png"
    s = SurfaceFigure(output_file="bar.png")
    assert s.output_file == "bar.png"


@pytest.mark.skipif(PLOTLY_INSTALLED,
                    reason='Plotly is installed.')
def test_plotly_surface_figure_import_error():
    """Test that an ImportError is raised when instantiating a
    PlotlySurfaceFigure without having Plotly installed.
    """
    with pytest.raises(ImportError, match="Plotly is required"):
        PlotlySurfaceFigure()


@pytest.mark.skipif(not PLOTLY_INSTALLED or KALEIDO_INSTALLED,
                    reason=("This test only runs if Plotly is "
                            "installed, but not kaleido."))
def test_plotly_surface_figure_savefig_error():
    """Test that an ImportError is raised when saving a
    PlotlySurfaceFigure without having kaleido installed.
    """
    with pytest.raises(ImportError, match="`kaleido` is required"):
        PlotlySurfaceFigure().savefig()


@pytest.mark.skipif(not PLOTLY_INSTALLED or not KALEIDO_INSTALLED,
                    reason=("Plotly and/or kaleido not installed; "
                            "required for this test."))
def test_plotly_surface_figure():
    ps = PlotlySurfaceFigure()
    assert ps.output_file is None
    assert ps.figure is None
    ps.show()
    with pytest.raises(ValueError, match="You must provide an output file"):
        ps.savefig()
    ps.savefig('foo.png')


@pytest.mark.skipif(not PLOTLY_INSTALLED or not IPYTHON_INSTALLED,
                    reason=("Plotly and/or Ipython is not installed; "
                            "required for this test."))
@pytest.mark.parametrize("renderer", ['png', 'jpeg', 'svg'])
def test_plotly_show(renderer):
    ps = PlotlySurfaceFigure(go.Figure())
    assert ps.output_file is None
    assert ps.figure is not None
    with mock.patch("IPython.display.display") as mock_display:
        ps.show(renderer=renderer)
    assert len(mock_display.call_args.args) == 1
    key = 'svg+xml' if renderer == 'svg' else renderer
    assert f'image/{key}' in mock_display.call_args.args[0]


@pytest.mark.skipif(not PLOTLY_INSTALLED or not KALEIDO_INSTALLED,
                    reason=("Plotly and/or kaleido not installed; "
                            "required for this test."))
def test_plotly_savefig(tmpdir):
    ps = PlotlySurfaceFigure(go.Figure(), output_file=str(tmpdir / "foo.png"))
    assert ps.output_file == str(tmpdir / "foo.png")
    assert ps.figure is not None
    ps.savefig()
    assert os.path.exists(str(tmpdir / "foo.png"))


@pytest.mark.skipif(not PLOTLY_INSTALLED,
                    reason='Plotly is not installed; required for this test.')
@pytest.mark.parametrize("input_obj", ["foo", Figure(), ["foo", "bar"]])
def test_instantiation_error_plotly_surface_figure(input_obj):
    with pytest.raises(TypeError,
                       match=("`PlotlySurfaceFigure` accepts only "
                              "plotly figure objects.")):
        PlotlySurfaceFigure(input_obj)


def test_set_view_plot_surf_errors():
    from nilearn.plotting.surf_plotting import (_set_view_plot_surf_matplotlib,
                                                _set_view_plot_surf_plotly)
    with pytest.raises(ValueError,
                       match="hemi must be one of"):
        _set_view_plot_surf_matplotlib("foo", "medial")
        _set_view_plot_surf_plotly("bar", "anterior")
    with pytest.raises(ValueError,
                       match="view must be one of"):
        _set_view_plot_surf_matplotlib("left", "foo")
        _set_view_plot_surf_matplotlib("right", "bar")
        _set_view_plot_surf_plotly("left", "foo")
        _set_view_plot_surf_plotly("right", "bar")


def test_configure_title_plotly():
    from nilearn.plotting.surf_plotting import _configure_title_plotly
    assert _configure_title_plotly(None, None) == dict()
    assert _configure_title_plotly(None, 22) == dict()
    config = _configure_title_plotly("Test Title", 22, color="green")
    assert config["text"] == "Test Title"
    assert config["x"] == 0.5
    assert config["y"] == 0.96
    assert config["xanchor"] == "center"
    assert config["yanchor"] == "top"
    assert config["font"]["size"] == 22
    assert config["font"]["color"] == "green"


@pytest.mark.parametrize("data,expected",
                         [(np.linspace(0, 1, 100), (0, 1)),
                          (np.linspace(-.7, -.01, 40), (-.7, -.01))])
def test_get_bounds(data, expected):
    from nilearn.plotting.surf_plotting import _get_bounds
    assert _get_bounds(data) == expected
    assert _get_bounds(data, vmin=.2) == (.2, expected[1])
    assert _get_bounds(data, vmax=.8) == (expected[0], .8)
    assert _get_bounds(data, vmin=.1, vmax=.8) == (.1, .8)


def test_plot_surf_engine_error():
    mesh = generate_surf()
    with pytest.raises(ValueError,
                       match="Unknown plotting engine"):
        plot_surf(mesh, engine="foo")


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf(engine, tmp_path):
    if not PLOTLY_INSTALLED and engine == "plotly":
        pytest.skip('Plotly is not installed; required for this test.')
    mesh = generate_surf()
    rng = np.random.RandomState(42)
    bg = rng.standard_normal(size=mesh[0].shape[0])

    # Plot mesh only
    plot_surf(mesh, engine=engine)

    # Plot mesh with background
    plot_surf(mesh, bg_map=bg, engine=engine)
    plot_surf(mesh, bg_map=bg, darkness=0.5, engine=engine)
    plot_surf(mesh, bg_map=bg, alpha=0.5,
              output_file=tmp_path / 'tmp.png', engine=engine)

    # Plot different views
    plot_surf(mesh, bg_map=bg, hemi='right', engine=engine)
    plot_surf(mesh, bg_map=bg, view='medial', engine=engine)
    plot_surf(mesh, bg_map=bg, hemi='right', view='medial', engine=engine)

    # Plot with colorbar
    plot_surf(mesh, bg_map=bg, colorbar=True, engine=engine)
    plot_surf(mesh, bg_map=bg, colorbar=True, cbar_vmin=0,
              cbar_vmax=150, cbar_tick_format="%i", engine=engine)
    # Save execution time and memory
    plt.close()

    # Plot with title
    display = plot_surf(mesh, bg_map=bg, title='Test title',
                        engine=engine)
    if engine == 'matplotlib':
        assert len(display.axes) == 1
        assert display.axes[0].title._text == 'Test title'


def test_plot_surf_avg_method():
    mesh = generate_surf()
    rng = np.random.RandomState(42)
    # Plot with avg_method
    ## Test all built-in methods and check
    mapp = rng.standard_normal(size=mesh[0].shape[0])
    mesh_ = load_surf_mesh(mesh)
    coords, faces = mesh_[0], mesh_[1]

    for method in ['mean', 'median', 'min', 'max']:
        display = plot_surf(mesh, surf_map=mapp,
                            avg_method=method,
                            engine='matplotlib')
        if method == 'mean':
            agg_faces = np.mean(mapp[faces], axis=1)
        elif method == 'median':
            agg_faces = np.median(mapp[faces], axis=1)
        elif method == 'min':
            agg_faces = np.min(mapp[faces], axis=1)
        elif method == 'max':
            agg_faces = np.max(mapp[faces], axis=1)
        vmin = np.min(agg_faces)
        vmax = np.max(agg_faces)
        agg_faces -= vmin
        agg_faces /= (vmax - vmin)
        cmap = plt.get_cmap(plt.rcParamsDefault['image.cmap'])
        assert_array_equal(
            cmap(agg_faces),
            display._axstack.as_list()[0].collections[0]._facecolors
        )
    ## Try custom avg_method
    def custom_avg_function(vertices):
        return vertices[0] * vertices[1] * vertices[2]
    plot_surf(
        mesh,
        surf_map=rng.standard_normal(size=mesh[0].shape[0]),
        avg_method=custom_avg_function,
        engine='matplotlib',
    )
    # Save execution time and memory
    plt.close()


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_error(engine):
    if not PLOTLY_INSTALLED and engine == "plotly":
        pytest.skip('Plotly is not installed; required for this test.')
    mesh = generate_surf()
    rng = np.random.RandomState(42)

    # Wrong inputs for view or hemi
    with pytest.raises(ValueError, match='view must be one of'):
        plot_surf(mesh, view='middle', engine=engine)
    with pytest.raises(ValueError, match='hemi must be one of'):
        plot_surf(mesh, hemi='lft', engine=engine)

    # Wrong size of background image
    with pytest.raises(
            ValueError,
            match='bg_map does not have the same number of vertices'):
        plot_surf(mesh,
                  bg_map=rng.standard_normal(size=mesh[0].shape[0] - 1),
                  engine=engine
                  )

    # Wrong size of surface data
    with pytest.raises(
        ValueError, match="surf_map does not have the same number of vertices"
    ):
        plot_surf(
            mesh,
            surf_map=rng.standard_normal(size=mesh[0].shape[0] + 1),
            engine=engine
        )

    with pytest.raises(
        ValueError, match="'surf_map' can only have one dimension"
    ):
        plot_surf(
            mesh,
            surf_map=rng.standard_normal(size=(mesh[0].shape[0], 2)),
            engine=engine
        )


def test_plot_surf_avg_method_errors():
    mesh = generate_surf()
    rng = np.random.RandomState(42)
    with pytest.raises(
        ValueError,
        match=(
            "Array computed with the custom "
            "function from avg_method does "
            "not have the correct shape"
        )
    ):
        def custom_avg_function(vertices):
            return [vertices[0] * vertices[1], vertices[2]]

        plot_surf(mesh,
                  surf_map=rng.standard_normal(
                      size=mesh[0].shape[0]),
                  avg_method=custom_avg_function,
                  engine='matplotlib'
                  )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "avg_method should be either "
            "['mean', 'median', 'max', 'min'] "
            "or a custom function"
        )
    ):
        custom_avg_function = dict()

        plot_surf(mesh,
                  surf_map=rng.standard_normal(
                      size=mesh[0].shape[0]),
                  avg_method=custom_avg_function,
                  engine='matplotlib'
                  )

        plot_surf(mesh,
                  surf_map=rng.standard_normal(
                      size=mesh[0].shape[0]),
                  avg_method="foo",
                  engine='matplotlib'
                  )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Array computed with the custom function "
            "from avg_method should be an array of "
            "numbers (int or float)"
        )
    ):
        def custom_avg_function(vertices):
            return "string"

        plot_surf(mesh,
                  surf_map=rng.standard_normal(
                      size=mesh[0].shape[0]),
                  avg_method=custom_avg_function,
                  engine='matplotlib'
                  )


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map(engine):
    if not PLOTLY_INSTALLED and engine == "plotly":
        pytest.skip('Plotly is not installed; required for this test.')
    mesh = generate_surf()
    rng = np.random.RandomState(42)
    bg = rng.standard_normal(size=mesh[0].shape[0])
    data = 10 * rng.standard_normal(size=mesh[0].shape[0])

    # Plot mesh with stat map
    plot_surf_stat_map(mesh, stat_map=data, engine=engine)
    plot_surf_stat_map(mesh, stat_map=data, colorbar=True, engine=engine)
    plot_surf_stat_map(mesh, stat_map=data, alpha=1, engine=engine)

    # Plot mesh with background and stat map
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg, engine=engine)
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg,
                       bg_on_data=True, darkness=0.5,
                       engine=engine)
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg, colorbar=True,
                       bg_on_data=True, darkness=0.5, engine=engine)

    # Plot with title
    display = plot_surf_stat_map(mesh, stat_map=data, bg_map=bg,
                                 title="Stat map title")
    assert display.axes[0].title._text == "Stat map title"

    # Apply threshold
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg,
                       bg_on_data=True, darkness=0.5,
                       threshold=0.3, engine=engine)
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg, colorbar=True,
                       bg_on_data=True, darkness=0.5,
                       threshold=0.3, engine=engine)

    # Change colorbar tick format
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg, colorbar=True,
                       bg_on_data=True, darkness=0.5, cbar_tick_format="%.2g",
                       engine=engine)

    # Change vmax
    plot_surf_stat_map(mesh, stat_map=data, vmax=5, engine=engine)
    plot_surf_stat_map(mesh, stat_map=data, vmax=5,
                       colorbar=True, engine=engine)

    # Change colormap
    plot_surf_stat_map(mesh, stat_map=data, cmap='cubehelix', engine=engine)
    plot_surf_stat_map(mesh, stat_map=data, cmap='cubehelix',
                       colorbar=True, engine=engine)

    plt.close()


def test_plot_surf_stat_map_matplotlib_specific():
    mesh = generate_surf()
    rng = np.random.RandomState(42)
    data = 10 * rng.standard_normal(size=mesh[0].shape[0])
    # Plot to axes
    axes = plt.subplots(ncols=2, subplot_kw={'projection': '3d'})[1]
    for ax in axes.flatten():
        plot_surf_stat_map(mesh, stat_map=data, ax=ax)
    axes = plt.subplots(ncols=2, subplot_kw={'projection': '3d'})[1]
    for ax in axes.flatten():
        plot_surf_stat_map(mesh, stat_map=data, ax=ax, colorbar=True)

    fig = plot_surf_stat_map(mesh, stat_map=data, colorbar=False)
    assert len(fig.axes) == 1

    # symmetric_cbar
    fig = plot_surf_stat_map(
        mesh, stat_map=data, colorbar=True, symmetric_cbar=True)
    fig.canvas.draw()
    assert len(fig.axes) == 2
    yticklabels = fig.axes[1].get_yticklabels()
    first, last = yticklabels[0].get_text(), yticklabels[-1].get_text()
    assert float(first) == - float(last)

    # no symmetric_cbar
    fig = plot_surf_stat_map(
        mesh, stat_map=data, colorbar=True, symmetric_cbar=False)
    fig.canvas.draw()
    assert len(fig.axes) == 2
    yticklabels = fig.axes[1].get_yticklabels()
    first, last = yticklabels[0].get_text(), yticklabels[-1].get_text()
    assert float(first) != - float(last)

    # Test handling of nan values in texture data
    # Add nan values in the texture
    data[2] = np.nan
    # Plot the surface stat map
    fig = plot_surf_stat_map(mesh, stat_map=data)
    # Check that the resulting plot facecolors contain no transparent faces
    # (last column equals zero) even though the texture contains nan values
    assert(mesh[1].shape[0] ==
            ((fig._axstack.as_list()[0].collections[0]._facecolors[:, 3]) != 0).sum())  # noqa

    # Save execution time and memory
    plt.close()


def test_plot_surf_stat_map_error():
    mesh = generate_surf()
    rng = np.random.RandomState(42)
    data = 10 * rng.standard_normal(size=mesh[0].shape[0])

    # Try to input vmin
    with pytest.raises(
            ValueError,
            match='this function does not accept a "vmin" argument'):
        plot_surf_stat_map(mesh, stat_map=data, vmin=0)

    # Wrong size of stat map data
    with pytest.raises(
            ValueError,
            match='surf_map does not have the same number of vertices'):
        plot_surf_stat_map(mesh, stat_map=np.hstack((data, data)))

    with pytest.raises(
            ValueError,
            match="'surf_map' can only have one dimension"):
        plot_surf_stat_map(mesh, stat_map=np.vstack((data, data)).T)


def _generate_data_test_surf_roi():
    mesh = generate_surf()
    rng = np.random.RandomState(42)
    roi_idx = rng.randint(0, mesh[0].shape[0], size=10)
    roi_map = np.zeros(mesh[0].shape[0])
    roi_map[roi_idx] = 1
    parcellation = rng.uniform(size=mesh[0].shape[0])
    return mesh, roi_map, parcellation


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_roi(engine):
    if not PLOTLY_INSTALLED and engine == "plotly":
        pytest.skip('Plotly is not installed; required for this test.')
    mesh, roi_map, parcellation = _generate_data_test_surf_roi()
    # plot roi
    plot_surf_roi(mesh, roi_map=roi_map, engine=engine)
    plot_surf_roi(mesh, roi_map=roi_map,
                  colorbar=True, engine=engine)
    # plot parcellation
    plot_surf_roi(mesh, roi_map=parcellation, engine=engine)
    plot_surf_roi(mesh, roi_map=parcellation, colorbar=True,
                  engine=engine)
    plot_surf_roi(mesh, roi_map=parcellation, colorbar=True,
                  cbar_tick_fomat="%f", engine=engine)
    plt.close()


def test_plot_surf_roi_matplotlib_specific():
    mesh, roi_map, parcellation = _generate_data_test_surf_roi()
    # change vmin, vmax
    img = plot_surf_roi(mesh, roi_map=roi_map, vmin=1.2,
                        vmax=8.9, colorbar=True,
                        engine='matplotlib')
    img.canvas.draw()
    cbar = img.axes[-1]
    cbar_vmin = float(cbar.get_yticklabels()[0].get_text())
    cbar_vmax = float(cbar.get_yticklabels()[-1].get_text())
    assert cbar_vmin == 1.0
    assert cbar_vmax == 8.0
    img2 = plot_surf_roi(mesh, roi_map=roi_map, vmin=1.2,
                         vmax=8.9, colorbar=True,
                         cbar_tick_format="%.2g",
                         engine='matplotlib')
    img2.canvas.draw()
    cbar = img2.axes[-1]
    cbar_vmin = float(cbar.get_yticklabels()[0].get_text())
    cbar_vmax = float(cbar.get_yticklabels()[-1].get_text())
    assert cbar_vmin == 1.2
    assert cbar_vmax == 8.9
    # plot to axes
    plot_surf_roi(mesh, roi_map=roi_map, ax=None,
                  figure=plt.gcf(), engine='matplotlib')

    # plot to axes
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(mesh, roi_map=roi_map, ax=plt.gca(),
                      figure=None, output_file=tmp_file.name,
                      engine='matplotlib')
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(mesh, roi_map=roi_map, ax=plt.gca(),
                      figure=None, output_file=tmp_file.name,
                      colorbar=True, engine='matplotlib')

    # Test nans handling
    parcellation[::2] = np.nan
    img = plot_surf_roi(mesh, roi_map=parcellation,
                        engine='matplotlib')
    # Check that the resulting plot facecolors contain no transparent faces
    # (last column equals zero) even though the texture contains nan values
    assert(mesh[1].shape[0] ==
           ((img._axstack.as_list()[0].collections[0]._facecolors[:, 3]) != 0).sum())
    # Save execution time and memory
    plt.close()


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_roi_error(engine):
    if not PLOTLY_INSTALLED and engine == "plotly":
        pytest.skip('Plotly is not installed; required for this test.')
    mesh = generate_surf()
    rng = np.random.RandomState(42)
    roi_idx = rng.randint(0, mesh[0].shape[0], size=5)
    with pytest.raises(
            ValueError,
            match='roi_map does not have the same number of vertices'):
        plot_surf_roi(mesh, roi_map=roi_idx, engine=engine)


def _generate_img():
    mni_affine = MNI152TEMPLATE.get_affine()
    data_positive = np.zeros((7, 7, 3))
    rng = np.random.RandomState(42)
    data_rng = rng.uniform(size=(7, 7, 3))
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
    nii = nibabel.Nifti1Image(data_positive, mni_affine)
    return nii


def test_plot_img_on_surf_hemispheres_and_orientations():
    nii = _generate_img()
    # Check that all combinations of 1D or 2D hemis and orientations work.
    plot_img_on_surf(nii, hemispheres=['right'], views=['lateral'])
    plot_img_on_surf(nii, hemispheres=['left', 'right'], views=['lateral'])
    plot_img_on_surf(nii,
                     hemispheres=['right'],
                     views=['medial', 'lateral'])
    plot_img_on_surf(nii,
                     hemispheres=['left', 'right'],
                     views=['dorsal', 'medial'])


def test_plot_img_on_surf_colorbar():
    nii = _generate_img()
    plot_img_on_surf(nii, hemispheres=['right'], views=['lateral'],
                     colorbar=True, vmax=5, threshold=3)
    plot_img_on_surf(nii, hemispheres=['right'], views=['lateral'],
                     colorbar=False)
    plot_img_on_surf(nii, hemispheres=['right'], views=['lateral'],
                     colorbar=False, cmap='roy_big_bl')
    plot_img_on_surf(nii, hemispheres=['right'], views=['lateral'],
                     colorbar=True, cmap='roy_big_bl', vmax=2)


def test_plot_img_on_surf_inflate():
    nii = _generate_img()
    plot_img_on_surf(nii, hemispheres=['right'], views=['lateral'],
                     inflate=True)


def test_plot_img_on_surf_surf_mesh():
    nii = _generate_img()
    plot_img_on_surf(nii, hemispheres=['right', 'left'], views=['lateral'])
    plot_img_on_surf(nii, hemispheres=['right', 'left'], views=['lateral'],
                     surf_mesh='fsaverage5')
    surf_mesh = fetch_surf_fsaverage()
    plot_img_on_surf(nii, hemispheres=['right', 'left'], views=['lateral'],
                     surf_mesh=surf_mesh)


def test_plot_img_on_surf_with_invalid_orientation():
    kwargs = {"hemisphere": ["right"], "inflate": True}
    nii = _generate_img()
    with pytest.raises(ValueError):
        plot_img_on_surf(nii, views=['latral'], **kwargs)
    with pytest.raises(ValueError):
        plot_img_on_surf(nii, views=['dorsal', 'post'], **kwargs)
    with pytest.raises(TypeError):
        plot_img_on_surf(nii, views=0, **kwargs)
    with pytest.raises(ValueError):
        plot_img_on_surf(nii, views=['medial', {'a': 'a'}], **kwargs)


def test_plot_img_on_surf_with_invalid_hemisphere():
    nii = _generate_img()
    with pytest.raises(ValueError):
        plot_img_on_surf(
            nii, views=['lateral'], inflate=True, hemispheres=["lft]"]
        )
    with pytest.raises(ValueError):
        plot_img_on_surf(
            nii, views=['medial'], inflate=True, hemispheres=['lef']
        )
    with pytest.raises(ValueError):
        plot_img_on_surf(
            nii,
            views=['anterior', 'posterior'],
            inflate=True,
            hemispheres=['left', 'right', 'middle']
        )


def test_plot_img_on_surf_with_figure_kwarg():
    nii = _generate_img()
    with pytest.raises(ValueError):
        plot_img_on_surf(
            nii,
            views=["anterior"],
            hemispheres=["right"],
            figure=True,
        )


def test_plot_img_on_surf_with_axes_kwarg():
    nii = _generate_img()
    with pytest.raises(ValueError):
        plot_img_on_surf(
            nii,
            views=["anterior"],
            hemispheres=["right"],
            inflat=True,
            axes="something",
        )


def test_plot_img_on_surf_title():
    nii = _generate_img()
    title = "Title"
    fig, axes = plot_img_on_surf(
        nii, hemispheres=['right'], views=['lateral']
    )
    assert fig._suptitle is None, "Created title without title kwarg."
    fig, axes = plot_img_on_surf(
        nii, hemispheres=['right'], views=['lateral'], title=title
    )
    assert fig._suptitle is not None, "Title not created."
    assert fig._suptitle.get_text() == title, "Title text not assigned."


def test_plot_img_on_surf_output_file(tmp_path):
    nii = _generate_img()
    fname = tmp_path / 'tmp.png'
    return_value = plot_img_on_surf(nii,
                                    hemispheres=['right'],
                                    views=['lateral'],
                                    output_file=str(fname))
    assert return_value is None, "Returned figure and axes on file output."
    assert fname.is_file(), "Saved image file could not be found."


def test_plot_surf_contours():
    mesh = generate_surf()
    # we need a valid parcellation for testing
    parcellation = np.zeros((mesh[0].shape[0],))
    parcellation[mesh[1][3]] = 1
    parcellation[mesh[1][5]] = 2
    plot_surf_contours(mesh, parcellation)
    plot_surf_contours(mesh, parcellation, levels=[1, 2])
    plot_surf_contours(mesh, parcellation, levels=[1, 2], cmap='gist_ncar')
    plot_surf_contours(mesh, parcellation, levels=[1, 2],
                       colors=['r', 'g'])
    plot_surf_contours(mesh, parcellation, levels=[1, 2], colors=['r', 'g'],
                       labels=['1', '2'])
    fig = plot_surf_contours(mesh, parcellation, levels=[1, 2], colors=['r', 'g'],
                             labels=['1', '2'], legend=True)
    assert fig.legends is not None
    plot_surf_contours(mesh, parcellation, levels=[1, 2],
                       colors=[[0, 0, 0, 1], [1, 1, 1, 1]])
    fig, axes = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    plot_surf_contours(mesh, parcellation, axes=axes)
    plot_surf_contours(mesh, parcellation, figure=fig)
    fig = plot_surf(mesh)
    plot_surf_contours(mesh, parcellation, figure=fig)
    display = plot_surf_contours(mesh, parcellation, levels=[1, 2],
                                 labels=['1', '2'], colors=['r', 'g'],
                                 legend=True, title='title',
                                 figure=fig)
    # Non-regression assertion: we switched from _suptitle to axis title
    assert display._suptitle is None
    assert display.axes[0].get_title() == "title"
    fig = plot_surf(mesh, title='title 2')
    display = plot_surf_contours(mesh, parcellation, levels=[1, 2],
                                 labels=['1', '2'], colors=['r', 'g'],
                                 legend=True, figure=fig)
    # Non-regression assertion: we switched from _suptitle to axis title
    assert display._suptitle is None
    assert display.axes[0].get_title() == "title 2"
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_contours(mesh, parcellation, output_file=tmp_file.name)
    plt.close()


def test_plot_surf_contours_error():
    mesh = generate_surf()
    # we need an invalid parcellation for testing
    rng = np.random.RandomState(42)
    invalid_parcellation = rng.uniform(size=(mesh[0].shape[0]))
    parcellation = np.zeros((mesh[0].shape[0],))
    parcellation[mesh[1][3]] = 1
    parcellation[mesh[1][5]] = 2
    with pytest.raises(
            ValueError,
            match='Vertices in parcellation do not form region.'):
        plot_surf_contours(mesh, invalid_parcellation)
    fig, axes = plt.subplots(1, 1)
    with pytest.raises(
            ValueError,
            match='Axes must be 3D.'):
        plot_surf_contours(mesh, parcellation, axes=axes)
    with pytest.raises(
            ValueError,
            match='All elements of colors need to be either a matplotlib color string or RGBA values.'):
        plot_surf_contours(mesh, parcellation, levels=[1, 2], colors=[[1, 2], 3])
    with pytest.raises(
            ValueError,
            match='Levels, labels, and colors argument need to be either the same length or None.'):
        plot_surf_contours(mesh, parcellation, levels=[1, 2], colors=['r'], labels=['1', '2'])


@pytest.mark.parametrize("vmin,vmax,cbar_tick_format,expected", [
    (0, 0, "%i", [0]),
    (0, 3, "%i", [0, 1, 2, 3]),
    (0, 4, "%i", [0, 1, 2, 3, 4]),
    (1, 5, "%i", [1, 2, 3, 4, 5]),
    (0, 5, "%i", [0, 1.25, 2.5, 3.75, 5]),
    (0, 10, "%i", [0, 2.5, 5, 7.5, 10]),
    (0, 0, "%.1f", [0]),
    (0, 1, "%.1f", [0, 0.25, 0.5, 0.75, 1]),
    (1, 2, "%.1f", [1, 1.25, 1.5, 1.75, 2]),
    (1.1, 1.2, "%.1f", [1.1, 1.125, 1.15, 1.175, 1.2]),
    (0, np.nextafter(0, 1), "%.1f", [0.e+000, 5.e-324]),
])
def test_get_ticks_matplotlib(vmin, vmax, cbar_tick_format, expected):
    ticks = _get_ticks_matplotlib(vmin, vmax, cbar_tick_format)
    assert 1 <= len(ticks) <= 5
    assert ticks[0] == vmin and ticks[-1] == vmax
    assert len(ticks) == len(expected) and (ticks == expected).all()


def test_compute_facecolors_matplotlib():
    fsaverage = fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage['pial_left'])
    alpha = "auto"
    # Surface map whose value in each vertex is
    # 1 if this vertex's curv > 0
    # 0 if this vertex's curv is 0
    # -1 if this vertex's curv < 0
    bg_map = np.sign(load_surf_data(fsaverage['curv_left']))
    bg_min, bg_max = np.min(bg_map), np.max(bg_map)
    assert (bg_min < 0 or bg_max > 1)
    facecolors_auto_normalized = _compute_facecolors_matplotlib(
        bg_map,
        mesh[1],
        len(mesh[0]),
        None,
        alpha,
    )
    assert len(facecolors_auto_normalized) == len(mesh[1])

    # Manually set values of background map between 0 and 1
    bg_map_normalized = (bg_map - bg_min) / (bg_max - bg_min)
    assert np.min(bg_map_normalized) == 0 and np.max(bg_map_normalized) == 1
    facecolors_manually_normalized = _compute_facecolors_matplotlib(
        bg_map_normalized,
        mesh[1],
        len(mesh[0]),
        None,
        alpha,
    )
    assert len(facecolors_manually_normalized) == len(mesh[1])
    assert np.allclose(
        facecolors_manually_normalized, facecolors_auto_normalized
    )

    # Scale background map between 0.25 and 0.75
    bg_map_scaled = bg_map_normalized / 2 + 0.25
    assert np.min(bg_map_scaled) == 0.25 and np.max(bg_map_scaled) == 0.75
    facecolors_manually_rescaled = _compute_facecolors_matplotlib(
        bg_map_scaled,
        mesh[1],
        len(mesh[0]),
        None,
        alpha,
    )
    assert len(facecolors_manually_rescaled) == len(mesh[1])
    assert not np.allclose(
        facecolors_manually_rescaled, facecolors_auto_normalized
    )
