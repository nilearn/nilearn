# Tests for functions in surf_plotting.py
import numpy as np
import nibabel
import matplotlib.pyplot as plt
import pytest
import tempfile

from nilearn.plotting.img_plotting import MNI152TEMPLATE
from nilearn.plotting.surf_plotting import (plot_surf, plot_surf_stat_map,
                                            plot_surf_roi, plot_img_on_surf)
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface.testing_utils import generate_surf
from pathlib import Path


def test_plot_surf():
    mesh = generate_surf()
    rng = np.random.RandomState(0)
    bg = rng.randn(mesh[0].shape[0], )

    # Plot mesh only
    plot_surf(mesh)

    # Plot mesh with background
    plot_surf(mesh, bg_map=bg)
    plot_surf(mesh, bg_map=bg, darkness=0.5)
    plot_surf(mesh, bg_map=bg, alpha=0.5)

    # Plot different views
    plot_surf(mesh, bg_map=bg, hemi='right')
    plot_surf(mesh, bg_map=bg, view='medial')
    plot_surf(mesh, bg_map=bg, hemi='right', view='medial')

    # Plot with colorbar
    plot_surf(mesh, bg_map=bg, colorbar=True)

    # Save execution time and memory
    plt.close()


def test_plot_surf_error():
    mesh = generate_surf()
    rng = np.random.RandomState(0)

    # Wrong inputs for view or hemi
    with pytest.raises(ValueError, match='view must be one of'):
        plot_surf(mesh, view='middle')
    with pytest.raises(ValueError, match='hemi must be one of'):
        plot_surf(mesh, hemi='lft')

    # Wrong size of background image
    with pytest.raises(
            ValueError,
            match='bg_map does not have the same number of vertices'):
        plot_surf(mesh, bg_map=rng.randn(mesh[0].shape[0] - 1, ))

    # Wrong size of surface data
    with pytest.raises(
            ValueError,
            match='surf_map does not have the same number of vertices'):
        plot_surf(mesh, surf_map=rng.randn(mesh[0].shape[0] + 1, ))

    with pytest.raises(
            ValueError,
            match='surf_map can only have one dimension'):
        plot_surf(mesh, surf_map=rng.randn(mesh[0].shape[0], 2))


def test_plot_surf_stat_map():
    mesh = generate_surf()
    rng = np.random.RandomState(0)
    bg = rng.randn(mesh[0].shape[0], )
    data = 10 * rng.randn(mesh[0].shape[0], )

    # Plot mesh with stat map
    plot_surf_stat_map(mesh, stat_map=data)
    plot_surf_stat_map(mesh, stat_map=data, colorbar=True)
    plot_surf_stat_map(mesh, stat_map=data, alpha=1)

    # Plot mesh with background and stat map
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg)
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg,
                       bg_on_data=True, darkness=0.5)
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg, colorbar=True,
                       bg_on_data=True, darkness=0.5)

    # Apply threshold
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg,
                       bg_on_data=True, darkness=0.5,
                       threshold=0.3)
    plot_surf_stat_map(mesh, stat_map=data, bg_map=bg, colorbar=True,
                       bg_on_data=True, darkness=0.5,
                       threshold=0.3)

    # Change vmax
    plot_surf_stat_map(mesh, stat_map=data, vmax=5)
    plot_surf_stat_map(mesh, stat_map=data, vmax=5, colorbar=True)

    # Change colormap
    plot_surf_stat_map(mesh, stat_map=data, cmap='cubehelix')
    plot_surf_stat_map(mesh, stat_map=data, cmap='cubehelix', colorbar=True)

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
    # Save execution time and memory
    plt.close()


def test_plot_surf_stat_map_error():
    mesh = generate_surf()
    rng = np.random.RandomState(0)
    data = 10 * rng.randn(mesh[0].shape[0], )

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
            match='surf_map can only have one dimension'):
        plot_surf_stat_map(mesh, stat_map=np.vstack((data, data)).T)


def test_plot_surf_roi():
    mesh = generate_surf()
    rng = np.random.RandomState(0)
    roi_idx = rng.randint(0, mesh[0].shape[0], size=10)
    roi_map = np.zeros(mesh[0].shape[0])
    roi_map[roi_idx] = 1
    parcellation = rng.rand(mesh[0].shape[0])

    # plot roi
    plot_surf_roi(mesh, roi_map=roi_map)
    plot_surf_roi(mesh, roi_map=roi_map, colorbar=True)
    # change vmin, vmax
    img = plot_surf_roi(mesh, roi_map=roi_map, vmin=1.2,
                        vmax=8.9, colorbar=True)
    img.canvas.draw()
    cbar = img.axes[-1]
    cbar_vmin = float(cbar.get_yticklabels()[0].get_text())
    cbar_vmax = float(cbar.get_yticklabels()[-1].get_text())
    assert cbar_vmin == 1.2
    assert cbar_vmax == 8.9

    # plot parcellation
    plot_surf_roi(mesh, roi_map=parcellation)
    plot_surf_roi(mesh, roi_map=parcellation, colorbar=True)

    # plot to axes
    plot_surf_roi(mesh, roi_map=roi_map, ax=None, figure=plt.gcf())

    # plot to axes
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(mesh, roi_map=roi_map, ax=plt.gca(), figure=None,
                      output_file=tmp_file.name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(mesh, roi_map=roi_map, ax=plt.gca(), figure=None,
                      output_file=tmp_file.name, colorbar=True)

    # Save execution time and memory
    plt.close()


def test_plot_surf_roi_error():
    mesh = generate_surf()
    rng = np.random.RandomState(0)
    roi_idx = rng.randint(0, mesh[0].shape[0], size=5)
    with pytest.raises(
            ValueError,
            match='roi_map does not have the same number of vertices'):
        plot_surf_roi(mesh, roi_map=roi_idx)


def _generate_img():
    mni_affine = MNI152TEMPLATE.get_affine()
    data_positive = np.zeros((7, 7, 3))
    rng = np.random.RandomState(42)
    data_rng = rng.rand(7, 7, 3)
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
    plot_img_on_surf(nii, hemispheres=['right', 'left'], views=['lateral'],
                     surf_mesh=None)
    plot_img_on_surf(nii, hemispheres=['right', 'left'], views=['lateral'],
                     surf_mesh='fsaverage')
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


def test_plot_img_on_surf_output_file():
    nii = _generate_img()
    fname = Path('check.png').absolute()
    return_value = plot_img_on_surf(
        nii, hemispheres=['right'], views=['lateral'], output_file=str(fname)
    )
    assert return_value is None, "Returned figure and axes on file output."
    assert fname.is_file(), "Saved image file could not be found."
    fname.unlink()
