from pathlib import Path

import numpy as np
import pytest

from numpy.testing import assert_array_equal

from nilearn.plotting.surface._utils import check_surface_plotting_inputs
from nilearn.surface import InMemoryMesh
from nilearn.surface._testing import assert_surface_mesh_equal


@pytest.mark.parametrize("bg_map", ["some_path", Path("some_path"), None])
@pytest.mark.parametrize("surf_map", ["some_path", Path("some_path")])
@pytest.mark.parametrize("surf_mesh", ["some_path", Path("some_path")])
def test_check_surface_plotting_inputs_no_change(surf_map, surf_mesh, bg_map):
    """Cover use cases where the inputs are not changed."""
    hemi = "left"
    out_surf_map, out_surf_mesh, out_bg_map = check_surface_plotting_inputs(
        surf_map, surf_mesh, hemi, bg_map
    )
    assert surf_map == out_surf_map
    assert surf_mesh == out_surf_mesh
    assert bg_map == out_bg_map


@pytest.mark.parametrize("bg_map", ["some_path", Path("some_path"), None])
@pytest.mark.parametrize("mesh", [None])
def test_check_surface_plotting_inputs_extract_mesh_and_data(
    surf_img_1d, mesh, bg_map
):
    """Extract mesh and data when a SurfaceImage is passed."""
    hemi = "left"
    out_surf_map, out_surf_mesh, out_bg_map = check_surface_plotting_inputs(
        surf_map=surf_img_1d,
        surf_mesh=mesh,
        hemi=hemi,
        bg_map=bg_map,
    )

    assert_array_equal(out_surf_map, surf_img_1d.data.parts[hemi].T)
    assert_surface_mesh_equal(out_surf_mesh, surf_img_1d.mesh.parts[hemi])

    assert bg_map == out_bg_map


def test_check_surface_plotting_inputs_many_time_points(
    surf_img_1d, surf_img_2d
):
    """Extract mesh and data when a SurfaceImage is passed."""
    with pytest.raises(
        TypeError, match="Input data has incompatible dimensionality"
    ):
        check_surface_plotting_inputs(
            surf_map=surf_img_2d(10),
            surf_mesh=None,
            hemi="left",
            bg_map=None,
        )

    with pytest.raises(
        TypeError, match="Input data has incompatible dimensionality"
    ):
        check_surface_plotting_inputs(
            surf_map=surf_img_1d,
            surf_mesh=None,
            hemi="left",
            bg_map=surf_img_2d(10),
        )


@pytest.mark.parametrize("bg_map", ["some_path", Path("some_path"), None])
def test_check_surface_plotting_inputs_extract_mesh_from_polymesh(
    surf_img_1d, surf_mesh, bg_map
):
    """Extract mesh from Polymesh and data from SurfaceImage."""
    hemi = "left"
    out_surf_map, out_surf_mesh, out_bg_map = check_surface_plotting_inputs(
        surf_map=surf_img_1d,
        surf_mesh=surf_mesh,
        hemi=hemi,
        bg_map=bg_map,
    )
    assert_array_equal(out_surf_map, surf_img_1d.data.parts[hemi].T)
    assert_surface_mesh_equal(out_surf_mesh, surf_mesh.parts[hemi])
    assert bg_map == out_bg_map


def test_check_surface_plotting_inputs_extract_bg_map_data(
    surf_img_1d, surf_mesh
):
    """Extract background map data."""
    hemi = "left"
    _, _, out_bg_map = check_surface_plotting_inputs(
        surf_map=surf_img_1d,
        surf_mesh=surf_mesh,
        hemi=hemi,
        bg_map=surf_img_1d,
    )
    assert_array_equal(out_bg_map, surf_img_1d.data.parts[hemi])


def test_check_surface_plotting_inputs_error_mash_and_data_none():
    """Fail if no mesh or data is passed."""
    with pytest.raises(TypeError, match="cannot both be None"):
        check_surface_plotting_inputs(None, None)


def test_check_surface_plotting_inputs_errors(surf_img_1d, surf_mesh):
    """Fail if mesh is none and data is not not SurfaceImage."""
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        check_surface_plotting_inputs(surf_map=1, surf_mesh=None)
    with pytest.raises(
        TypeError, match="'surf_mesh' cannot be a SurfaceImage instance."
    ):
        check_surface_plotting_inputs(
            surf_map=surf_img_1d, surf_mesh=surf_img_1d
        )


def test_check_surface_plotting_hemi_both_all_inputs(surf_img_1d, surf_mesh):
    """Test that hemi="both" works as expected when all inputs are provided."""
    hemi = "both"
    combined_map, combined_mesh, combined_bg = check_surface_plotting_inputs(
        surf_map=surf_img_1d,
        surf_mesh=surf_mesh,
        hemi=hemi,
        bg_map=surf_img_1d,
    )
    # check that the data is concatenated
    for data in [combined_map, combined_bg]:
        assert_array_equal(
            data,
            np.concatenate(
                (
                    surf_img_1d.data.parts["left"],
                    surf_img_1d.data.parts["right"],
                )
            ),
        )
        assert isinstance(data, np.ndarray)
    # check that the mesh is concatenated
    assert combined_mesh.n_vertices == surf_mesh.n_vertices
    assert isinstance(combined_mesh, InMemoryMesh)


def test_check_surface_plotting_hemi_both_mesh_none(surf_img_1d):
    """Test that hemi="both" works as expected when mesh is not provided."""
    hemi = "both"
    combined_map, combined_mesh, combined_bg = check_surface_plotting_inputs(
        surf_map=surf_img_1d,
        surf_mesh=None,
        hemi=hemi,
    )
    # check that the mesh is taken from surf_map
    assert combined_mesh.n_vertices == surf_img_1d.mesh.n_vertices
    assert isinstance(combined_mesh, InMemoryMesh)


def test_check_surface_plotting_hemi_error(surf_img_1d, surf_mesh):
    """Test that an error is raised when hemi is not valid."""
    with pytest.raises(
        ValueError, match="hemi must be one of left, right or both"
    ):
        check_surface_plotting_inputs(
            surf_map=surf_img_1d, surf_mesh=surf_mesh, hemi="foo"
        )
