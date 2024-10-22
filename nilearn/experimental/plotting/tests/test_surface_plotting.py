from pathlib import Path

import pytest
from numpy.testing import assert_array_equal

from nilearn.experimental.plotting._surface_plotting import _check_inputs


@pytest.mark.parametrize("bg_map", ["some_path", Path("some_path"), None])
@pytest.mark.parametrize("surf_map", ["some_path", Path("some_path")])
@pytest.mark.parametrize("surf_mesh", ["some_path", Path("some_path")])
def test_check_inputs_no_change(surf_map, surf_mesh, bg_map):
    """Cover use cases where the inputs are not changed."""
    hemi = "left"
    out_surf_map, out_surf_mesh, out_bg_map = _check_inputs(
        surf_map, surf_mesh, hemi, bg_map
    )
    assert surf_map == out_surf_map
    assert surf_mesh == out_surf_mesh
    assert bg_map == out_bg_map


@pytest.mark.parametrize("bg_map", ["some_path", Path("some_path"), None])
@pytest.mark.parametrize("mesh", [None])
def test_check_inputs_extract_mesh_and_data(
    surf_img, mesh, bg_map, assert_surf_mesh_equal
):
    """Extract mesh and data when a SurfaceImage is passed."""
    hemi = "left"
    out_surf_map, out_surf_mesh, out_bg_map = _check_inputs(
        surf_map=surf_img((10,)),
        surf_mesh=mesh,
        hemi=hemi,
        bg_map=bg_map,
    )
    assert_array_equal(out_surf_map, surf_img((10,)).data.parts[hemi])
    assert_surf_mesh_equal(out_surf_mesh, surf_img((10,)).mesh.parts[hemi])
    assert bg_map == out_bg_map


@pytest.mark.parametrize("bg_map", ["some_path", Path("some_path"), None])
def test_check_inputs_extract_mesh_from_polymesh(
    surf_img, surf_mesh, bg_map, assert_surf_mesh_equal
):
    """Extract mesh from Polymesh and data from SurfaceImage."""
    hemi = "left"
    out_surf_map, out_surf_mesh, out_bg_map = _check_inputs(
        surf_map=surf_img((10,)),
        surf_mesh=surf_mesh(),
        hemi=hemi,
        bg_map=bg_map,
    )
    assert_array_equal(out_surf_map, surf_img((10,)).data.parts[hemi])
    assert_surf_mesh_equal(out_surf_mesh, surf_mesh().parts[hemi])
    assert bg_map == out_bg_map


def test_check_inputs_extract_bg_map_data(surf_img, surf_mesh):
    """Extract background map data."""
    hemi = "left"
    _, _, out_bg_map = _check_inputs(
        surf_map=surf_img((10,)),
        surf_mesh=surf_mesh(),
        hemi=hemi,
        bg_map=surf_img(),
    )
    assert_array_equal(out_bg_map, surf_img().data.parts[hemi])
