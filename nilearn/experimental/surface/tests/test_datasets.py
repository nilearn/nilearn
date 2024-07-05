import pytest

from nilearn.experimental.surface import (
    SurfaceImage,
    fetch_destrieux,
    fetch_nki,
    load_fsaverage,
    load_fsaverage_data,
)


def test_load_fsaverage():
    """Call default function smoke test and assert return."""
    result = load_fsaverage()
    assert isinstance(result, dict)
    assert result["pial"].parts["left"].n_vertices == 10242  # fsaverage5


def test_load_fsaverage_errors():
    """Give incorrect value argument."""
    with pytest.raises(ValueError, match="'mesh' should be one of"):
        load_fsaverage(mesh_name="foo")


def test_load_fsaverage_data_smoke():
    assert isinstance(load_fsaverage_data(), SurfaceImage)


def test_load_fsaverage_data_errors():
    """Give incorrect value argument."""
    with pytest.raises(ValueError, match="'mesh_type' must be one of"):
        load_fsaverage_data(mesh_type="foo")
    with pytest.raises(ValueError, match="'data_type' must be one of"):
        load_fsaverage_data(data_type="foo")


def test_load_fsaverage_hemispheres_have_file():
    """Make sure file paths are present."""
    result = load_fsaverage()
    left_hemisphere_meshes = [
        mesh for mesh in result.values() if "left" in mesh.parts
    ]
    assert left_hemisphere_meshes
    right_hemisphere_meshes = [
        mesh for mesh in result.values() if "right" in mesh.parts
    ]
    assert right_hemisphere_meshes


def test_dfetch_datasets_errors():
    """Give incorrect value argument."""
    with pytest.raises(ValueError, match="'mesh_type' must be one of"):
        fetch_nki(mesh_type="foo")
        fetch_destrieux(mesh_type="foo")
