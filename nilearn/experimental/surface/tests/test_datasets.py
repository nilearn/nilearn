import pytest

from nilearn.experimental.surface import load_fsaverage


def test_load_fsaverage():
    """Call default function smoke test and assert return."""
    result = load_fsaverage()
    assert isinstance(result, dict)
    assert result["pial"]["left_hemisphere"].n_vertices == 10242  # fsaverage5


def test_load_fsaverage_wrong_mesh_name():
    """Give incorrect value to mesh_name argument."""
    with pytest.raises(ValueError, match="'mesh' should be one of"):
        load_fsaverage(mesh_name="foo")


def test_load_fsaverage_hemispheres_have_file():
    """Make sure file paths are present."""
    result = load_fsaverage()
    left_hemisphere_meshes = [
        mesh for mesh in result.values() if "left_hemisphere" in mesh
    ]
    assert len(left_hemisphere_meshes) > 0
    right_hemisphere_meshes = [
        mesh for mesh in result.values() if "right_hemisphere" in mesh
    ]
    assert len(right_hemisphere_meshes) > 0
