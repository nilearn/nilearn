import pytest

from nilearn.experimental.surface import load_fsaverage


def test_load_fsaverage():
    """Call default function smoke test."""
    result = load_fsaverage()
    assert isinstance(result, dict)


@pytest.mark.parametrize(
    "mesh_name,vertices", [("fsaverage3", 642), ("fsaverage4", 2562)]
)
def test_load_fsaverage_with_different_resolutions(mesh_name, vertices):
    """Parameterize mesh_name argument."""
    result = load_fsaverage(mesh_name)
    assert result["pial"]["left_hemisphere"].n_vertices == vertices


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
