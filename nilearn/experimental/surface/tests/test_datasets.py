import pytest

from nilearn.experimental.surface import fetch_destrieux, fetch_nki


def test_dfetch_datasets_errors():
    """Give incorrect value argument."""
    with pytest.raises(ValueError, match="'mesh_type' must be one of"):
        fetch_nki(mesh_type="foo")
        fetch_destrieux(mesh_type="foo")
