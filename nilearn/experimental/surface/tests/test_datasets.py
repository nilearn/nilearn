from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from nibabel import gifti

from nilearn.experimental.surface import (
    SurfaceImage,
    fetch_destrieux,
    fetch_nki,
    load_fsaverage,
    load_fsaverage_data,
)


def test_load_fsaverage():
    """Call default function smoke test and assert return."""
    meshes = load_fsaverage()
    assert isinstance(meshes, dict)
    assert meshes["pial"]["left"].n_vertices == 10242  # fsaverage5


def test_load_load_fsaverage_data():
    """Call default function smoke test and assert return."""
    img = load_fsaverage_data()
    assert isinstance(img, SurfaceImage)


def test_load_load_fsaverage_data_errors():
    """Check wrong parameter values."""
    with pytest.raises(ValueError, match="'mesh_type' must be one of"):
        load_fsaverage_data(mesh_type="foo")
    with pytest.raises(ValueError, match="'data_type' must be one of"):
        load_fsaverage_data(data_type="foo")


def test_load_fsaverage_wrong_mesh_name():
    """Give incorrect value to mesh_name argument."""
    with pytest.raises(ValueError, match="'mesh' should be one of"):
        load_fsaverage(mesh_name="foo")


def test_load_fsaverage_hemispheres_have_file():
    """Make sure file paths are present."""
    meshes = load_fsaverage()
    left_hemisphere_meshes = [
        mesh for mesh in meshes.values() if "left" in mesh
    ]
    assert left_hemisphere_meshes
    right_hemisphere_meshes = [
        mesh for mesh in meshes.values() if "right" in mesh
    ]
    assert right_hemisphere_meshes


def test_destrieux_nki_wrong_mesh_type():
    with pytest.raises(ValueError, match="'mesh_type' must be one of"):
        fetch_nki(mesh_type="foo")

    with pytest.raises(ValueError, match="'mesh_type' must be one of"):
        fetch_destrieux(mesh_type="foo")


def test_smoke_nki(request_mocker):
    # TODO mocking taken from test_fetch_surf_nki_enhanced
    #  in nilearn/datasets/tests/test_func.py
    ids = np.asarray(
        [
            "A00028185",
            "A00035827",
            "A00037511",
            "A00039431",
            "A00033747",
            "A00035840",
            "A00038998",
            "A00035072",
            "A00037112",
            "A00039391",
        ],
        dtype="U9",
    )
    age = np.ones(len(ids), dtype="<f8")
    hand = np.asarray(len(ids) * ["x"], dtype="U1")
    sex = np.asarray(len(ids) * ["x"], dtype="U1")
    pheno_data = pd.DataFrame(
        OrderedDict([("id", ids), ("age", age), ("hand", hand), ("sex", sex)])
    )
    request_mocker.url_mapping["*pheno_nki_nilearn.csv"] = pheno_data.to_csv(
        index=False
    )

    # added part mocking gifti
    darray = gifti.GiftiDataArray(
        data=np.zeros((20,)), datatype="NIFTI_TYPE_FLOAT32"
    )
    gii = gifti.GiftiImage(darrays=[darray])
    request_mocker.url_mapping["*.gii"] = gii
