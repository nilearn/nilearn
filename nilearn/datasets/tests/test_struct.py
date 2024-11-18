"""Test the datasets module."""

# Authors: Alexandre Abraham, Ana Luisa Pinho

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image
from sklearn.utils import Bunch

from nilearn.datasets import struct
from nilearn.datasets.struct import (
    fetch_surf_fsaverage,
    load_fsaverage,
    load_fsaverage_data,
)
from nilearn.datasets.tests._testing import dict_to_archive, list_to_archive
from nilearn.surface import PolyMesh, SurfaceImage


def test_fetch_icbm152_2009(tmp_path, request_mocker):
    dataset = struct.fetch_icbm152_2009(data_dir=str(tmp_path), verbose=0)

    assert isinstance(dataset.csf, str)
    assert isinstance(dataset.eye_mask, str)
    assert isinstance(dataset.face_mask, str)
    assert isinstance(dataset.gm, str)
    assert isinstance(dataset.mask, str)
    assert isinstance(dataset.pd, str)
    assert isinstance(dataset.t1, str)
    assert isinstance(dataset.t2, str)
    assert isinstance(dataset.t2_relax, str)
    assert isinstance(dataset.wm, str)
    assert request_mocker.url_count == 1
    assert dataset.description != ""


def _make_oasis_data(dartel=True):
    n_subjects = 457
    prefix = "mwrc" if dartel else "mwc"
    ids = pd.DataFrame(
        {"ID": [f"OAS1_{i:04}" for i in range(n_subjects)]}
    ).to_csv(index=False, sep="\t")
    data = {"oasis_cross-sectional.csv": ids, "data_usage_agreement.txt": ""}
    path_pattern = str(
        Path(
            "OAS1_{subj:04}_MR1",
            "{prefix}{kind}OAS1_{subj:04}_MR1_mpr_anon_fslswapdim_bet.nii.gz",
        )
    )
    for i in range(457):
        for kind in [1, 2]:
            data[path_pattern.format(subj=i, kind=kind, prefix=prefix)] = ""
    return dict_to_archive(data)


@pytest.mark.parametrize("legacy_format", [True, False])
def test_fetch_oasis_vbm(tmp_path, request_mocker, legacy_format):
    request_mocker.url_mapping["*archive_dartel.tgz*"] = _make_oasis_data()
    request_mocker.url_mapping["*archive.tgz*"] = _make_oasis_data(False)

    dataset = struct.fetch_oasis_vbm(
        data_dir=str(tmp_path), verbose=0, legacy_format=legacy_format
    )

    assert len(dataset.gray_matter_maps) == 403
    assert len(dataset.white_matter_maps) == 403
    assert isinstance(dataset.gray_matter_maps[0], str)
    assert isinstance(dataset.white_matter_maps[0], str)
    if legacy_format:
        assert isinstance(dataset.ext_vars, np.recarray)
    else:
        assert isinstance(dataset.ext_vars, pd.DataFrame)
    assert isinstance(dataset.data_usage_agreement, str)
    assert request_mocker.url_count == 1

    dataset = struct.fetch_oasis_vbm(
        data_dir=str(tmp_path),
        dartel_version=False,
        verbose=0,
        legacy_format=legacy_format,
    )

    assert len(dataset.gray_matter_maps) == 415
    assert len(dataset.white_matter_maps) == 415
    assert isinstance(dataset.gray_matter_maps[0], str)
    assert isinstance(dataset.white_matter_maps[0], str)
    if legacy_format:
        assert isinstance(dataset.ext_vars, np.recarray)
    else:
        assert isinstance(dataset.ext_vars, pd.DataFrame)
    assert isinstance(dataset.data_usage_agreement, str)
    assert request_mocker.url_count == 2
    assert dataset.description != ""


@pytest.mark.parametrize(
    "func",
    [
        struct.load_mni152_brain_mask,
        struct.load_mni152_gm_mask,
        struct.load_mni152_gm_template,
        struct.load_mni152_template,
        struct.load_mni152_wm_mask,
        struct.load_mni152_wm_template,
    ],
)
@pytest.mark.parametrize("resolution", [None, 2])
def test_load_mni152(func, resolution):
    img = func(resolution=resolution)

    assert isinstance(img, Nifti1Image)

    if resolution is None:
        expected_shape = (197, 233, 189)
        expected_zooms = (1.0, 1.0, 1.0)
    elif resolution == 2:
        expected_shape = (99, 117, 95)
        expected_zooms = (2.0, 2.0, 2.0)

    assert img.shape == expected_shape
    assert img.header.get_zooms() == expected_zooms


def test_fetch_icbm152_brain_gm_mask(tmp_path):
    dataset = struct.fetch_icbm152_2009(data_dir=str(tmp_path), verbose=0)
    struct.load_mni152_template(resolution=2).to_filename(dataset.gm)
    grey_matter_img = struct.fetch_icbm152_brain_gm_mask(
        data_dir=str(tmp_path), verbose=0
    )

    assert isinstance(grey_matter_img, Nifti1Image)


@pytest.mark.parametrize(
    "mesh",
    [
        "fsaverage3",
        "fsaverage4",
        "fsaverage5",
        "fsaverage6",
        "fsaverage7",
        "fsaverage",
    ],
)
def test_fetch_surf_fsaverage(mesh, tmp_path, request_mocker):
    # Define attribute list that nilearn meshs should contain
    # (each attribute should eventually map to a _.gii.gz file
    # named after the attribute)
    mesh_attributes = {
        f"{part}_{side}"
        for part in [
            "area",
            "curv",
            "flat",
            "infl",
            "pial",
            "sphere",
            "sulc",
            "thick",
            "white",
        ]
        for side in ["left", "right"]
    }

    # Mock fsaverage3, 4, 6, 7 download (with actual url)
    fs_urls = [
        "https://osf.io/azhdf/download",
        "https://osf.io/28uma/download",
        "https://osf.io/jzxyr/download",
        "https://osf.io/svf8k/download",
    ]
    for fs_url in fs_urls:
        request_mocker.url_mapping[fs_url] = list_to_archive(
            [f"{name}.gii.gz" for name in mesh_attributes]
        )

    dataset = fetch_surf_fsaverage(mesh, data_dir=str(tmp_path))

    assert mesh_attributes.issubset(set(dataset.keys()))
    assert dataset.description != ""


def test_fetch_load_fsaverage():
    """Check that PolyMesh are returned."""
    result = load_fsaverage()
    assert isinstance(result, Bunch)
    assert isinstance(result.pial, PolyMesh)
    nb_vertices_fsaverage5 = 10242
    assert result["pial"].parts["left"].n_vertices == nb_vertices_fsaverage5
    assert result.pial.parts["left"].n_vertices == nb_vertices_fsaverage5


def test_load_fsaverage_data_smoke():
    assert isinstance(load_fsaverage_data(), SurfaceImage)


def test_load_fsaverage_data_errors():
    """Give incorrect value argument."""
    with pytest.raises(ValueError, match="'mesh_type' must be one of"):
        load_fsaverage_data(mesh_type="foo")
    with pytest.raises(ValueError, match="'data_type' must be one of"):
        load_fsaverage_data(data_type="foo")
