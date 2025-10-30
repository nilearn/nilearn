"""Test the datasets module."""

import itertools
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Header, Nifti1Image, freesurfer, load, nifti1
from numpy.testing import assert_array_equal
from sklearn.utils import Bunch

from nilearn._utils import data_gen
from nilearn._utils.testing import serialize_niimg
from nilearn.conftest import _rng
from nilearn.datasets import atlas
from nilearn.datasets._utils import fetch_files
from nilearn.datasets.atlas import (
    fetch_atlas_aal,
    fetch_atlas_allen_2011,
    fetch_atlas_basc_multiscale_2015,
    fetch_atlas_craddock_2012,
    fetch_atlas_destrieux_2009,
    fetch_atlas_difumo,
    fetch_atlas_harvard_oxford,
    fetch_atlas_juelich,
    fetch_atlas_msdl,
    fetch_atlas_pauli_2017,
    fetch_atlas_schaefer_2018,
    fetch_atlas_smith_2009,
    fetch_atlas_surf_destrieux,
    fetch_atlas_talairach,
    fetch_atlas_yeo_2011,
)
from nilearn.datasets.tests._testing import (
    check_fetcher_verbosity,
    check_type_fetcher,
    dict_to_archive,
)
from nilearn.image import get_data


def validate_atlas(atlas_data, check_type=True):
    """Validate content of the atlas bunch.

    Atlas must:
    - be a bunch
    - with description, template and atlas_type attributes
    - deterministic atlases must have:
      - a labels attribute that is a list
      - a lut attribute that is a pd.DataFrame
    """
    if check_type:
        check_type_fetcher(atlas_data)
    assert isinstance(atlas_data, Bunch)

    assert atlas_data.template != ""
    assert atlas_data.atlas_type in {"deterministic", "probabilistic"}
    if atlas_data.atlas_type == "deterministic":
        assert isinstance(atlas_data.labels, list)
        assert all(isinstance(x, str) for x in atlas_data.labels)
        assert isinstance(atlas_data.lut, pd.DataFrame)
        if "fsaverage" not in atlas_data.template:
            assert "Background" in atlas_data.labels


def test_downloader(tmp_path, request_mocker):
    # Sandboxing test
    # ===============

    # When nilearn downloads a file, everything is first downloaded in a
    # temporary directory (sandbox) and moved to the "real" data directory if
    # all files are present. In case of error, the sandbox is deleted.

    # To test this feature, we do as follow:
    # - create the data dir with a file that has a specific content
    # - try to download the dataset but make it fail
    #   on purpose (by requesting a file that is not in the archive)
    # - check that the previously created file is untouched :
    #   - if sandboxing is faulty, the file would be replaced
    #     by the file of the archive
    #   - if sandboxing works, the file must be untouched.

    local_archive = (
        Path(__file__).parent / "data" / "craddock_2011_parcellations.tar.gz"
    )
    url = "http://example.com/craddock_atlas"
    request_mocker.url_mapping["*craddock*"] = local_archive
    datasetdir = tmp_path / "craddock_2012"
    datasetdir.mkdir()

    # Create a dummy file. If sandboxing is successful, it won't be overwritten
    dummy_file = datasetdir / "random_all.nii.gz"
    with dummy_file.open("w") as f:
        f.write("stuff")

    opts = {"uncompress": True}
    files = [
        ("random_all.nii.gz", url, opts),
        # The following file does not exists. It will cause an abortion of
        # the fetching procedure
        ("bald.nii.gz", url, opts),
    ]

    with pytest.raises(IOError):
        fetch_files(
            str(tmp_path / "craddock_2012"),
            files,
            verbose=0,
        )
    with dummy_file.open("r") as f:
        stuff = f.read(5)
    assert stuff == "stuff"

    # Downloading test
    # ================

    # Now, we use the regular downloading feature. This will override the dummy
    # file created before.

    fetch_atlas_craddock_2012(data_dir=tmp_path)
    with dummy_file.open() as f:
        stuff = f.read()
    assert stuff == ""


def test_fetch_atlas_source():
    # specify non-existing atlas source
    with pytest.raises(ValueError, match="'atlas_source' must be one of"):
        atlas._get_atlas_data_and_labels("new_source", "not_inside")


def _write_sample_atlas_metadata(ho_dir, filename, is_symm):
    with Path(ho_dir, f"{filename}.xml").open("w") as dm:
        if not is_symm:
            dm.write(
                "<?xml version='1.0' encoding='us-ascii'?>\n"
                "<data>\n"
                '<label index="0" x="48" y="94" z="35">R1</label>\n'
                '<label index="1" x="25" y="70" z="32">R2</label>\n'
                '<label index="2" x="33" y="73" z="63">R3</label>\n'
                "</data>"
            )
        else:
            dm.write(
                "<?xml version='1.0' encoding='us-ascii'?>\n"
                "<data>\n"
                '<label index="0" x="63" y="86" z="49">Left R1</label>\n'
                '<label index="1" x="21" y="86" z="33">Right R1</label>\n'
                '<label index="2" x="64" y="69" z="32">Left R2</label>\n'
                '<label index="3" x="26" y="70" z="32">Right R2</label>\n'
                '<label index="4" x="47" y="75" z="66">Left R3</label>\n'
                '<label index="5" x="43" y="80" z="61">Right R3</label>\n'
                "</data>"
            )
        dm.close()


def _test_atlas_instance_should_match_data(atlas, is_symm):
    assert Path(atlas.filename).exists() and Path(atlas.filename).is_absolute()
    assert isinstance(atlas.maps, Nifti1Image)
    assert isinstance(atlas.labels, list)

    expected_atlas_labels = (
        [
            "Background",
            "Left R1",
            "Right R1",
            "Left R2",
            "Right R2",
            "Left R3",
            "Right R3",
        ]
        if is_symm
        else ["Background", "R1", "R2", "R3"]
    )
    assert atlas.labels == expected_atlas_labels


@pytest.fixture
def fsl_fetcher(name):
    return (
        fetch_atlas_juelich
        if name == "Juelich"
        else fetch_atlas_harvard_oxford
    )


@pytest.mark.parametrize(
    "name,prob", [("HarvardOxford", "cortl-prob-1mm"), ("Juelich", "prob-1mm")]
)
def test_fetch_atlas_fsl_errors(prob, fsl_fetcher, tmp_path):
    # specify non-existing atlas item
    with pytest.raises(ValueError, match="'atlas_name' must be one of"):
        fsl_fetcher(atlas_name="not_inside")
    # Choose a probabilistic atlas with symmetric split
    with pytest.raises(ValueError, match="Region splitting"):
        fsl_fetcher(prob, data_dir=str(tmp_path), symmetric_split=True)


@pytest.fixture
def atlas_data():
    # Create false atlas
    atlas_data = np.zeros((10, 10, 10), dtype="int32")
    # Create an interhemispheric map
    atlas_data[:, :2, :] = 1
    # Create a left map
    atlas_data[5:, 7:9, :] = 3
    atlas_data[5:, 3, :] = 2
    # Create a right map, with one voxel on the left side
    atlas_data[:5:, 3:5, :] = 2
    atlas_data[:5, 8, :] = 3
    atlas_data[4, 7, 0] = 3
    return atlas_data


@pytest.mark.parametrize(
    "name, label_fname, fname, is_symm, split",
    [
        ("HarvardOxford", "-Cortical", "cort-prob-1mm", False, False),
        ("HarvardOxford", "-Subcortical", "sub-maxprob-thr0-1mm", False, True),
        (
            "HarvardOxford",
            "-Cortical-Lateralized",
            "cortl-maxprob-thr0-1mm",
            True,
            True,
        ),
        ("Juelich", "", "prob-1mm", False, False),
        ("Juelich", "", "maxprob-thr0-1mm", False, False),
        ("Juelich", "", "maxprob-thr0-1mm", False, True),
    ],
)
def test_fetch_atlas_fsl(
    name,
    label_fname,
    fname,
    is_symm,
    split,
    atlas_data,
    fsl_fetcher,
    tmp_path,
    affine_eye,
):
    # Create directory which will contain fake atlas data
    atlas_dir = tmp_path / "fsl" / "data" / "atlases"
    nifti_dir = atlas_dir / name
    nifti_dir.mkdir(parents=True)
    # Write labels and sample atlas image in directory
    _write_sample_atlas_metadata(
        atlas_dir,
        f"{name}{label_fname}",
        is_symm=is_symm,
    )
    target_atlas_nii = nifti_dir / f"{name}-{fname}.nii.gz"
    Nifti1Image(atlas_data, affine_eye * 3).to_filename(target_atlas_nii)
    # Check that the fetch lead to consistent results
    atlas_instance = fsl_fetcher(
        fname,
        data_dir=tmp_path,
        symmetric_split=split,
    )

    validate_atlas(atlas_instance)
    _test_atlas_instance_should_match_data(
        atlas_instance,
        is_symm=is_symm or split,
    )

    # check for typo in label names
    for label in atlas_instance.labels:
        # no extra whitespace
        assert label.strip() == label


@pytest.mark.parametrize(
    "homogeneity, grp_mean, expected",
    [
        ("spatial", True, "scorr05_mean_all.nii.gz"),
        ("random", True, "random_all.nii.gz"),
        ("spatial", False, "scorr05_2level_all.nii.gz"),
    ],
)
def test_fetch_atlas_craddock_2012(
    tmp_path, request_mocker, homogeneity, grp_mean, expected, capsys
):
    local_archive = (
        Path(__file__).parent / "data" / "craddock_2011_parcellations.tar.gz"
    )
    request_mocker.url_mapping["*craddock*"] = local_archive

    bunch = fetch_atlas_craddock_2012(
        data_dir=tmp_path,
        verbose=0,
        homogeneity=homogeneity,
        grp_mean=grp_mean,
    )

    validate_atlas(bunch)
    assert bunch["maps"] == str(tmp_path / "craddock_2012" / expected)

    assert request_mocker.url_count == 1

    check_fetcher_verbosity(
        fetch_atlas_craddock_2012, capsys, data_dir=tmp_path
    )


def test_fetch_atlas_smith_2009(tmp_path):
    bunch = fetch_atlas_smith_2009(data_dir=tmp_path, verbose=0, dimension=20)

    validate_atlas(bunch)
    assert bunch["maps"] == str(tmp_path / "smith_2009" / "rsn20.nii.gz")


def test_fetch_coords_power_2011():
    bunch = atlas.fetch_coords_power_2011()

    assert len(bunch.rois) == 264


def test_fetch_coords_seitzman_2018():
    bunch = atlas.fetch_coords_seitzman_2018()

    assert len(bunch.rois) == 300
    assert len(bunch.radius) == 300
    assert len(bunch.networks) == 300
    assert len(bunch.regions) == 300
    assert len(np.unique(bunch.networks)) == 14
    assert len(np.unique(bunch.regions)) == 8
    assert_array_equal(bunch.networks, np.sort(bunch.networks))
    assert bunch.regions[0] == "cortexL"

    bunch = atlas.fetch_coords_seitzman_2018(ordered_regions=False)
    assert np.any(bunch.networks != np.sort(bunch.networks))


def _destrieux_data():
    """Mock the download of the destrieux atlas."""
    data = {"destrieux2009.rst": "readme"}
    background_value = 0
    atlas = _rng().integers(background_value, 10, (10, 10, 10), dtype="int32")
    atlas_img = Nifti1Image(atlas, np.eye(4))
    labels = "\n".join([f"{idx},label {idx}" for idx in range(10)])
    labels = f"index,name\n0,Background\n{labels}"
    for lat in ["_lateralized", ""]:
        lat_data = {
            f"destrieux2009_rois_labels{lat}.csv": labels,
            f"destrieux2009_rois{lat}.nii.gz": atlas_img,
        }
        data.update(lat_data)
    return dict_to_archive(data)


@pytest.mark.parametrize("lateralized", [True, False])
def test_fetch_atlas_destrieux_2009(
    tmp_path, request_mocker, lateralized, capsys
):
    """Tests for function `fetch_atlas_destrieux_2009`.

    The atlas is fetched with different values for `lateralized`.
    """
    request_mocker.url_mapping["*destrieux2009.tgz"] = _destrieux_data()
    bunch = fetch_atlas_destrieux_2009(
        lateralized=lateralized, data_dir=tmp_path, verbose=0
    )

    validate_atlas(bunch)

    assert request_mocker.url_count == 1

    name = "_lateralized" if lateralized else ""

    assert bunch["maps"] == str(
        tmp_path / "destrieux_2009" / f"destrieux2009_rois{name}.nii.gz"
    )

    check_fetcher_verbosity(
        fetch_atlas_destrieux_2009,
        capsys,
        lateralized=lateralized,
        data_dir=tmp_path,
    )


def test_fetch_atlas_msdl(tmp_path, request_mocker, capsys):
    labels = pd.DataFrame(
        {
            "x": [1.5, 1.2],
            "y": [1.5, 1.3],
            "z": [1.5, 1.4],
            "name": ["Aud", "DMN"],
            "net name": ["Aud", "DMN"],
        }
    )
    root = Path("MSDL_rois")
    archive = {
        root / "msdl_rois_labels.csv": labels.to_csv(index=False),
        root / "msdl_rois.nii": "",
        root / "README.txt": "",
    }
    request_mocker.url_mapping["*MSDL_rois.zip"] = dict_to_archive(
        archive, "zip"
    )
    dataset = fetch_atlas_msdl(data_dir=tmp_path, verbose=0)

    validate_atlas(dataset)
    assert isinstance(dataset.region_coords, list)
    assert isinstance(dataset.networks, list)
    assert isinstance(dataset.maps, str)
    assert request_mocker.url_count == 1

    check_fetcher_verbosity(fetch_atlas_msdl, capsys, data_dir=tmp_path)


def _generate_yeo_data(tmp_path):
    """Generate files for a dummy Yeo atlas.

    So far it only generates 7 networks data.
    """
    dataset_name = "yeo_2011"
    yeo_archive_root = "Yeo_JNeurophysiol11_MNI152"

    mock_dir = tmp_path / dataset_name / yeo_archive_root
    mock_dir.mkdir(exist_ok=True, parents=True)

    # only mock 7 networks for now
    n_roi = 7

    to_archive = {}

    mock_map = data_gen.generate_labeled_regions((53, 64, 52), n_roi)
    for basename in (
        "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz",
        "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
        "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz",
        "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
        "FSL_MNI152_FreeSurferConformed_1mm.nii.gz",
    ):
        mock_file = mock_dir / basename
        mock_map.to_filename(mock_file)
        to_archive[Path(yeo_archive_root) / basename] = mock_file

    mock_lut = pd.DataFrame(
        {
            "name": [None] + [f"7Networks_{x}" for x in range(1, n_roi + 1)],
            "r": [1] * (n_roi + 1),
            "g": [1] * (n_roi + 1),
            "b": [1] * (n_roi + 1),
            "o": [0] * (n_roi + 1),
        }
    )
    for basename in (
        "Yeo2011_7Networks_ColorLUT.txt",
        "Yeo2011_17Networks_ColorLUT.txt",
    ):
        mock_file = mock_dir / basename
        mock_lut.to_csv(mock_file, sep=" ", header=False)
        to_archive[Path(yeo_archive_root) / basename] = mock_file

    return dict_to_archive(to_archive, archive_format="zip")


def test_fetch_atlas_yeo_2011(tmp_path, request_mocker, capsys):
    """Check fetcher for the Yeo atlas.

    Mocks data for each deterministic atlas and their look up tables.
    """
    yeo_data = _generate_yeo_data(tmp_path)

    request_mocker.url_mapping["*Yeo_JNeurophysiol11_MNI152*"] = yeo_data

    dataset = fetch_atlas_yeo_2011(data_dir=tmp_path, verbose=0)
    dataset = fetch_atlas_yeo_2011(data_dir=tmp_path, verbose=0, n_networks=17)
    dataset = fetch_atlas_yeo_2011(
        data_dir=tmp_path, verbose=0, thickness="thin"
    )

    validate_atlas(dataset)

    check_fetcher_verbosity(fetch_atlas_yeo_2011, capsys, data_dir=tmp_path)


def test_fetch_atlas_yeo_2011_error(tmp_path):
    """Raise errors when the wrong values are passed."""
    with pytest.raises(ValueError, match="'n_networks' must be one of"):
        fetch_atlas_yeo_2011(data_dir=tmp_path, verbose=0, n_networks=10)

    with pytest.raises(ValueError, match="'thickness' must be one of"):
        fetch_atlas_yeo_2011(
            data_dir=tmp_path, verbose=0, thickness="dead_parot"
        )


def test_fetch_atlas_difumo(tmp_path, request_mocker, capsys):
    resolutions = [2, 3]  # Valid resolution values
    dimensions = [64, 128, 256, 512, 1024]  # Valid dimension values
    dimension_urls = ["pqu9r", "wjvd5", "3vrct", "9b76y", "34792"]
    url_mapping = dict(zip(dimensions, dimension_urls, strict=False))
    for url_count, dim in enumerate(dimensions, start=2):
        url = f"*osf.io/{url_mapping[dim]}/*"
        labels = pd.DataFrame(
            {
                "Component": list(range(1, dim + 1)),
                "Difumo_names": ["" for _ in range(dim)],
                "Yeo_networks7": ["" for _ in range(dim)],
                "Yeo_networks17": ["" for _ in range(dim)],
                "GM": ["" for _ in range(dim)],
                "WM": ["" for _ in range(dim)],
                "CSF": ["" for _ in range(dim)],
            }
        )
        root = Path(f"{dim}")
        archive = {
            root / f"labels_{dim}_dictionary.csv": labels.to_csv(index=False),
            root / "2mm" / "maps.nii.gz": "",
            root / "3mm" / "maps.nii.gz": "",
        }
        request_mocker.url_mapping[url] = dict_to_archive(archive, "zip")

        for res in resolutions:
            dataset = fetch_atlas_difumo(
                data_dir=tmp_path, dimension=dim, resolution_mm=res, verbose=0
            )

            validate_atlas(dataset)
            assert len(dataset.labels) == dim
            assert isinstance(dataset.maps, str)
            assert request_mocker.url_count == url_count

    with pytest.raises(ValueError):
        fetch_atlas_difumo(data_dir=tmp_path, dimension=42, resolution_mm=3)
        fetch_atlas_difumo(
            data_dir=tmp_path, dimension=128, resolution_mm=3.14
        )

    check_fetcher_verbosity(fetch_atlas_difumo, capsys, data_dir=tmp_path)


@pytest.fixture
def aal_archive_root(version):
    if version == "SPM12":
        return Path("aal", "atlas")
    else:
        return Path(f"aal_for_{version}")


@pytest.mark.parametrize(
    "version,archive_format,url_key",
    [
        ("SPM5", "zip", "uploads"),
        ("SPM8", "zip", "uploads"),
        ("SPM12", "gztar", "AAL_files"),
    ],
)
def test_fetch_atlas_aal(
    version,
    archive_format,
    url_key,
    aal_archive_root,
    tmp_path,
    request_mocker,
    img_3d_rand_eye,
):
    metadata = "1\t2\t3\n"
    if version == "SPM12":
        metadata = (
            b"<?xml version='1.0' encoding='us-ascii'?>"
            b"<metadata><label><index>1</index>"
            b"<name>A</name></label></metadata>"
        )
    label_file = "AAL.xml" if version == "SPM12" else "ROI_MNI_V4.txt"

    atlas_file = "AAL.nii" if version == "SPM12" else "ROI_MNI_V4.nii"

    mock_file = tmp_path / f"aal_{version}" / aal_archive_root / atlas_file
    mock_file.parent.mkdir(exist_ok=True, parents=True)
    img_3d_rand_eye.to_filename(mock_file)

    aal_data = dict_to_archive(
        {
            aal_archive_root / label_file: metadata,
            aal_archive_root / atlas_file: mock_file,
        },
        archive_format=archive_format,
    )
    request_mocker.url_mapping[f"*{url_key}*"] = aal_data

    dataset = fetch_atlas_aal(version=version, data_dir=tmp_path, verbose=0)

    validate_atlas(dataset)
    assert isinstance(dataset.maps, str)
    assert isinstance(dataset.indices, list)
    assert request_mocker.url_count == 1


def test_fetch_atlas_aal_version_error(tmp_path):
    with pytest.raises(ValueError, match="'version' must be one of"):
        fetch_atlas_aal(version="FLS33", data_dir=tmp_path, verbose=0)


def test_fetch_atlas_basc_multiscale_2015(tmp_path, capsys):
    resolution = 7

    dataset_name = "basc_multiscale_2015"
    name_sym = "template_cambridge_basc_multiscale_nii_sym"
    basename_sym = "template_cambridge_basc_multiscale_sym_scale007.nii.gz"

    mock_map = data_gen.generate_labeled_regions((53, 64, 52), resolution)
    mock_file = tmp_path / dataset_name / name_sym / basename_sym
    mock_file.parent.mkdir(exist_ok=True, parents=True)
    mock_map.to_filename(mock_file)

    # default version='sym',
    data_sym = fetch_atlas_basc_multiscale_2015(
        data_dir=tmp_path, verbose=0, resolution=resolution
    )

    validate_atlas(data_sym)
    assert data_sym["maps"] == str(
        tmp_path / dataset_name / name_sym / basename_sym
    )

    name_asym = "template_cambridge_basc_multiscale_nii_asym"
    basename_asym = "template_cambridge_basc_multiscale_asym_scale007.nii.gz"

    # version='asym'
    mock_file = tmp_path / dataset_name / name_asym / basename_asym
    mock_file.parent.mkdir(exist_ok=True, parents=True)
    mock_map.to_filename(mock_file)

    data_asym = fetch_atlas_basc_multiscale_2015(
        version="asym", verbose=0, data_dir=tmp_path, resolution=resolution
    )

    validate_atlas(data_asym)
    assert data_asym["maps"] == str(
        tmp_path / dataset_name / name_asym / basename_asym
    )

    check_fetcher_verbosity(
        fetch_atlas_basc_multiscale_2015, capsys, data_dir=tmp_path
    )


def test_fetch_atlas_basc_multiscale_2015_error(tmp_path):
    with pytest.raises(ValueError, match="'version' must be one of"):
        fetch_atlas_basc_multiscale_2015(
            version="aym", data_dir=tmp_path, verbose=0
        )


def test_fetch_coords_dosenbach_2010():
    bunch = atlas.fetch_coords_dosenbach_2010()

    assert len(bunch.rois) == 160
    assert len(bunch.labels) == 160
    assert len(np.unique(bunch.networks)) == 6
    assert_array_equal(bunch.networks, np.sort(bunch.networks))

    bunch = atlas.fetch_coords_dosenbach_2010(ordered_regions=False)

    assert np.any(bunch.networks != np.sort(bunch.networks))


def test_fetch_atlas_allen_2011(tmp_path, request_mocker, capsys):
    """Fetch allen atlas and checks filenames are those expected."""
    bunch = fetch_atlas_allen_2011(data_dir=tmp_path, verbose=0)
    keys = ("maps", "rsn28", "comps")

    filenames = [
        "ALL_HC_unthresholded_tmaps.nii.gz",
        "RSN_HC_unthresholded_tmaps.nii.gz",
        "rest_hcp_agg__component_ica_.nii.gz",
    ]

    validate_atlas(bunch)
    assert request_mocker.url_count == 1
    for key, fn in zip(keys, filenames, strict=False):
        assert bunch[key] == str(
            tmp_path / "allen_rsn_2011" / "allen_rsn_2011" / fn
        )

    check_fetcher_verbosity(fetch_atlas_allen_2011, capsys, data_dir=tmp_path)


def test_fetch_atlas_surf_destrieux(tmp_path, capsys):
    data_dir = tmp_path / "destrieux_surface"
    data_dir.mkdir()

    # Create mock annots
    for hemi in ("left", "right"):
        freesurfer.write_annot(
            data_dir / f"{hemi}.aparc.a2009s.annot",
            np.arange(4),
            np.zeros((4, 5)),
            5 * ["a"],
        )

    bunch = fetch_atlas_surf_destrieux(data_dir=tmp_path, verbose=0)

    # one exception is made here to return some numpy array
    # so we do not check the type
    validate_atlas(bunch, check_type=False)

    # Our mock annots have 4 labels
    assert len(bunch.labels) == 4
    assert bunch.map_left.shape == (4,)
    assert bunch.map_right.shape == (4,)

    check_fetcher_verbosity(
        fetch_atlas_surf_destrieux, capsys, data_dir=tmp_path
    )


def _get_small_fake_talairach():
    labels = ["*", "b", "a"]
    all_labels = itertools.product(*(labels,) * 5)
    labels_txt = "\n".join(map(".".join, all_labels))
    extensions = nifti1.Nifti1Extensions(
        [nifti1.Nifti1Extension("afni", labels_txt.encode("utf-8"))]
    )
    img = Nifti1Image(
        np.arange(243, dtype="int32").reshape((3, 9, 9)),
        np.eye(4),
        Nifti1Header(extensions=extensions),
    )
    return serialize_niimg(img, gzipped=False)


@pytest.mark.slow
def test_fetch_atlas_talairach(tmp_path, request_mocker, capsys):
    request_mocker.url_mapping["*talairach.nii"] = _get_small_fake_talairach()
    level_values = np.ones((81, 3)) * [0, 1, 2]
    talairach = fetch_atlas_talairach("hemisphere", data_dir=tmp_path)

    validate_atlas(talairach)

    assert_array_equal(
        get_data(talairach.maps).ravel(), level_values.T.ravel()
    )
    assert_array_equal(talairach.labels, ["Background", "b", "a"])

    talairach = fetch_atlas_talairach("ba", data_dir=tmp_path)

    assert_array_equal(get_data(talairach.maps).ravel(), level_values.ravel())

    with pytest.raises(ValueError):
        fetch_atlas_talairach("bad_level")

    check_fetcher_verbosity(
        fetch_atlas_talairach,
        capsys,
        level_name="hemisphere",
        data_dir=tmp_path,
    )


def test_fetch_atlas_pauli_2017(tmp_path, request_mocker):
    labels = pd.DataFrame({"label": [f"label_{i}" for i in range(16)]}).to_csv(
        sep="\t", header=False
    )
    det_atlas = data_gen.generate_labeled_regions((7, 6, 5), 16)
    prob_atlas, _ = data_gen.generate_maps((7, 6, 5), 16)
    request_mocker.url_mapping["*osf.io/6qrcb/*"] = labels
    request_mocker.url_mapping["*osf.io/5mqfx/*"] = det_atlas
    request_mocker.url_mapping["*osf.io/w8zq2/*"] = prob_atlas
    data_dir = str(tmp_path / "pauli_2017")

    data = fetch_atlas_pauli_2017("deterministic", data_dir)

    validate_atlas(data)
    assert len(data.labels) == 17

    values = get_data(load(data.maps))

    assert len(np.unique(values)) == 17

    data = fetch_atlas_pauli_2017("probabilistic", data_dir)

    validate_atlas(data)
    assert load(data.maps).shape[-1] == 16

    with pytest.raises(ValueError, match="'atlas_type' must be one of"):
        fetch_atlas_pauli_2017(
            atlas_type="junk for testing", data_dir=data_dir
        )


def _schaefer_labels(match, requests):  # noqa: ARG001
    # fails if requests is not passed
    info = match.groupdict()
    label_names = [f"{info['network']}Networks"] * int(info["n_rois"])
    labels = pd.DataFrame({"label": label_names})
    return labels.to_csv(sep="\t", header=False).encode("utf-8")


def _schaefer_img(match, requests):  # noqa: ARG001
    # fails if requests is not passed
    info = match.groupdict()
    shape = (15, 14, 13)
    affine = np.eye(4) * float(info["res"])
    affine[3, 3] = 1.0
    img = data_gen.generate_labeled_regions(
        shape, int(info["n_rois"]), affine=affine
    )
    return serialize_niimg(img)


def test_fetch_atlas_schaefer_2018_errors():
    with pytest.raises(ValueError):
        fetch_atlas_schaefer_2018(n_rois=44)
    with pytest.raises(ValueError):
        fetch_atlas_schaefer_2018(yeo_networks=10)
    with pytest.raises(ValueError):
        fetch_atlas_schaefer_2018(resolution_mm=3)


@pytest.mark.parametrize("n_rois", list(range(100, 1100, 100)))
@pytest.mark.parametrize("yeo_networks", [7, 17])
@pytest.mark.parametrize("resolution_mm", [1, 2])
def test_fetch_atlas_schaefer_2018(
    tmp_path, request_mocker, n_rois, yeo_networks, resolution_mm, capsys
):
    labels_pattern = re.compile(
        r".*2018_(?P<n_rois>\d+)Parcels_(?P<network>\d+)Networks_order.txt"
    )
    img_pattern = re.compile(
        r".*_(?P<n_rois>\d+)Parcels_(?P<network>\d+)"
        r"Networks_order_FSLMNI152_(?P<res>\d)mm.nii.gz"
    )
    request_mocker.url_mapping[labels_pattern] = _schaefer_labels
    request_mocker.url_mapping[img_pattern] = _schaefer_img

    mock_lut = pd.DataFrame(
        {
            "name": [f"{yeo_networks}Networks_{x}" for x in range(1, n_rois)],
            "r": [1] * (n_rois - 1),
            "g": [1] * (n_rois - 1),
            "b": [1] * (n_rois - 1),
            "o": [0] * (n_rois - 1),
        }
    )
    basename = f"Schaefer2018_{n_rois}Parcels_{yeo_networks}Networks_order.txt"
    mock_dir = tmp_path / "schaefer_2018"
    mock_dir.mkdir(exist_ok=True, parents=True)
    mock_file = mock_dir / basename
    mock_lut.to_csv(mock_file, sep="\t", header=False)

    data = fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=resolution_mm,
        data_dir=tmp_path,
        verbose=0,
    )

    validate_atlas(data)

    assert isinstance(data.maps, str)

    assert len(data.labels) == n_rois
    assert data.labels[0] == "Background"
    assert data.labels[1].startswith(f"{yeo_networks}Networks")

    img = load(data.maps)

    assert img.header.get_zooms()[0] == resolution_mm
    assert np.array_equal(np.unique(img.dataobj), np.arange(n_rois + 1))

    check_fetcher_verbosity(
        fetch_atlas_schaefer_2018,
        capsys,
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=resolution_mm,
        data_dir=tmp_path,
    )


@pytest.fixture
def aal_xml():
    atlas = ET.Element("atlas", version="2.0")

    data = ET.SubElement(atlas, "data")

    label1 = ET.SubElement(data, "label")
    ET.SubElement(label1, "index").text = "2001"
    ET.SubElement(label1, "name").text = "Precentral_L"

    # Convert the XML tree to a string with proper encoding and declaration
    return ET.ElementTree(atlas)
