"""Test the datasets module."""

# Author: Alexandre Abraham

import json
import os
import re
import shutil
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

import nibabel
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nilearn.datasets import func
from nilearn.datasets._testing import dict_to_archive, list_to_archive
from nilearn.datasets.utils import _get_dataset_dir


def _load_localizer_index():
    data_dir = Path(__file__).parent / "data"
    with (data_dir / "localizer_index.json").open() as of:
        localizer_template = json.load(of)
    localizer_index = {}
    for idx in range(1, 95):
        sid = f"S{idx:02}"
        localizer_index.update(
            {key.format(sid): uuid.uuid4().hex for key in localizer_template}
        )
    localizer_index["/localizer/phenotype/behavioural.tsv"] = uuid.uuid4().hex
    localizer_index["/localizer/participants.tsv"] = uuid.uuid4().hex
    tsv_files = {
        "/localizer/phenotype/behavioural.tsv": pd.read_csv(
            data_dir / "localizer_behavioural.tsv", sep="\t"
        )
    }
    tsv_files["/localizer/participants.tsv"] = pd.read_csv(
        data_dir / "localizer_participants.tsv", sep="\t"
    )
    return localizer_index, tsv_files


@pytest.fixture()
def localizer_mocker(request_mocker):
    """Mocks the index for localizer dataset."""
    index, tsv_files = _load_localizer_index()
    request_mocker.url_mapping["https://osf.io/hwbm2/download"] = json.dumps(
        index
    )
    for k, v in tsv_files.items():
        request_mocker.url_mapping[f"*{index[k][1:]}?"] = v.to_csv(
            index=False, sep="\t"
        )


def _make_haxby_subject_data(match, response):
    sub_files = [
        "bold.nii.gz",
        "labels.txt",
        "mask4_vt.nii.gz",
        "mask8b_face_vt.nii.gz",
        "mask8b_house_vt.nii.gz",
        "mask8_face_vt.nii.gz",
        "mask8_house_vt.nii.gz",
        "anat.nii.gz",
    ]
    return list_to_archive(Path(match.group(1), f) for f in sub_files)


def test_fetch_haxby(tmp_path, request_mocker):
    request_mocker.url_mapping[
        re.compile(r".*(subj\d).*\.tar\.gz")
    ] = _make_haxby_subject_data
    for i in range(1, 6):
        haxby = func.fetch_haxby(data_dir=tmp_path, subjects=[i], verbose=0)
        # subject_data + (md5 + mask if first subj)

        assert request_mocker.url_count == i + 2
        assert len(haxby.func) == 1
        assert len(haxby.anat) == 1
        assert len(haxby.session_target) == 1
        assert haxby.mask is not None
        assert len(haxby.mask_vt) == 1
        assert len(haxby.mask_face) == 1
        assert len(haxby.mask_house) == 1
        assert len(haxby.mask_face_little) == 1
        assert len(haxby.mask_house_little) == 1
        assert haxby.description != ""

    # subjects with list
    subjects = [1, 2, 6]
    haxby = func.fetch_haxby(data_dir=tmp_path, subjects=subjects, verbose=0)

    assert len(haxby.func) == len(subjects)
    assert len(haxby.mask_house_little) == len(subjects)
    assert len(haxby.anat) == len(subjects)
    assert haxby.anat[2] is None
    assert isinstance(haxby.mask, str)
    assert len(haxby.mask_face) == len(subjects)
    assert len(haxby.session_target) == len(subjects)
    assert len(haxby.mask_vt) == len(subjects)
    assert len(haxby.mask_face_little) == len(subjects)

    subjects = ["a", 8]
    message = "You provided invalid subject id {0} in a list"

    for sub_id in subjects:
        with pytest.raises(ValueError, match=message.format(sub_id)):
            func.fetch_haxby(data_dir=tmp_path, subjects=[sub_id])


def _adhd_example_subject(match, request):
    contents = [
        Path("data", match.group(1), match.expand(r"\1_regressors.csv")),
        Path(
            "data",
            match.group(1),
            match.expand(r"\1_rest_tshift_RPI_voreg_mni.nii.gz"),
        ),
    ]
    return list_to_archive(contents)


def _adhd_metadata():
    sub1 = [3902469, 7774305, 3699991]
    sub2 = [
        2014113,
        4275075,
        1019436,
        3154996,
        3884955,
        27034,
        4134561,
        27018,
        6115230,
        27037,
        8409791,
        27011,
    ]
    sub3 = [
        3007585,
        8697774,
        9750701,
        10064,
        21019,
        10042,
        10128,
        2497695,
        4164316,
        1552181,
        4046678,
        23012,
    ]
    sub4 = [
        1679142,
        1206380,
        23008,
        4016887,
        1418396,
        2950754,
        3994098,
        3520880,
        1517058,
        9744150,
        1562298,
        3205761,
        3624598,
    ]
    subs = pd.DataFrame({"Subject": sub1 + sub2 + sub3 + sub4})
    tmp = "ADHD200_40subs_motion_parameters_and_phenotypics.csv"
    return dict_to_archive({tmp: subs.to_csv(index=False)})


def test_fetch_adhd(tmp_path, request_mocker):
    request_mocker.url_mapping["*metadata.tgz"] = _adhd_metadata()
    request_mocker.url_mapping[
        re.compile(r".*adhd40_([0-9]+)\.tgz")
    ] = _adhd_example_subject
    adhd = func.fetch_adhd(data_dir=tmp_path, n_subjects=12, verbose=0)

    assert len(adhd.func) == 12
    assert len(adhd.confounds) == 12
    assert request_mocker.url_count == 13  # Subjects + phenotypic
    assert adhd.description != ""


def test_miyawaki2008(tmp_path, request_mocker):
    dataset = func.fetch_miyawaki2008(data_dir=tmp_path, verbose=0)

    assert len(dataset.func) == 32
    assert len(dataset.label) == 32
    assert isinstance(dataset.mask, str)
    assert len(dataset.mask_roi) == 38
    assert isinstance(dataset.background, str)
    assert request_mocker.url_count == 1
    assert dataset.description != ""


def test_fetch_localizer_contrasts(tmp_path, localizer_mocker):
    # 2 subjects
    dataset = func.fetch_localizer_contrasts(
        ["checkerboard"],
        n_subjects=2,
        data_dir=tmp_path,
        verbose=1,
        legacy_format=True,
    )

    assert not hasattr(dataset, "anats")
    assert not hasattr(dataset, "tmaps")
    assert not hasattr(dataset, "masks")
    assert isinstance(dataset.cmaps[0], str)
    assert isinstance(dataset.ext_vars, np.recarray)
    assert len(dataset.cmaps) == 2
    assert dataset.ext_vars.size == 2

    dataset = func.fetch_localizer_contrasts(
        ["checkerboard"],
        n_subjects=2,
        data_dir=tmp_path,
        verbose=1,
        legacy_format=False,
    )

    assert not hasattr(dataset, "anats")
    assert not hasattr(dataset, "tmaps")
    assert not hasattr(dataset, "masks")
    assert isinstance(dataset.cmaps[0], str)
    assert isinstance(dataset.ext_vars, pd.DataFrame)
    assert len(dataset.cmaps) == 2
    assert len(dataset["ext_vars"]) == 2


def test_fetch_localizer_contrasts_multiple_contrasts(
    tmp_path, localizer_mocker
):
    dataset = func.fetch_localizer_contrasts(
        ["checkerboard", "horizontal checkerboard"],
        n_subjects=2,
        data_dir=tmp_path,
        verbose=1,
        legacy_format=False,
    )

    assert isinstance(dataset.ext_vars, pd.DataFrame)
    assert isinstance(dataset.cmaps[0], str)
    assert len(dataset.cmaps) == 2 * 2  # two contrasts are fetched
    assert len(dataset["ext_vars"]) == 2


def test_fetch_localizer_contrasts_get_all(tmp_path, localizer_mocker):
    # all get_*=True
    dataset = func.fetch_localizer_contrasts(
        ["checkerboard"],
        n_subjects=1,
        data_dir=tmp_path,
        get_anats=True,
        get_masks=True,
        get_tmaps=True,
        verbose=1,
        legacy_format=False,
    )

    assert isinstance(dataset.ext_vars, pd.DataFrame)
    assert isinstance(dataset.anats[0], str)
    assert isinstance(dataset.cmaps[0], str)
    assert isinstance(dataset.masks[0], str)
    assert isinstance(dataset.tmaps[0], str)
    assert len(dataset["ext_vars"]) == 1
    assert len(dataset.anats) == 1
    assert len(dataset.cmaps) == 1
    assert len(dataset.masks) == 1
    assert len(dataset.tmaps) == 1
    assert dataset.description != ""


def test_fetch_localizer_contrasts_list_subjects(tmp_path, localizer_mocker):
    # grab a given list of subjects
    dataset2 = func.fetch_localizer_contrasts(
        ["checkerboard"],
        n_subjects=[2, 3, 5],
        data_dir=tmp_path,
        verbose=1,
        legacy_format=False,
    )

    assert len(dataset2["ext_vars"]) == 3
    assert len(dataset2.cmaps) == 3
    assert list(dataset2["ext_vars"]["participant_id"].values) == [
        "S02",
        "S03",
        "S05",
    ]


def test_fetch_localizer_calculation_task(tmp_path, localizer_mocker):
    # 2 subjects
    dataset = func.fetch_localizer_calculation_task(
        n_subjects=2, data_dir=tmp_path, verbose=1, legacy_format=False
    )

    assert isinstance(dataset.ext_vars, pd.DataFrame)
    assert isinstance(dataset.cmaps[0], str)
    assert len(dataset["ext_vars"]) == 2
    assert len(dataset.cmaps) == 2
    assert dataset.description != ""

    dataset = func.fetch_localizer_calculation_task(
        n_subjects=2, data_dir=tmp_path, verbose=1, legacy_format=True
    )

    assert isinstance(dataset.ext_vars, np.recarray)
    assert isinstance(dataset.cmaps[0], str)
    assert dataset.ext_vars.size == 2
    assert len(dataset.cmaps) == 2
    assert dataset.description != ""


def test_fetch_localizer_button_task(tmp_path, localizer_mocker):
    # Disabled: cannot be tested without actually fetching covariates CSV file
    # Only one subject
    dataset = func.fetch_localizer_button_task(data_dir=tmp_path, verbose=1)

    assert isinstance(dataset.tmaps, list)
    assert isinstance(dataset.anats, list)

    assert len(dataset.tmaps) == 1
    assert len(dataset.anats) == 1

    assert isinstance(dataset.tmap, str)
    assert isinstance(dataset.anat, str)

    assert dataset.description != ""


@pytest.mark.parametrize("quality_checked", [False, True])
def test_fetch_abide_pcp(tmp_path, request_mocker, quality_checked):
    n_subjects = 800
    ids = list(range(n_subjects))
    filenames = ["no_filename"] * n_subjects
    filenames[::2] = ["filename"] * (n_subjects // 2)
    qc_rater_1 = ["OK"] * n_subjects
    qc_rater_1[::4] = ["fail"] * (n_subjects // 4)
    pheno = pd.DataFrame(
        {
            "subject_id": ids,
            "FILE_ID": filenames,
            "qc_rater_1": qc_rater_1,
            "qc_anat_rater_2": qc_rater_1,
            "qc_func_rater_2": qc_rater_1,
            "qc_anat_rater_3": qc_rater_1,
            "qc_func_rater_3": qc_rater_1,
        },
        columns=[
            "subject_id",
            "FILE_ID",
            "qc_rater_1",
            "qc_anat_rater_2",
            "qc_func_rater_2",
            "qc_anat_rater_3",
            "qc_func_rater_3",
        ],
    )
    request_mocker.url_mapping["*rocessed1.csv"] = pheno.to_csv(index=False)

    # All subjects
    dataset = func.fetch_abide_pcp(
        data_dir=tmp_path, quality_checked=quality_checked, verbose=0
    )
    div = 4 if quality_checked else 2

    assert len(dataset.func_preproc) == n_subjects / div
    assert dataset.description != ""

    # Smoke test using only a string, rather than a list of strings
    dataset = func.fetch_abide_pcp(
        data_dir=tmp_path,
        quality_checked=quality_checked,
        verbose=0,
        derivatives="func_preproc",
    )


def test__load_mixed_gambles(rng):
    n_trials = 48
    affine = np.eye(4)
    for n_subjects in [1, 5, 16]:
        zmaps = [
            nibabel.Nifti1Image(rng.randn(3, 4, 5, n_trials), affine)
            for _ in range(n_subjects)
        ]
        zmaps, gain, _ = func._load_mixed_gambles(zmaps)

        assert len(zmaps) == n_subjects * n_trials
        assert len(zmaps) == len(gain)


def test_fetch_mixed_gambles(tmp_path):
    for n_subjects in [1, 5, 16]:
        mgambles = func.fetch_mixed_gambles(
            n_subjects=n_subjects,
            data_dir=tmp_path,
            verbose=0,
            return_raw_data=True,
        )
        datasetdir = tmp_path / "jimura_poldrack_2012_zmaps"

        assert mgambles["zmaps"][0] == str(
            datasetdir / "zmaps" / "sub001_zmaps.nii.gz"
        )
        assert len(mgambles["zmaps"]) == n_subjects


def test_check_parameters_megatrawls_datasets():
    # testing whether the function raises the same error message
    # if invalid input parameters are provided
    message = "Invalid {0} input is provided: {1}."

    for invalid_input_dim in [1, 5, 30]:
        with pytest.raises(
            ValueError,
            match=message.format("dimensionality", invalid_input_dim),
        ):
            func.fetch_megatrawls_netmats(dimensionality=invalid_input_dim)

    for invalid_input_timeserie in ["asdf", "time", "st2"]:
        with pytest.raises(
            ValueError,
            match=message.format("timeseries", invalid_input_timeserie),
        ):
            func.fetch_megatrawls_netmats(timeseries=invalid_input_timeserie)

    for invalid_output_name in ["net1", "net2"]:
        with pytest.raises(
            ValueError, match=message.format("matrices", invalid_output_name)
        ):
            func.fetch_megatrawls_netmats(matrices=invalid_output_name)


def test_fetch_megatrawls_netmats(tmp_path):
    # smoke test to see that files are fetched and read properly
    # since we are loading data present in it
    for file, folder in zip(
        ["Znet2.txt", "Znet1.txt"],
        [
            "3T_Q1-Q6related468_MSMsulc_d100_ts3",
            "3T_Q1-Q6related468_MSMsulc_d300_ts2",
        ],
    ):
        files_dir = tmp_path / "Megatrawls" / folder
        files_dir.mkdir(parents=True, exist_ok=True)
        with open(files_dir / file, "w") as net_file:
            net_file.write("1")

    megatrawl_netmats_data = func.fetch_megatrawls_netmats(data_dir=tmp_path)

    # expected number of returns in output name should be equal
    assert len(megatrawl_netmats_data) == 5
    # check if returned bunch should not be empty
    # dimensions
    assert megatrawl_netmats_data.dimensions != ""
    # timeseries
    assert megatrawl_netmats_data.timeseries != ""
    # matrices
    assert megatrawl_netmats_data.matrices != ""
    # correlation matrices
    assert megatrawl_netmats_data.correlation_matrices != ""
    # description
    assert megatrawl_netmats_data.description != ""

    # check if input provided for dimensions, timeseries, matrices to be same
    # to user settings
    netmats_data = func.fetch_megatrawls_netmats(
        data_dir=tmp_path,
        dimensionality=300,
        timeseries="multiple_spatial_regression",
        matrices="full_correlation",
    )

    assert netmats_data.dimensions == 300
    assert netmats_data.timeseries == "multiple_spatial_regression"
    assert netmats_data.matrices == "full_correlation"


def test_fetch_surf_nki_enhanced(tmp_path, request_mocker):
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
    nki_data = func.fetch_surf_nki_enhanced(data_dir=tmp_path)

    assert nki_data.description != ""
    assert len(nki_data.func_left) == 10
    assert len(nki_data.func_right) == 10
    assert isinstance(nki_data.phenotypic, np.ndarray)
    assert nki_data.phenotypic.shape == (10,)
    assert nki_data.description != ""


def _mock_participants_data(n_ids=5):
    """Maximum 8 ids are allowed to mock"""
    ids = [
        "sub-pixar052",
        "sub-pixar073",
        "sub-pixar074",
        "sub-pixar110",
        "sub-pixar042",
        "sub-pixar109",
        "sub-pixar068",
        "sub-pixar007",
    ][:n_ids]
    age = np.ones(len(ids))
    age_group = len(ids) * ["2yo"]
    child_adult = [["child", "adult"][i % 2] for i in range(n_ids)]
    gender = len(ids) * ["m"]
    handedness = len(ids) * ["r"]
    participants = pd.DataFrame(
        OrderedDict(
            [
                ("participant_id", ids),
                ("Age", age),
                ("AgeGroup", age_group),
                ("Child_Adult", child_adult),
                ("Gender", gender),
                ("Handedness", handedness),
            ]
        )
    )
    return participants


def _mock_development_confounds():
    keep_confounds = [
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "framewise_displacement",
        "a_comp_cor_00",
        "a_comp_cor_01",
        "a_comp_cor_02",
        "a_comp_cor_03",
        "a_comp_cor_04",
        "a_comp_cor_05",
        "csf",
        "white_matter",
    ]
    other_confounds = ["some_confound"] * 13
    confounds = keep_confounds + other_confounds
    return pd.DataFrame(np.ones((10, len(confounds))), columns=confounds)


def test_fetch_development_fmri_participants(tmp_path, request_mocker):
    mock_participants = _mock_participants_data()
    request_mocker.url_mapping[
        "https://osf.io/yr3av/download"
    ] = mock_participants.to_csv(index=False, sep="\t")
    participants = func._fetch_development_fmri_participants(
        data_dir=tmp_path, url=None, verbose=1
    )

    assert isinstance(participants, np.ndarray)
    assert participants.shape == (5,)


def test_fetch_development_fmri_functional(tmp_path):
    mock_participants = _mock_participants_data(n_ids=8)
    funcs, confounds = func._fetch_development_fmri_functional(
        mock_participants, data_dir=tmp_path, url=None, resume=True, verbose=1
    )

    assert len(funcs) == 8
    assert len(confounds) == 8


def test_fetch_development_fmri(tmp_path, request_mocker):
    mock_participants = _mock_participants_data()
    request_mocker.url_mapping["*"] = _mock_development_confounds().to_csv(
        index=False, sep="\t"
    )
    request_mocker.url_mapping[
        "https://osf.io/yr3av/download"
    ] = mock_participants.to_csv(index=False, sep="\t")

    data = func.fetch_development_fmri(
        n_subjects=2, data_dir=tmp_path, verbose=1
    )

    assert len(data.func) == 2
    assert len(data.confounds) == 2
    assert isinstance(data.phenotypic, np.ndarray)
    assert data.phenotypic.shape == (2,)
    assert data.description != ""

    # check reduced confounds
    confounds = np.recfromcsv(data.confounds[0], delimiter="\t")

    assert len(confounds[0]) == 15

    # check full confounds
    data = func.fetch_development_fmri(
        n_subjects=2, reduce_confounds=False, verbose=1
    )
    confounds = np.recfromcsv(data.confounds[0], delimiter="\t")

    assert len(confounds[0]) == 28

    # check first subject is an adult
    data = func.fetch_development_fmri(
        n_subjects=1, reduce_confounds=False, verbose=1
    )
    age_group = data.phenotypic["Child_Adult"][0]

    assert age_group == "adult"

    # check first subject is an child if requested with age_group
    data = func.fetch_development_fmri(
        n_subjects=1, reduce_confounds=False, verbose=1, age_group="child"
    )
    age_group = data.phenotypic["Child_Adult"][0]

    assert age_group == "child"

    # check one of each age group returned if n_subject == 2
    # and age_group == 'both
    data = func.fetch_development_fmri(
        n_subjects=2, reduce_confounds=False, verbose=1, age_group="both"
    )
    age_group = data.phenotypic["Child_Adult"]

    assert all(age_group == ["adult", "child"])

    # check age_group
    data = func.fetch_development_fmri(
        n_subjects=2, reduce_confounds=False, verbose=1, age_group="child"
    )

    assert all(x == "child" for x in data.phenotypic["Child_Adult"])


def test_fetch_development_fmri_invalid_n_subjects():
    max_subjects = 155
    n_subjects = func._set_invalid_n_subjects_to_max(
        n_subjects=None, max_subjects=max_subjects, age_group="adult"
    )

    assert n_subjects == max_subjects
    with pytest.warns(UserWarning, match="Wrong value for n_subjects="):
        func._set_invalid_n_subjects_to_max(
            n_subjects=-1, max_subjects=max_subjects, age_group="adult"
        )


def test_fetch_development_fmri_exception():
    with pytest.raises(ValueError, match="Wrong value for age_group"):
        func._filter_func_regressors_by_participants(
            participants="junk", age_group="junk for test"
        )


# datasets tests originally belonging to nistats follow

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, "data")


def test_fetch_bids_langloc_dataset(tmp_path):
    data_dir = str(tmp_path / "bids_langloc_example")
    os.mkdir(data_dir)
    main_folder = os.path.join(data_dir, "bids_langloc_dataset")
    os.mkdir(main_folder)

    datadir, dl_files = func.fetch_bids_langloc_dataset(tmp_path)

    assert isinstance(datadir, str)
    assert isinstance(dl_files, list)


def test_select_from_index():
    dataset_version = "ds000030_R1.0.4"
    data_prefix = (
        f"{dataset_version.split('_')[0]}/{dataset_version}/uncompressed"
    )
    # Prepare url files for subject and filter tests
    urls = [
        f"{data_prefix}/{f}"
        for f in [
            "stuff.html",
            "sub-xxx.html",
            "sub-yyy.html",
            "sub-xxx/ses-01_task-rest.txt",
            "sub-xxx/ses-01_task-other.txt",
            "sub-xxx/ses-02_task-rest.txt",
            "sub-xxx/ses-02_task-other.txt",
            "sub-yyy/ses-01.txt",
            "sub-yyy/ses-02.txt",
        ]
    ]

    # Only 1 subject and not subject specific files get downloaded
    new_urls = func.select_from_index(urls, n_subjects=1)

    assert len(new_urls) == 6
    assert data_prefix + "/sub-yyy.html" not in new_urls

    # 2 subjects and not subject specific files get downloaded
    new_urls = func.select_from_index(urls, n_subjects=2)

    assert len(new_urls) == 9
    assert data_prefix + "/sub-yyy.html" in new_urls

    # ALL subjects and not subject specific files get downloaded
    new_urls = func.select_from_index(urls, n_subjects=None)

    assert len(new_urls) == 9

    # test inclusive filters. Only files with task-rest
    new_urls = func.select_from_index(urls, inclusion_filters=["*task-rest*"])

    assert len(new_urls) == 2
    assert data_prefix + "/stuff.html" not in new_urls

    # test exclusive filters. only files without ses-01
    new_urls = func.select_from_index(urls, exclusion_filters=["*ses-01*"])

    assert len(new_urls) == 6
    assert data_prefix + "/stuff.html" in new_urls

    # test filter combination. only files with task-rest and without ses-01
    new_urls = func.select_from_index(
        urls, inclusion_filters=["*task-rest*"], exclusion_filters=["*ses-01*"]
    )

    assert len(new_urls) == 1
    assert data_prefix + "/sub-xxx/ses-02_task-rest.txt" in new_urls


def test_fetch_ds000030_urls():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_version = "ds000030_R1.0.4"
        subdir_names = ["ds000030", "ds000030_R1.0.4", "uncompressed"]
        tmp_list = []
        for subdir in subdir_names:
            tmp_list.append(subdir)
            subdirpath = os.path.join(tmpdir, *tmp_list)
            os.mkdir(subdirpath)

        filepath = os.path.join(subdirpath, "urls.json")
        mock_json_content = ["junk1", "junk2"]
        with open(filepath, "w") as f:
            json.dump(mock_json_content, f)

        # fetch_ds000030_urls should retrieve the appropriate URLs
        urls_path, urls = func.fetch_ds000030_urls(
            data_dir=tmpdir,
            verbose=1,
        )
        urls_path = urls_path.replace("/", os.sep)

        assert urls_path == filepath
        assert urls == mock_json_content

        # fetch_openneuro_dataset_index should do the same, but with a warning
        with pytest.warns(DeprecationWarning):
            urls_path, urls = func.fetch_openneuro_dataset_index(
                data_dir=tmpdir,
                dataset_version=dataset_version,
                verbose=1,
            )

        urls_path = urls_path.replace("/", os.sep)

        assert urls_path == filepath
        assert urls == mock_json_content

        # fetch_openneuro_dataset_index should even grab ds000030 when you
        # provide a different dataset name
        with pytest.warns(
            UserWarning,
            match='"ds000030_R1.0.4" will be downloaded',
        ):
            urls_path, urls = func.fetch_openneuro_dataset_index(
                data_dir=tmpdir,
                dataset_version="ds500_v2",
                verbose=1,
            )

        urls_path = urls_path.replace("/", os.sep)

        assert urls_path == filepath
        assert urls == mock_json_content


def test_fetch_openneuro_dataset(tmp_path):
    dataset_version = "ds000030_R1.0.4"
    data_prefix = (
        f"{dataset_version.split('_')[0]}/{dataset_version}/uncompressed"
    )
    data_dir = _get_dataset_dir(
        data_prefix,
        data_dir=tmp_path,
        verbose=1,
    )
    url_file = os.path.join(data_dir, "urls.json")

    # Prepare url files for subject and filter tests
    urls = [
        f"https://example.com/{data_prefix}/stuff.html",
        f"https://example.com/{data_prefix}/sub-xxx.html",
        f"https://example.com/{data_prefix}/sub-yyy.html",
        f"https://example.com/{data_prefix}/sub-xxx/ses-01_task-rest.txt",
        f"https://example.com/{data_prefix}/sub-xxx/ses-01_task-other.txt",
        f"https://example.com/{data_prefix}/sub-xxx/ses-02_task-rest.txt",
        f"https://example.com/{data_prefix}/sub-xxx/ses-02_task-other.txt",
        f"https://example.com/{data_prefix}/sub-yyy/ses-01.txt",
        f"https://example.com/{data_prefix}/sub-yyy/ses-02.txt",
    ]
    json.dump(urls, open(url_file, "w"))

    # Only 1 subject and not subject specific files get downloaded
    datadir, dl_files = func.fetch_openneuro_dataset(
        urls, tmp_path, dataset_version
    )

    assert isinstance(datadir, str)
    assert isinstance(dl_files, list)
    assert len(dl_files) == 9

    # Try downloading a different dataset without providing URLs
    # This should raise a warning and download ds000030.
    with pytest.warns(
        UserWarning,
        match='Downloading "ds000030_R1.0.4".',
    ):
        _, urls = func.fetch_openneuro_dataset(
            urls=None,
            data_dir=tmp_path,
            dataset_version="ds500_v2",
            verbose=1,
        )


def test_fetch_openneuro_dataset_errors(tmp_path):
    dataset_version = "ds000030_R1.0.4"
    # URLs do not contain the data_prefix, which should raise a ValueError
    urls = [
        "https://example.com/stuff.html",
        "https://example.com/sub-yyy/ses-01.txt",
    ]
    with pytest.raises(ValueError, match="This indicates that the URLs"):
        func.fetch_openneuro_dataset(urls, tmp_path, dataset_version)


def test_fetch_localizer(tmp_path):
    dataset = func.fetch_localizer_first_level(data_dir=tmp_path)

    assert isinstance(dataset["events"], str)
    assert isinstance(dataset.epi_img, str)


def _mock_original_spm_auditory_events_file():
    expected_events_data = {
        "onset": [factor * 42.0 for factor in range(16)],
        "duration": [42.0] * 16,
        "trial_type": ["rest", "active"] * 8,
    }
    expected_events_data = pd.DataFrame(expected_events_data)
    expected_events_data_string = expected_events_data.to_csv(
        sep="\t",
        index=0,
        columns=["onset", "duration", "trial_type"],
    )
    return expected_events_data_string


def _mock_bids_compliant_spm_auditory_events_file():
    events_filepath = os.path.join(os.getcwd(), "tests_events.tsv")
    func._make_events_file_spm_auditory_data(events_filepath=events_filepath)
    actual_events_data_string = Path(events_filepath).read_text()
    return actual_events_data_string, events_filepath


def test_fetch_language_localizer_demo_dataset(tmp_path):
    data_dir = tmp_path
    expected_data_dir = tmp_path / "fMRI-language-localizer-demo-dataset"
    contents_dir = Path(__file__).parent / "data" / "archive_contents"
    contents_list_file = contents_dir / "language_localizer.txt"
    with contents_list_file.open() as f:
        expected_files = [
            str(expected_data_dir / file_path.strip())
            for file_path in f.readlines()[1:]
        ]
    actual_dir, actual_subdirs = func.fetch_language_localizer_demo_dataset(
        data_dir
    )

    assert actual_dir == str(expected_data_dir)
    assert actual_subdirs == sorted(expected_files)


def test_make_spm_auditory_events_file():
    try:
        (
            actual_events_data_string,
            events_filepath,
        ) = _mock_bids_compliant_spm_auditory_events_file()
    finally:
        os.remove(events_filepath)
    expected_events_data_string = _mock_original_spm_auditory_events_file()

    replace_win_line_ends = (
        lambda text: text.replace("\r\n", "\n")
        if text.find("\r\n") != -1
        else text
    )
    actual_events_data_string = replace_win_line_ends(
        actual_events_data_string
    )
    expected_events_data_string = replace_win_line_ends(
        expected_events_data_string
    )

    assert actual_events_data_string == expected_events_data_string


def test_fetch_spm_auditory(tmp_path):
    saf = [f"fM00223/fM00223_{int(index):03}.img" for index in range(4, 100)]
    saf_ = [f"fM00223/fM00223_{int(index):03}.hdr" for index in range(4, 100)]

    data_dir = str(tmp_path / "spm_auditory")
    os.mkdir(data_dir)
    subject_dir = os.path.join(data_dir, "sub001")
    os.mkdir(subject_dir)
    os.mkdir(os.path.join(subject_dir, "fM00223"))
    os.mkdir(os.path.join(subject_dir, "sM00223"))

    path_img = str(tmp_path / "tmp.img")
    path_hdr = str(tmp_path / "tmp.hdr")
    nib.save(nib.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4)), path_img)
    shutil.copy(path_img, os.path.join(subject_dir, "sM00223/sM00223_002.img"))
    shutil.copy(path_hdr, os.path.join(subject_dir, "sM00223/sM00223_002.hdr"))
    for file_ in saf:
        shutil.copy(path_img, os.path.join(subject_dir, file_))
    for file_ in saf_:
        shutil.copy(path_hdr, os.path.join(subject_dir, file_))

    dataset = func.fetch_spm_auditory(data_dir=tmp_path)

    assert isinstance(dataset.anat, str)
    assert isinstance(dataset.func[0], str)
    assert len(dataset.func) == 96


def test_fetch_spm_multimodal(tmp_path):
    data_dir = str(tmp_path / "spm_multimodal_fmri")
    os.mkdir(data_dir)
    subject_dir = os.path.join(data_dir, "sub001")
    os.mkdir(subject_dir)
    os.mkdir(os.path.join(subject_dir, "fMRI"))
    os.mkdir(os.path.join(subject_dir, "sMRI"))
    open(os.path.join(subject_dir, "sMRI", "smri.img"), "a").close()
    for session in [0, 1]:
        open(
            os.path.join(
                subject_dir, "fMRI", f"trials_ses{int(session + 1)}.mat"
            ),
            "a",
        ).close()
        dir_ = os.path.join(subject_dir, "fMRI", f"Session{int(session + 1)}")
        os.mkdir(dir_)
        for i in range(390):
            open(
                os.path.join(
                    dir_, f"fMETHODS-000{int(session + 5)}-{int(i)}-01.img"
                ),
                "a",
            ).close()

    dataset = func.fetch_spm_multimodal_fmri(data_dir=tmp_path)

    assert isinstance(dataset.anat, str)
    assert isinstance(dataset.func1[0], str)
    assert len(dataset.func1) == 390
    assert isinstance(dataset.func2[0], str)
    assert len(dataset.func2) == 390
    assert dataset.slice_order == "descending"
    assert isinstance(dataset.trials_ses1, str)
    assert isinstance(dataset.trials_ses2, str)


def test_fiac(tmp_path):
    # Create dummy 'files'
    fiac_dir = str(
        tmp_path / "fiac_nilearn.glm" / "nipy-data-0.2" / "data" / "fiac"
    )
    fiac0_dir = os.path.join(fiac_dir, "fiac0")
    os.makedirs(fiac0_dir)
    for session in [1, 2]:
        # glob func data for session session + 1
        session_func = os.path.join(fiac0_dir, f"run{int(session)}.nii.gz")
        open(session_func, "a").close()
        sess_dmtx = os.path.join(fiac0_dir, f"run{int(session)}_design.npz")
        open(sess_dmtx, "a").close()
    mask = os.path.join(fiac0_dir, "mask.nii.gz")
    open(mask, "a").close()

    dataset = func.fetch_fiac_first_level(data_dir=tmp_path)

    assert isinstance(dataset.func1, str)
    assert isinstance(dataset.func2, str)
    assert isinstance(dataset.design_matrix1, str)
    assert isinstance(dataset.design_matrix2, str)
    assert isinstance(dataset.mask, str)
