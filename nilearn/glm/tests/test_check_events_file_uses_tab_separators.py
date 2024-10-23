import pandas as pd
import pytest

from nilearn.glm.first_level.first_level import (
    _check_events_file_uses_tab_separators,
)


def make_data_for_test_runs():
    data_for_temp_datafile = [
        ["csf", "constant", "linearTrend", "wm"],
        [13343.032102491035, 1.0, 0.0, 9486.199545677482],
        [13329.224068063204, 1.0, 1.0, 9497.003324892803],
        [13291.755627241291, 1.0, 2.0, 9484.012965365506],
    ]

    delimiters = {
        "tab": "\t",
        "comma": ",",
        "space": " ",
        "semicolon": ";",
        "hyphen": "-",
    }

    return data_for_temp_datafile, delimiters


def _create_test_file(temp_csv, test_data, delimiter):
    test_data = pd.DataFrame(test_data)
    test_data.to_csv(temp_csv, sep=delimiter)


def _run_test_for_invalid_separator(filepath, delimiter_name):
    if delimiter_name not in ("tab", "comma"):
        with pytest.raises(
            ValueError,
            match="The values in the events file are not separated by tabs",
        ):
            _check_events_file_uses_tab_separators(events_files=filepath)
    else:
        result = _check_events_file_uses_tab_separators(events_files=filepath)
        assert result is None


def test_for_invalid_separator(tmp_path):
    data_for_temp_datafile, delimiters = make_data_for_test_runs()
    for delimiter_name, delimiter_char in delimiters.items():
        temp_tsv_file = (
            tmp_path / f"tempfile.{delimiter_name} separated values"
        )
        _create_test_file(
            temp_csv=temp_tsv_file,
            test_data=data_for_temp_datafile,
            delimiter=delimiter_char,
        )
        _run_test_for_invalid_separator(
            filepath=temp_tsv_file, delimiter_name=delimiter_name
        )


def test_with_2d_dataframe():
    data_for_pandas_dataframe, _ = make_data_for_test_runs()
    events_pandas_dataframe = pd.DataFrame(data_for_pandas_dataframe)
    result = _check_events_file_uses_tab_separators(
        events_files=events_pandas_dataframe
    )
    assert result is None


def test_with_1d_dataframe():
    data_for_pandas_dataframe, _ = make_data_for_test_runs()
    for dataframe_ in data_for_pandas_dataframe:
        events_pandas_dataframe = pd.DataFrame(dataframe_)
        result = _check_events_file_uses_tab_separators(
            events_files=events_pandas_dataframe
        )
        assert result is None


def test_for_invalid_filepath():
    filepath = "junk_file_path.csv"
    result = _check_events_file_uses_tab_separators(events_files=filepath)
    assert result is None


def test_for_pandas_dataframe():
    events_pandas_dataframe = pd.DataFrame([["a", "b", "c"], [0, 1, 2]])
    result = _check_events_file_uses_tab_separators(
        events_files=events_pandas_dataframe
    )
    assert result is None


def test_binary_bytearray_of_ints_data_error(tmp_path):
    temp_data_bytearray_from_ints = bytearray([0, 1, 0, 11, 10])
    temp_bin_file = tmp_path / "temp_bin.bin"
    with temp_bin_file.open("wb") as temp_bin_obj:
        temp_bin_obj.write(temp_data_bytearray_from_ints)
    with pytest.raises(
        ValueError,
        match="The values in the events file are not separated by tabs",
    ):
        _check_events_file_uses_tab_separators(events_files=temp_bin_file)
