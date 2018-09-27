import csv
import os
from tempfile import NamedTemporaryFile

import pandas as pd
from nose.tools import (assert_raises,
                        assert_true,
                        )

from nistats.utils import _verify_events_file_uses_tab_separators


def make_data_for_test_runs():
    data_for_temp_datafile = [
        ['csf', 'constant', 'linearTrend', 'wm'],
        [13343.032102491035, 1.0, 0.0, 9486.199545677482],
        [13329.224068063204, 1.0, 1.0, 9497.003324892803],
        [13291.755627241291, 1.0, 2.0, 9484.012965365506],
        ]
    
    delimiters = {
        'tab': '\t',
        'comma': ',',
        'space': ' ',
        'semicolon': ';',
        'hyphen': '-',
        }
    
    return data_for_temp_datafile, delimiters


def _create_test_file(temp_csv, test_data, delimiter):
    csv_writer = csv.writer(temp_csv, delimiter=delimiter)
    for row in test_data:
        csv_writer.writerow(row)
    temp_csv.flush()


def _run_test_for_invalid_separator(filepath, delimiter_name):
    if delimiter_name not in ('tab', 'comma'):
        with assert_raises(ValueError):
            _verify_events_file_uses_tab_separators(events_files=filepath)
    else:
        result = _verify_events_file_uses_tab_separators(events_files=filepath)
        assert_true(result == [])


def test_for_invalid_separator():
    data_for_temp_datafile, delimiters = make_data_for_test_runs()
    for delimiter_name, delimiter_char in delimiters.items():
        tmp_file_prefix, temp_file_suffix = (
            'tmp ', ' ' + delimiter_name + '.csv')
        with NamedTemporaryFile(mode='w', dir=os.getcwd(),
                                prefix=tmp_file_prefix,
                                suffix=temp_file_suffix) as temp_csv_obj:
            _create_test_file(temp_csv=temp_csv_obj,
                              test_data=data_for_temp_datafile,
                              delimiter=delimiter_char)
            _run_test_for_invalid_separator(filepath=temp_csv_obj.name,
                                            delimiter_name=delimiter_name)


def test_with_2D_dataframe():
    data_for_pandas_dataframe, _ = make_data_for_test_runs()
    events_pandas_dataframe = pd.DataFrame(data_for_pandas_dataframe)
    result = _verify_events_file_uses_tab_separators(
            events_files=events_pandas_dataframe)
    expected_error = result[0][1]
    with assert_raises(TypeError):
        raise expected_error


def test_with_1D_dataframe():
    data_for_pandas_dataframe, _ = make_data_for_test_runs()
    for dataframe_ in data_for_pandas_dataframe:
        events_pandas_dataframe = pd.DataFrame(dataframe_)
        result = _verify_events_file_uses_tab_separators(
                events_files=events_pandas_dataframe)
        expected_error = result[0][1]
        with assert_raises(TypeError):
            raise expected_error


def test_for_invalid_filepath():
    filepath = 'junk_file_path.csv'
    result = _verify_events_file_uses_tab_separators(events_files=filepath)
    expected_error = result[0][1]
    with assert_raises(IOError):
        raise expected_error


def test_for_pandas_dataframe():
    events_pandas_dataframe = pd.DataFrame([['a', 'b', 'c'], [0, 1, 2]])
    result = _verify_events_file_uses_tab_separators(
            events_files=events_pandas_dataframe)
    expected_error = result[0][1]
    with assert_raises(TypeError):
        raise expected_error


def test_binary_opening_an_image():
    img_data = bytearray(
            b'GIF87a\x01\x00\x01\x00\xe7*\x00\x00\x00\x00\x01\x01\x01\x02\x02'
            b'\x07\x08\x08\x08\t\t\t\n\n\n\x0b\x0b\x0b\x0c\x0c\x0c\r;')
    with NamedTemporaryFile(mode='wb', suffix='.gif',
                            dir=os.getcwd()) as temp_img_obj:
        temp_img_obj.write(img_data)
        with assert_raises(ValueError):
            _verify_events_file_uses_tab_separators(
                    events_files=temp_img_obj.name)


def test_binary_bytearray_of_ints_data():
    temp_data_bytearray_from_ints = bytearray([0, 1, 0, 11, 10])
    with NamedTemporaryFile(mode='wb', dir=os.getcwd(),
                            suffix='.bin') as temp_bin_obj:
        temp_bin_obj.write(temp_data_bytearray_from_ints)
        with assert_raises(ValueError):
            result = _verify_events_file_uses_tab_separators(
                    events_files=temp_bin_obj.name)


if __name__ == '__main__':

    def _run_tests_print_test_messages(test_func):
        from pprint import pprint
        pprint(['Running', test_func.__name__])
        test_func()
        pprint('... complete')
    
    
    def run_test_suite():
        tests = [
            test_for_invalid_filepath,
            test_with_2D_dataframe,
            test_with_1D_dataframe,
            test_for_invalid_filepath,
            test_for_pandas_dataframe,
            test_binary_opening_an_image,
            test_binary_bytearray_of_ints_data,
            ]
    
    for test_ in tests:
        _run_tests_print_test_messages(test_func=test_)



    run_test_suite()
