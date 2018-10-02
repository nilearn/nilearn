import os
from tempfile import NamedTemporaryFile

from nose.tools import assert_true
import pandas as pd
from nistats.datasets import _make_bids_compliant_localizer_first_level_paradigm_file

def _input_data_for_test_file():
    file_data = [
        [0, 'calculvideo', 0.0],
        [0, 'calculvideo', 2.400000095],
        [0, 'damier_H', 8.699999809],
        [0, 'clicDaudio', 11.39999961],
    ]
    return pd.DataFrame(file_data)
    

def _expected_output_data_from_test_file():
    file_data = [
        ['calculvideo', 0.0],
        ['calculvideo', 2.400000095],
        ['damier_H', 8.699999809],
        ['clicDaudio', 11.39999961],
    ]
    file_data = pd.DataFrame(file_data)
    file_data.columns = ['trial_type', 'onset']
    return file_data


def run_test():
    data_for_tests = _input_data_for_test_file()
    expected_data_from_test_file = _expected_output_data_from_test_file()
    with NamedTemporaryFile(mode='w',
                            dir=os.getcwd(),
                            suffix='.csv') as temp_csv_obj:
        data_for_tests.to_csv(temp_csv_obj.name,
                              index=False,
                              header=False,
                              sep=' ',
                              )
        _make_bids_compliant_localizer_first_level_paradigm_file(
                temp_csv_obj.name
                )
        data_from_test_file_post_mod = pd.read_csv(temp_csv_obj.name, sep='\t')
        assert_true(all(
                expected_data_from_test_file ==  data_from_test_file_post_mod
                )
                )
        
        
if __name__ == '__main__':
    run_test()
