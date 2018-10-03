import os

from nose.tools import assert_equal
import pandas as pd

from nistats.datasets import _make_spm_auditory_events_file


def create_expected_data():
    expected_filename = 'tests_events.tsv'
    expected_events_data = {
        'onset': [factor * 42.0 for factor in range(0, 16)],
        'duration': [42.0] * 16,
        'trial_type': ['rest', 'active'] * 8,
                            }
    expected_events_data  = pd.DataFrame(expected_events_data )
    expected_events_data_string = expected_events_data.to_csv(sep='\t', index=0, columns=['onset', 'duration', 'trial_type'])
    return expected_events_data_string, expected_filename


def create_actual_data():
    events_filepath = _make_spm_auditory_events_file(events_file_location=
                                                     os.getcwd()
                                                     )
    events_filename = os.path.basename(events_filepath)
    with open(events_filepath , 'r') as actual_events_file_obj:
        actual_events_data_string = actual_events_file_obj.read()
    return actual_events_data_string, events_filename, events_filepath
    
    
def run_test():
    try:
        expected_events_data_string, expected_filename = create_expected_data()
        actual_events_data_string, actual_filename, events_filepath = create_actual_data()
        assert_equal(actual_filename, expected_filename)
        assert_equal(actual_events_data_string, expected_events_data_string)
    finally:
        os.remove(events_filepath)
    
if __name__ == '__main__':
    run_test()
    


    
    

