import os

from nose.tools import assert_equal
import pandas as pd

from nistats.datasets import _make_events_file_spm_auditory


def create_expected_data():
    expected_events_data = {
        'onset': [factor * 42.0 for factor in range(0, 16)],
        'duration': [42.0] * 16,
        'trial_type': ['rest', 'active'] * 8,
                            }
    expected_events_data  = pd.DataFrame(expected_events_data )
    expected_events_data_string = expected_events_data.to_csv(sep='\t', index=0, columns=['onset', 'duration', 'trial_type'])
    return expected_events_data_string


def create_actual_data():
    events_filepath = os.path.join(os.getcwd(), 'tests_events.tsv')
    _make_events_file_spm_auditory(events_filepath=events_filepath)
    with open(events_filepath , 'r') as actual_events_file_obj:
        actual_events_data_string = actual_events_file_obj.read()
    return actual_events_data_string, events_filepath
    
    
def run_test():
    try:
        actual_events_data_string, events_filepath = create_actual_data()
    finally:
        os.remove(events_filepath)
    expected_events_data_string = create_expected_data()
    assert_equal(actual_events_data_string, expected_events_data_string)
    
    
if __name__ == '__main__':
    run_test()
    


    
    

