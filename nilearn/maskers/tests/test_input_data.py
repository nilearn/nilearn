

def test_import_from_input_data_with_warning():
    """Tests that importing maskers from deprecated module ``input_data``
    still works.
    """
    from nilearn import input_data, maskers
    assert maskers == input_data
