
import pytest


def test_warning_import_from_input_data():
    """Tests that importing maskers from deprecated module
    input_data works but gives a warning.
    """
    with pytest.warns(UserWarning,
                      match=("The module 'input_data' is deprecated "
                             "since 0.8.2. Please import maskers from "
                             "the 'maskers' module.")):
        from nilearn.input_data import NiftiMasker  # noqa
