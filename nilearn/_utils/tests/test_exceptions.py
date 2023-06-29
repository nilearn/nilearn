import re

from nilearn._utils.exceptions import DimensionError


def test_dimension_error_message():
    error = DimensionError(file_dimension=3, required_dimension=5)
    error.increment_stack_counter()
    error.increment_stack_counter()
    assert re.match("^.*7D.*list of list of 3D images.*5D.*$", error.message)
