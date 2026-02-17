import pytest

from ..spatialimages import supported_np_types


# Generate dynamic fixtures
def pytest_generate_tests(metafunc):
    if 'supported_dtype' in metafunc.fixturenames:
        if metafunc.cls is None or not metafunc.cls.image_class:
            raise pytest.UsageError(
                'Attempting to use supported_dtype fixture outside an image test case'
            )
        # xdist needs a consistent ordering, so sort by class name
        supported_dtypes = sorted(
            supported_np_types(metafunc.cls.image_class.header_class()),
            key=lambda x: x.__name__,
        )
        metafunc.parametrize('supported_dtype', supported_dtypes)
