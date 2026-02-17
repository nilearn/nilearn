"""Testing tripwire module"""

import pytest

from ..tripwire import TripWire, TripWireError, is_tripwire


def test_is_tripwire():
    assert not is_tripwire(object())
    assert is_tripwire(TripWire('some message'))


def test_tripwire():
    # Test tripwire object
    silly_module_name = TripWire('We do not have silly_module_name')
    with pytest.raises(TripWireError):
        silly_module_name.do_silly_thing
    # Check AttributeError can be checked too
    with pytest.raises(AttributeError):
        silly_module_name.__wrapped__
