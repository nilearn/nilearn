from functools import cached_property

from nibabel.onetime import ResetMixin, setattr_on_read
from nibabel.testing import deprecated_to, expires


class A(ResetMixin):
    @cached_property
    def y(self):
        return self.x / 2.0

    @cached_property
    def z(self):
        return self.x / 3.0

    def __init__(self, x=1.0):
        self.x = x


@expires('5.0.0')
def test_setattr_on_read():
    with deprecated_to('5.0.0'):

        class MagicProp:
            @setattr_on_read
            def a(self):
                return object()

    x = MagicProp()
    assert 'a' not in x.__dict__
    obj = x.a
    assert 'a' in x.__dict__
    # Each call to object() produces a unique object. Verify we get the same one every time.
    assert x.a is obj


def test_ResetMixin():
    a = A(10)
    assert 'y' not in a.__dict__
    assert a.y == 5
    assert 'y' in a.__dict__
    a.x = 20
    assert a.y == 5
    # Call reset and no error should be raised even though z was never accessed
    a.reset()
    assert 'y' not in a.__dict__
    assert a.y == 10
