"""Test tractogramFile base class"""

import pytest

from ..tractogram import Tractogram
from ..tractogram_file import TractogramFile


def test_subclassing_tractogram_file():
    # Missing 'save' method
    class DummyTractogramFile(TractogramFile):
        @classmethod
        def is_correct_format(cls, fileobj):
            return False

        @classmethod
        def load(cls, fileobj, lazy_load=True):
            return None

        @classmethod
        def create_empty_header(cls):
            return None

    with pytest.raises(TypeError):
        DummyTractogramFile(Tractogram())

    # Missing 'load' method
    class DummyTractogramFile(TractogramFile):
        @classmethod
        def is_correct_format(cls, fileobj):
            return False

        def save(self, fileobj):
            pass

        @classmethod
        def create_empty_header(cls):
            return None

    with pytest.raises(TypeError):
        DummyTractogramFile(Tractogram())

    # Now we have everything required.
    class DummyTractogramFile(TractogramFile):
        @classmethod
        def is_correct_format(cls, fileobj):
            return False

        @classmethod
        def load(cls, fileobj, lazy_load=True):
            return None

        def save(self, fileobj):
            pass

    # No error
    dtf = DummyTractogramFile(Tractogram())

    # Default create_empty_header is empty dict
    assert dtf.header == {}


def test_tractogram_file():
    with pytest.raises(NotImplementedError):
        TractogramFile.is_correct_format('')
    with pytest.raises(NotImplementedError):
        TractogramFile.load('')

    # Testing calling the 'save' method of `TractogramFile` object.
    class DummyTractogramFile(TractogramFile):
        @classmethod
        def is_correct_format(cls, fileobj):
            return False

        @classmethod
        def load(cls, fileobj, lazy_load=True):
            return None

        @classmethod
        def save(self, fileobj):
            pass

    with pytest.raises(NotImplementedError):
        super(DummyTractogramFile, DummyTractogramFile(Tractogram)).save('')
