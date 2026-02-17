"""Testing fileholders"""

from io import BytesIO

from ..fileholders import FileHolder


def test_init():
    fh = FileHolder('a_fname')
    assert fh.filename == 'a_fname'
    assert fh.fileobj is None
    assert fh.pos == 0
    sio0 = BytesIO()
    fh = FileHolder('a_test', sio0)
    assert fh.filename == 'a_test'
    assert fh.fileobj is sio0
    assert fh.pos == 0
    fh = FileHolder('a_test_2', sio0, 3)
    assert fh.filename == 'a_test_2'
    assert fh.fileobj is sio0
    assert fh.pos == 3


def test_same_file_as():
    fh = FileHolder('a_fname')
    assert fh.same_file_as(fh)
    fh2 = FileHolder('a_test')
    assert not fh.same_file_as(fh2)
    sio0 = BytesIO()
    fh3 = FileHolder('a_fname', sio0)
    fh4 = FileHolder('a_fname', sio0)
    assert fh3.same_file_as(fh4)
    assert not fh3.same_file_as(fh)
    fh5 = FileHolder(fileobj=sio0)
    fh6 = FileHolder(fileobj=sio0)
    assert fh5.same_file_as(fh6)
    # Not if the filename is the same
    assert not fh5.same_file_as(fh3)
    # pos doesn't matter
    fh4_again = FileHolder('a_fname', sio0, pos=4)
    assert fh3.same_file_as(fh4_again)


def test_file_like():
    # Test returning file object or filename
    fh = FileHolder('a_fname')
    assert fh.file_like == 'a_fname'
    bio = BytesIO()
    fh = FileHolder(fileobj=bio)
    assert fh.file_like is bio
    fh = FileHolder('a_fname', fileobj=bio)
    assert fh.file_like is bio
