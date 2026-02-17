"""Testing filebasedimages module"""

import warnings
from itertools import product

import numpy as np
import pytest

from ..filebasedimages import FileBasedHeader, FileBasedImage, SerializableImage
from .test_image_api import GenericImageAPI, SerializeMixin


class FBNumpyImage(FileBasedImage):
    header_class = FileBasedHeader
    valid_exts = ('.npy',)
    files_types = (('image', '.npy'),)

    def __init__(self, arr, header=None, extra=None, file_map=None):
        super().__init__(header, extra, file_map)
        self.arr = arr

    @property
    def shape(self):
        return self.arr.shape

    def get_data(self):
        warnings.warn('Deprecated', DeprecationWarning)
        return self.arr

    @property
    def dataobj(self):
        return self.arr

    def get_fdata(self):
        return self.arr.astype(np.float64)

    @classmethod
    def from_file_map(klass, file_map):
        with file_map['image'].get_prepare_fileobj('rb') as fobj:
            arr = np.load(fobj)
        return klass(arr)

    def to_file_map(self, file_map=None):
        file_map = self.file_map if file_map is None else file_map
        with file_map['image'].get_prepare_fileobj('wb') as fobj:
            np.save(fobj, self.arr)

    def get_data_dtype(self):
        return self.arr.dtype

    def set_data_dtype(self, dtype):
        self.arr = self.arr.astype(dtype)


class SerializableNumpyImage(FBNumpyImage, SerializableImage):
    pass


class TestFBImageAPI(GenericImageAPI):
    """Validation for FileBasedImage instances"""

    # A callable returning an image from ``image_maker(data, header)``
    image_maker = FBNumpyImage
    # A callable returning a header from ``header_maker()``
    header_maker = FileBasedHeader
    # Example shapes for created images
    example_shapes = ((2,), (2, 3), (2, 3, 4), (2, 3, 4, 5))
    example_dtypes = (np.int8, np.uint16, np.int32, np.float32)
    can_save = True
    standard_extension = '.npy'

    def make_imaker(self, arr, header=None):
        return lambda: self.image_maker(arr, header)

    def obj_params(self):
        # Create new images
        for shape, dtype in product(self.example_shapes, self.example_dtypes):
            arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
            hdr = self.header_maker()
            func = self.make_imaker(arr.copy(), hdr)
            params = dict(dtype=dtype, data=arr, shape=shape, is_proxy=False)
            yield func, params


class TestSerializableImageAPI(TestFBImageAPI, SerializeMixin):
    image_maker = SerializableNumpyImage

    @staticmethod
    def _header_eq(header_a, header_b):
        """FileBasedHeader is an abstract class, so __eq__ is undefined.
        Checking for the same header type is sufficient, here."""
        return type(header_a) == type(header_b) == FileBasedHeader


def test_filebased_header():
    # Test stuff about the default FileBasedHeader

    class H(FileBasedHeader):
        def __init__(self, seq=None):
            if seq is None:
                seq = []
            self.a_list = list(seq)

    in_list = [1, 3, 2]
    hdr = H(in_list)
    hdr_c = hdr.copy()
    assert hdr_c.a_list == hdr.a_list
    # Copy is independent of original
    hdr_c.a_list[0] = 99
    assert hdr_c.a_list != hdr.a_list
    # From header does a copy
    hdr2 = H.from_header(hdr)
    assert isinstance(hdr2, H)
    assert hdr2.a_list == hdr.a_list
    hdr2.a_list[0] = 42
    assert hdr2.a_list != hdr.a_list
    # Default header input to from_heder gives new empty header
    hdr3 = H.from_header()
    assert isinstance(hdr3, H)
    assert hdr3.a_list == []
    hdr4 = H.from_header(None)
    assert isinstance(hdr4, H)
    assert hdr4.a_list == []


class MultipartNumpyImage(FBNumpyImage):
    # We won't actually try to write these out, just need to test an edge case
    files_types = (('header', '.hdr'), ('image', '.npy'))


class SerializableMPNumpyImage(MultipartNumpyImage, SerializableImage):
    pass


def test_multifile_stream_failure():
    shape = (2, 3, 4)
    arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    img = SerializableMPNumpyImage(arr)
    with pytest.raises(NotImplementedError):
        img.to_bytes()
    img = SerializableNumpyImage(arr)
    bstr = img.to_bytes()
    with pytest.raises(NotImplementedError):
        SerializableMPNumpyImage.from_bytes(bstr)
