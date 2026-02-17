# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Read MINC1 format images"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from .externals.netcdf import netcdf_file
from .fileslice import canonical_slicers
from .spatialimages import SpatialHeader, SpatialImage

_dt_dict = {
    ('b', 'unsigned'): np.uint8,
    ('b', 'signed__'): np.int8,
    ('c', 'unsigned'): 'S1',
    ('h', 'unsigned'): np.uint16,
    ('h', 'signed__'): np.int16,
    ('i', 'unsigned'): np.uint32,
    ('i', 'signed__'): np.int32,
}

# See
# https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#MINC_specific_convenience_functions
_default_dir_cos = {'xspace': [1, 0, 0], 'yspace': [0, 1, 0], 'zspace': [0, 0, 1]}


class MincError(Exception):
    """Error when reading MINC files"""


class Minc1File:
    """Class to wrap MINC1 format opened netcdf object

    Although it has some of the same methods as a ``Header``, we use
    this only when reading a MINC file, to pull out useful header
    information, and for the method of reading the data out
    """

    def __init__(self, mincfile):
        self._mincfile = mincfile
        self._image = mincfile.variables['image']
        self._dim_names = self._image.dimensions
        # The code below will error with vector_dimensions.  See:
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#An_Introduction_to_NetCDF
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#Image_dimensions
        self._dims = [self._mincfile.variables[s] for s in self._dim_names]
        # We don't currently support irregular spacing
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#MINC_specific_convenience_functions
        for dim in self._dims:
            if dim.spacing != b'regular__':
                raise ValueError('Irregular spacing not supported')
        self._spatial_dims = [name for name in self._dim_names if name.endswith('space')]
        # the MINC standard appears to allow the following variables to
        # be undefined.
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#Image_conversion_variables
        # It wasn't immediately obvious what the defaults were.
        self._image_max = self._mincfile.variables['image-max']
        self._image_min = self._mincfile.variables['image-min']

    def _get_dimensions(self, var):
        # Dimensions for a particular variable
        # Differs for MINC1 and MINC2 - see:
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference#Associating_HDF5_dataspaces_with_MINC_dimensions
        return var.dimensions

    def get_data_dtype(self):
        typecode = self._image.typecode()
        if typecode == 'f':
            dtt = np.dtype(np.float32)
        elif typecode == 'd':
            dtt = np.dtype(np.float64)
        else:
            signtype = self._image.signtype.decode('latin-1')
            dtt = _dt_dict[(typecode, signtype)]
        return np.dtype(dtt).newbyteorder('>')

    def get_data_shape(self):
        return self._image.data.shape

    def get_zooms(self):
        """Get real-world sizes of voxels"""
        # zooms must be positive; but steps in MINC can be negative
        return tuple(abs(float(dim.step)) if hasattr(dim, 'step') else 1.0 for dim in self._dims)

    def get_affine(self):
        nspatial = len(self._spatial_dims)
        rot_mat = np.eye(nspatial)
        steps = np.zeros((nspatial,))
        starts = np.zeros((nspatial,))
        dim_names = list(self._dim_names)  # for indexing in loop
        for i, name in enumerate(self._spatial_dims):
            dim = self._dims[dim_names.index(name)]
            rot_mat[:, i] = (
                dim.direction_cosines
                if hasattr(dim, 'direction_cosines')
                else _default_dir_cos[name]
            )
            steps[i] = dim.step if hasattr(dim, 'step') else 1.0
            starts[i] = dim.start if hasattr(dim, 'start') else 0.0
        origin = np.dot(rot_mat, starts)
        aff = np.eye(nspatial + 1)
        aff[:nspatial, :nspatial] = rot_mat * steps
        aff[:nspatial, nspatial] = origin
        return aff

    def _get_valid_range(self):
        """Return valid range for image data

        The valid range can come from the image 'valid_range' or
        image 'valid_min' and 'valid_max', or, failing that, from the
        data type range
        """
        ddt = self.get_data_dtype()
        info = np.iinfo(ddt.type)
        try:
            valid_range = self._image.valid_range
        except AttributeError:
            try:
                valid_range = [self._image.valid_min, self._image.valid_max]
            except AttributeError:
                valid_range = [info.min, info.max]
        if valid_range[0] < info.min or valid_range[1] > info.max:
            raise ValueError('Valid range outside input data type range')
        return np.asarray(valid_range, dtype=np.float64)

    def _get_scalar(self, var):
        """Get scalar value from NetCDF scalar"""
        return var.getValue()

    def _get_array(self, var):
        """Get array from NetCDF array"""
        return var.data

    def _normalize(self, data, sliceobj=()):
        """Apply scaling to image data `data` already sliced with `sliceobj`

        https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#Pixel_values_and_real_values

        MINC normalization uses "image-min" and "image-max" variables to
        map the data from the valid range of the image to the range
        specified by "image-min" and "image-max".

        The "image-max" and "image-min" are variables that describe the
        "max" and "min" of image over some dimensions of "image".

        The usual case is that "image" has dimensions ["zspace", "yspace",
        "xspace"] and "image-max" has dimensions ["zspace"], but there can be
        up to two dimensions for over which scaling is specified.

        Parameters
        ----------
        data : ndarray
            data after applying `sliceobj` slicing to full image
        sliceobj : tuple, optional
            slice definition. If not specified, assume no slicing has been
            applied to `data`
        """
        ddt = self.get_data_dtype()
        if np.issubdtype(ddt.type, np.floating):
            return data
        image_max = self._image_max
        image_min = self._image_min
        mx_dims = self._get_dimensions(image_max)
        mn_dims = self._get_dimensions(image_min)
        if mx_dims != mn_dims:
            raise MincError('"image-max" and "image-min" do not have the same dimensions')
        nscales = len(mx_dims)
        if nscales > 2:
            raise MincError('More than two scaling dimensions')
        if mx_dims != self._dim_names[:nscales]:
            raise MincError('image-max and image dimensions do not match')
        dmin, dmax = self._get_valid_range()
        out_data = np.clip(data, dmin, dmax)
        if nscales == 0:  # scalar values
            imax = self._get_scalar(image_max)
            imin = self._get_scalar(image_min)
        else:  # 1D or 2D array of scaling values
            # We need to get the correct values from image-max and image-min to
            # do the scaling.
            shape = self.get_data_shape()
            sliceobj = canonical_slicers(sliceobj, shape)
            # Indices into sliceobj referring to image axes
            ax_inds = [i for i, obj in enumerate(sliceobj) if obj is not None]
            assert len(ax_inds) == len(shape)
            # Slice imax, imin using same slicer as for data
            nscales_ax = ax_inds[nscales]
            i_slicer = sliceobj[:nscales_ax]
            # Fill slicer to broadcast against sliced data; add length 1 axis
            # for each axis except int axes (which are dropped by slicing)
            broad_part = tuple(
                None for s in sliceobj[ax_inds[nscales] :] if not isinstance(s, Integral)
            )
            i_slicer += broad_part
            imax = self._get_array(image_max)[i_slicer]
            imin = self._get_array(image_min)[i_slicer]
        slope = (imax - imin) / (dmax - dmin)
        inter = imin - dmin * slope
        out_data *= slope
        out_data += inter
        return out_data

    def get_scaled_data(self, sliceobj=()):
        """Return scaled data for slice definition `sliceobj`

        Parameters
        ----------
        sliceobj : tuple, optional
            slice definition. If not specified, return whole array

        Returns
        -------
        scaled_arr : array
            array from minc file with scaling applied
        """
        if sliceobj == ():
            raw_data = self._image.data
        else:
            raw_data = self._image.data[sliceobj]
        dtype = self.get_data_dtype()
        data = np.asarray(raw_data).view(dtype)
        return self._normalize(data, sliceobj)


class MincImageArrayProxy:
    """MINC implementation of array proxy protocol

    The array proxy allows us to freeze the passed fileobj and
    header such that it returns the expected data array.
    """

    def __init__(self, minc_file):
        self.minc_file = minc_file
        self._shape = minc_file.get_data_shape()

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_proxy(self):
        return True

    def __array__(self, dtype=None):
        """Read data from file and apply scaling, casting to ``dtype``

        If ``dtype`` is unspecified, the dtype is automatically determined.

        Parameters
        ----------
        dtype : numpy dtype specifier, optional
            A numpy dtype specifier specifying the type of the returned array.

        Returns
        -------
        array
            Scaled image data with type `dtype`.
        """
        arr = self.minc_file.get_scaled_data(sliceobj=())
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __getitem__(self, sliceobj):
        """Read slice `sliceobj` of data from file"""
        return self.minc_file.get_scaled_data(sliceobj)


class MincHeader(SpatialHeader):
    """Class to contain header for MINC formats"""

    # We don't use the data layout - this just in case we do later
    data_layout = 'C'

    def data_to_fileobj(self, data, fileobj, rescale=True):
        """See Header class for an implementation we can't use"""
        raise NotImplementedError

    def data_from_fileobj(self, fileobj):
        """See Header class for an implementation we can't use"""
        raise NotImplementedError


class Minc1Header(MincHeader):
    @classmethod
    def may_contain_header(klass, binaryblock):
        return binaryblock[:4] == b'CDF\x01'


class Minc1Image(SpatialImage):
    """Class for MINC1 format images

    The MINC1 image class uses the default header type, rather than a specific
    MINC header type - and reads the relevant information from the MINC file on
    load.
    """

    header_class: type[MincHeader] = Minc1Header
    header: MincHeader
    _meta_sniff_len: int = 4
    valid_exts: tuple[str, ...] = ('.mnc',)
    files_types: tuple[tuple[str, str], ...] = (('image', '.mnc'),)
    _compressed_suffixes: tuple[str, ...] = ('.gz', '.bz2', '.zst')

    makeable = True
    rw = False

    ImageArrayProxy = MincImageArrayProxy

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        # Note that mmap and keep_file_open are included for proper
        with file_map['image'].get_prepare_fileobj() as fobj:
            minc_file = Minc1File(netcdf_file(fobj))
            affine = minc_file.get_affine()
            if affine.shape != (4, 4):
                raise MincError('Image does not have 3 spatial dimensions')
            data_dtype = minc_file.get_data_dtype()
            shape = minc_file.get_data_shape()
            zooms = minc_file.get_zooms()
            header = klass.header_class(data_dtype, shape, zooms)
            data = klass.ImageArrayProxy(minc_file)
        return klass(data, affine, header, extra=None, file_map=file_map)


load = Minc1Image.from_filename
