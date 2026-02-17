"""Read / write access to TCK streamlines format.

TCK format is defined at
http://mrtrix.readthedocs.io/en/latest/getting_started/image_data.html?highlight=format#tracks-file-format-tck
"""

import os
import warnings
from contextlib import suppress

import numpy as np

from nibabel.openers import Opener

from .array_sequence import ArraySequence
from .header import Field
from .tractogram import LazyTractogram, Tractogram, TractogramItem
from .tractogram_file import DataError, DataWarning, HeaderError, HeaderWarning, TractogramFile
from .utils import peek_next

MEGABYTE = 1024 * 1024


class TckFile(TractogramFile):
    """Convenience class to encapsulate TCK file format.

    Notes
    -----
    MRtrix (so its file format: TCK) considers streamlines coordinates
    to be in world space (RAS+ and mm space). MRtrix refers to that space
    as the "real" or "scanner" space [#]_.

    Moreover, when streamlines are mapped back to voxel space [#]_, a
    streamline point located at an integer coordinate (i,j,k) is considered
    to be at the center of the corresponding voxel. This is in contrast with
    TRK's internal convention where it would have referred to a corner.

    NiBabel's streamlines internal representation follows the same
    convention as MRtrix.

    .. [#] http://www.nitrc.org/pipermail/mrtrix-discussion/2014-January/000859.html
    .. [#] http://nipy.org/nibabel/coordinate_systems.html#voxel-coordinates-are-in-voxel-space
    """

    # Constants
    MAGIC_NUMBER = b'mrtrix tracks'
    SUPPORTS_DATA_PER_POINT = False  # Not yet
    SUPPORTS_DATA_PER_STREAMLINE = False  # Not yet

    FIBER_DELIMITER = np.array([[np.nan, np.nan, np.nan]], '<f4')
    EOF_DELIMITER = np.array([[np.inf, np.inf, np.inf]], '<f4')

    def __init__(self, tractogram, header=None):
        """
        Parameters
        ----------
        tractogram : :class:`Tractogram` object
            Tractogram that will be contained in this :class:`TckFile`.
        header : None or dict, optional
            Metadata associated to this tractogram file. If None, make
            default empty header.

        Notes
        -----
        Streamlines of the tractogram are assumed to be in *RAS+* and *mm*
        space. It is also assumed that when streamlines are mapped back to
        voxel space, a streamline point located at an integer coordinate
        (i,j,k) is considered to be at the center of the corresponding voxel.
        This is in contrast with TRK's internal convention where it would
        have referred to a corner.
        """
        super().__init__(tractogram, header)

    @classmethod
    def is_correct_format(cls, fileobj):
        """Check if the file is in TCK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object in
            binary mode pointing to TCK file (and ready to read from the
            beginning of the TCK header). Note that calling this function
            does not change the file position.

        Returns
        -------
        is_correct_format : {True, False}
            Returns True if `fileobj` is compatible with TCK format,
            otherwise returns False.
        """
        with Opener(fileobj) as f:
            magic_number = f.read(len(cls.MAGIC_NUMBER))
            f.seek(-len(cls.MAGIC_NUMBER), os.SEEK_CUR)

        return magic_number == cls.MAGIC_NUMBER

    @classmethod
    def create_empty_header(cls):
        """Return an empty compliant TCK header as dict"""
        header = {}

        # Default values
        header[Field.MAGIC_NUMBER] = cls.MAGIC_NUMBER
        header[Field.NB_STREAMLINES] = 0
        header['datatype'] = 'Float32LE'
        return header

    @classmethod
    def load(cls, fileobj, lazy_load=False):
        """Loads streamlines from a filename or file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object in
            binary mode pointing to TCK file (and ready to read from the
            beginning of the TCK header). Note that calling this function
            does not change the file position.
        lazy_load : {False, True}, optional
            If True, load streamlines in a lazy manner i.e. they will not be
            kept in memory. Otherwise, load all streamlines in memory.

        Returns
        -------
        tck_file : :class:`TckFile` object
            Returns an object containing tractogram data and header
            information.

        Notes
        -----
        Streamlines of the tractogram are assumed to be in *RAS+* and *mm*
        space. It is also assumed that when streamlines are mapped back to
        voxel space, a streamline point located at an integer coordinate
        (i,j,k) is considered to be at the center of the corresponding voxel.
        This is in contrast with TRK's internal convention where it would
        have referred to a corner.
        """
        hdr = cls._read_header(fileobj)

        if lazy_load:

            def _read():
                for pts in cls._read(fileobj, hdr):
                    yield TractogramItem(pts, {}, {})

            tractogram = LazyTractogram.from_data_func(_read)

        else:
            tck_reader = cls._read(fileobj, hdr)
            streamlines = ArraySequence(tck_reader)
            tractogram = Tractogram(streamlines)

        # By definition.
        tractogram.affine_to_rasmm = np.eye(4)
        hdr[Field.VOXEL_TO_RASMM] = np.eye(4)

        return cls(tractogram, header=hdr)

    def _finalize_header(self, f, header, offset=0):
        # Overwrite header with updated one.
        f.seek(offset, os.SEEK_SET)
        self._write_header(f, header)

    def save(self, fileobj):
        """Save tractogram to a filename or file-like object using TCK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object in
            binary mode pointing to TCK file (and ready to write from the
            beginning of the TCK header data).
        """
        # Enforce float32 in little-endian byte order for data.
        dtype = np.dtype('<f4')
        header = self.create_empty_header()

        # Override hdr's fields by those contained in `header`.
        header.update(self.header)

        # Keep counts for correcting incoherent fields or warn.
        nb_streamlines = 0

        with Opener(fileobj, mode='wb') as f:
            # Keep track of the beginning of the header.
            beginning = f.tell()

            # Write temporary header that we will update at the end
            self._write_header(f, header)

            # Make sure streamlines are in rasmm.
            tractogram = self.tractogram.to_world(lazy=True)
            # Assume looping over the streamlines can be done only once.
            tractogram = iter(tractogram)

            try:
                # Use the first element to check
                #  1) the tractogram is not empty;
                #  2) quantity of information saved along each streamline.
                first_item, tractogram = peek_next(tractogram)
            except StopIteration:
                # Empty tractogram
                header[Field.NB_STREAMLINES] = 0
                self._finalize_header(f, header, offset=beginning)

                # Add the EOF_DELIMITER.
                f.write(self.EOF_DELIMITER.tobytes())
                return

            data_for_streamline = first_item.data_for_streamline
            if len(data_for_streamline) > 0:
                keys = ', '.join(data_for_streamline.keys())
                msg = (
                    'TCK format does not support saving additional '
                    f'data alongside streamlines. Dropping: {keys}'
                )
                warnings.warn(msg, DataWarning)

            data_for_points = first_item.data_for_points
            if len(data_for_points) > 0:
                keys = ', '.join(data_for_points.keys())
                msg = (
                    'TCK format does not support saving additional '
                    f'data alongside points. Dropping: {keys}'
                )
                warnings.warn(msg, DataWarning)

            for t in tractogram:
                data = np.r_[t.streamline, self.FIBER_DELIMITER]
                f.write(data.astype(dtype).tobytes())
                nb_streamlines += 1

            header[Field.NB_STREAMLINES] = nb_streamlines

            # Add the EOF_DELIMITER.
            f.write(self.EOF_DELIMITER.tobytes())
            self._finalize_header(f, header, offset=beginning)

    @staticmethod
    def _write_header(fileobj, header):
        """Write TCK header to file-like object.

        Parameters
        ----------
        fileobj : file-like object
            An open file-like object in binary mode pointing to TCK file (and
            ready to read from the beginning of the TCK header).
        """
        # Fields to exclude
        exclude = [
            Field.MAGIC_NUMBER,  # Handled separately.
            Field.NB_STREAMLINES,  # Handled separately.
            Field.ENDIANNESS,  # Handled separately.
            Field.VOXEL_TO_RASMM,  # Streamlines are always in RAS+ mm.
            'count',
            'datatype',
            'file',
        ]  # Fields being replaced.

        lines = [
            f'count: {header[Field.NB_STREAMLINES]:010}',
            'datatype: Float32LE',  # Always Float32LE.
        ]
        lines.extend(
            f'{k}: {v}' for k, v in header.items() if k not in exclude and not k.startswith('_')
        )
        out = '\n'.join(lines)

        if out.count(':') > len(lines):
            # : only one per line (except the last one which contains END).
            msg = f"Key-value pairs cannot contain ':':\n{out}"
            raise HeaderError(msg)

        out = header[Field.MAGIC_NUMBER] + b'\n' + out.encode('utf-8')

        # Compute data offset considering the offset string representation
        # headers + "file" header + END + \n's
        hdr_offset = len(out) + 8 + 3 + 3
        offset_repr = f'{hdr_offset}'

        # Adding the offset may increase one char to the offset repr
        hdr_offset += len(f'{hdr_offset + len(offset_repr)}')

        # Write header to file.
        fileobj.write(out)
        fileobj.write(f'\nfile: . {hdr_offset}\nEND\n'.encode())

    @classmethod
    def _read_header(cls, fileobj):
        """Reads a TCK header from a file.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object in
            binary mode pointing to TCK file (and ready to read from the
            beginning of the TCK header). Note that calling this function
            does not change the file position.

        Returns
        -------
        header : dict
            Metadata associated with this tractogram file.
        """

        # Build header dictionary from the buffer
        hdr = {}
        offset_data = 0

        with Opener(fileobj) as f:
            # Record start position
            start_position = f.tell()

            # Make sure we are at the beginning of the file
            f.seek(0, os.SEEK_SET)

            # Read magic number
            magic_number = f.read(len(cls.MAGIC_NUMBER))

            if magic_number != cls.MAGIC_NUMBER:
                raise HeaderError(f'Invalid magic number: {magic_number}')

            hdr[Field.MAGIC_NUMBER] = magic_number

            f.seek(1, os.SEEK_CUR)  # Skip \n

            found_end = False
            key = None
            tmp_hdr = {}

            # Read all key-value pairs contained in the header, stop at EOF
            for n_line, line in enumerate(f, 1):
                line = line.decode('utf-8').strip()

                if not line:  # Skip empty lines
                    continue

                if line == 'END':  # End of the header
                    found_end = True
                    break

                # Set new key if available, otherwise append to last known key
                with suppress(ValueError):
                    key, line = line.split(':', 1)
                    key = key.strip()

                # Apparent continuation line before any keys are found
                if key is None:
                    raise HeaderError(f'Invalid header (line {n_line}): {line}')

                tmp_hdr.setdefault(key, []).append(line.strip())

            if not found_end:
                raise HeaderError('Missing END in the header.')

            hdr.update({key: '\n'.join(val) for key, val in tmp_hdr.items()})

            offset_data = f.tell()

            # Set the file position where it was, in case it was previously open
            if start_position is not None:
                f.seek(start_position, os.SEEK_SET)

        # Check integrity of TCK header.
        if 'datatype' not in hdr:
            msg = "Missing 'datatype' attribute in TCK header. Assuming it is Float32LE."
            warnings.warn(msg, HeaderWarning)
            hdr['datatype'] = 'Float32LE'

        if not hdr['datatype'].startswith('Float32'):
            msg = (
                f"TCK only supports float32 dtype but 'datatype: {hdr['datatype']}' "
                'was specified in the header.'
            )
            raise HeaderError(msg)

        if 'file' not in hdr:
            msg = "Missing 'file' attribute in TCK header. Will try to guess it."
            warnings.warn(msg, HeaderWarning)
            hdr['file'] = f'. {offset_data}'

        if hdr['file'].split()[0] != '.':
            msg = (
                'TCK only supports single-file - in other words the filename part must be '
                f"specified as '.' but '{hdr['file'].split()[0]}' was specified."
            )
            raise HeaderError("Missing 'file' attribute in TCK header.")

        # Set endianness and _dtype attributes in the header.
        hdr[Field.ENDIANNESS] = '>' if hdr['datatype'].endswith('BE') else '<'

        hdr['_dtype'] = np.dtype(hdr[Field.ENDIANNESS] + 'f4')

        # Keep the file position where the data begin.
        hdr['_offset_data'] = int(hdr['file'].split()[1])

        return hdr

    @classmethod
    def _read(cls, fileobj, header, buffer_size=4):
        """Return generator that reads TCK data from `fileobj` given `header`

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object in
            binary mode pointing to TCK file (and ready to read from the
            beginning of the TCK header). Note that calling this function
            does not change the file position.
        header : dict
            Metadata associated with this tractogram file.
        buffer_size : float, optional
            Size (in Mb) for buffering.

        Yields
        ------
        points : ndarray of shape (n_pts, 3)
            Streamline points
        """
        dtype = header['_dtype']
        coordinate_size = 3 * dtype.itemsize
        # Make buffer_size an integer and a multiple of coordinate_size.
        buffer_size = int(buffer_size * MEGABYTE)
        buffer_size += coordinate_size - (buffer_size % coordinate_size)

        with Opener(fileobj) as f:
            start_position = f.tell()

            # Set the file position at the beginning of the data.
            f.seek(header['_offset_data'], os.SEEK_SET)

            eof = False
            leftover = np.empty((0, 3), dtype='<f4')
            n_streams = 0

            while not eof:
                buff = bytearray(buffer_size)
                n_read = f.readinto(buff)
                eof = n_read != buffer_size
                if eof:
                    buff = buff[:n_read]

                raw_values = np.frombuffer(buff, dtype=dtype)

                # Convert raw_values into a list of little-endian triples (for x,y,z coord)
                coords = raw_values.astype('<f4', copy=False).reshape((-1, 3))

                # Find stream delimiter locations (all NaNs)
                delims = np.where(np.isnan(coords).all(axis=1))[0]

                # Recover leftovers, which can't have delimiters in them
                if leftover.size:
                    delims += leftover.shape[0]
                    coords = np.vstack((leftover, coords))

                begin = 0
                for delim in delims:
                    pts = coords[begin:delim]
                    if pts.size:
                        yield pts
                        n_streams += 1
                    begin = delim + 1

                # The rest becomes the new leftover.
                leftover = coords[begin:]

            if not (leftover.shape == (1, 3) and np.isinf(leftover).all()):
                if n_streams == 0:
                    msg = 'Cannot find a streamline delimiter. This file might be corrupted.'
                else:
                    msg = "Expecting end-of-file marker 'inf inf inf'"
                raise DataError(msg)

            # In case the 'count' field was not provided.
            header[Field.NB_STREAMLINES] = n_streams

            # Set the file position where it was (in case it was already open).
            f.seek(start_position, os.SEEK_CUR)

    def __str__(self):
        """Gets a formatted string of the header of a TCK file.

        Returns
        -------
        info : string
            Header information relevant to the TCK format.
        """
        hdr = self.header

        info = ''
        info += f'\nMAGIC NUMBER: {hdr[Field.MAGIC_NUMBER]}'
        info += '\n'
        info += '\n'.join(f'{k}: {v}' for k, v in hdr.items() if not k.startswith('_'))
        return info
