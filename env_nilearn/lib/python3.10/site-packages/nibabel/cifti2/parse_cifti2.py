# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from io import BytesIO

import numpy as np
from packaging.version import Version, parse

from .. import xmlutils as xml
from ..batteryrunners import Report
from ..nifti1 import Nifti1Extension, extension_codes, intent_codes
from ..nifti2 import Nifti2Header, Nifti2Image
from ..spatialimages import HeaderDataError
from .cifti2 import (
    CIFTI_BRAIN_STRUCTURES,
    CIFTI_MODEL_TYPES,
    Cifti2BrainModel,
    Cifti2Header,
    Cifti2HeaderError,
    Cifti2Label,
    Cifti2LabelTable,
    Cifti2Matrix,
    Cifti2MatrixIndicesMap,
    Cifti2MetaData,
    Cifti2NamedMap,
    Cifti2Parcel,
    Cifti2Surface,
    Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ,
    Cifti2VertexIndices,
    Cifti2Vertices,
    Cifti2Volume,
    Cifti2VoxelIndicesIJK,
    _underscore,
)


class Cifti2Extension(Nifti1Extension[Cifti2Header]):
    code = 32

    def _unmangle(self, value: bytes) -> Cifti2Header:
        parser = Cifti2Parser()
        parser.parse(string=value)
        return parser.header

    def _mangle(self, value: Cifti2Header) -> bytes:
        if not isinstance(value, Cifti2Header):
            raise ValueError('Can only mangle a Cifti2Header.')
        return value.to_xml()


extension_codes.add_codes(((Cifti2Extension.code, 'cifti', Cifti2Extension),))

intent_codes.add_codes(
    (
        # The codes below appear on the CIFTI-2 standard
        # http://www.nitrc.org/plugins/mwiki/index.php/cifti:ConnectivityMatrixFileFormats
        # https://www.nitrc.org/forum/attachment.php?attachid=341&group_id=454&forum_id=1955
        (3000, 'ConnUnknown', (), 'NIFTI_INTENT_CONNECTIVITY_UNKNOWN'),
        (3001, 'ConnDense', (), 'NIFTI_INTENT_CONNECTIVITY_DENSE'),
        (3002, 'ConnDenseSeries', (), 'NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES'),
        (3003, 'ConnParcels', (), 'NIFTI_INTENT_CONNECTIVITY_PARCELLATED'),
        (3004, 'ConnParcelSries', (), 'NIFTI_INTENT_CONNECTIVITY_PARCELLATED_SERIES'),
        (3006, 'ConnDenseScalar', (), 'NIFTI_INTENT_CONNECTIVITY_DENSE_SCALARS'),
        (3007, 'ConnDenseLabel', (), 'NIFTI_INTENT_CONNECTIVITY_DENSE_LABELS'),
        (3008, 'ConnParcelScalr', (), 'NIFTI_INTENT_CONNECTIVITY_PARCELLATED_SCALAR'),
        (3009, 'ConnParcelDense', (), 'NIFTI_INTENT_CONNECTIVITY_PARCELLATED_DENSE'),
        (3010, 'ConnDenseParcel', (), 'NIFTI_INTENT_CONNECTIVITY_DENSE_PARCELLATED'),
        (3011, 'ConnPPSr', (), 'NIFTI_INTENT_CONNECTIVITY_PARCELLATED_PARCELLATED_SERIES'),
        (3012, 'ConnPPSc', (), 'NIFTI_INTENT_CONNECTIVITY_PARCELLATED_PARCELLATED_SCALAR'),
    )
)


class _Cifti2AsNiftiHeader(Nifti2Header):
    """Class for Cifti2 header extension"""

    @classmethod
    def _valid_intent_code(klass, intent_code):
        """Return True if `intent_code` matches our class `klass`"""
        return intent_code >= 3000 and intent_code < 3100

    @classmethod
    def may_contain_header(klass, binaryblock):
        if not super().may_contain_header(binaryblock):
            return False
        hdr = klass(binaryblock=binaryblock[: klass.sizeof_hdr])
        return klass._valid_intent_code(hdr.get_intent('code')[0])

    @staticmethod
    def _chk_qfac(hdr, fix=False):
        # Allow qfac of 0 without complaint for CIFTI-2
        rep = Report(HeaderDataError)
        if hdr['pixdim'][0] in (-1, 0, 1):
            return hdr, rep
        rep.problem_level = 20
        rep.problem_msg = 'pixdim[0] (qfac) should be 1 (default) or 0 or -1'
        if fix:
            hdr['pixdim'][0] = 1
            rep.fix_msg = 'setting qfac to 1'
        return hdr, rep

    @staticmethod
    def _chk_pixdims(hdr, fix=False):
        rep = Report(HeaderDataError)
        pixdims = hdr['pixdim']
        spat_dims = pixdims[1:4]
        if not np.any(spat_dims < 0):
            return hdr, rep
        rep.problem_level = 35
        rep.problem_msg = 'pixdim[1,2,3] should be zero or positive'
        if fix:
            hdr['pixdim'][1:4] = np.abs(spat_dims)
            rep.fix_msg = 'setting to abs of pixdim values'
        return hdr, rep


class _Cifti2AsNiftiImage(Nifti2Image):
    """Load a NIfTI2 image with a Cifti2 header"""

    header_class = _Cifti2AsNiftiHeader
    makeable = False


class Cifti2Parser(xml.XmlParser):
    """Class to parse an XML string into a CIFTI-2 header object"""

    def __init__(self, encoding=None, buffer_size=3500000, verbose=0):
        super().__init__(encoding=encoding, buffer_size=buffer_size, verbose=verbose)
        self.fsm_state = []
        self.struct_state = []

        # where to write CDATA:
        self.write_to = None
        self.header = None

        # Collecting char buffer fragments
        self._char_blocks = None

    __init__.__doc__ = xml.XmlParser.__init__.__doc__

    def StartElementHandler(self, name, attrs):
        self.flush_chardata()
        if self.verbose > 0:
            print('Start element:\n\t', repr(name), attrs)

        if name == 'CIFTI':
            # create cifti2 image
            self.header = Cifti2Header()
            self.header.version = ver = attrs['Version']
            if parse(ver) < Version('2'):
                raise ValueError(f'Only CIFTI-2 files are supported; found version {ver}')
            self.fsm_state.append('CIFTI')
            self.struct_state.append(self.header)

        elif name == 'Matrix':
            self.fsm_state.append('Matrix')
            matrix = Cifti2Matrix()
            parent = self.struct_state[-1]
            if not isinstance(parent, Cifti2Header):
                raise Cifti2HeaderError(
                    'Matrix element can only be a child of the CIFTI-2 Header element'
                )
            parent.matrix = matrix
            self.struct_state.append(matrix)

        elif name == 'MetaData':
            self.fsm_state.append('MetaData')
            meta = Cifti2MetaData()
            parent = self.struct_state[-1]
            if not isinstance(parent, (Cifti2Matrix, Cifti2NamedMap)):
                raise Cifti2HeaderError(
                    'MetaData element can only be a child of the CIFTI-2 Matrix '
                    'or NamedMap elements'
                )

            self.struct_state.append(meta)

        elif name == 'MD':
            pair = ['', '']
            self.fsm_state.append('MD')
            self.struct_state.append(pair)

        elif name == 'Name':
            self.write_to = 'Name'

        elif name == 'Value':
            self.write_to = 'Value'

        elif name == 'MatrixIndicesMap':
            self.fsm_state.append('MatrixIndicesMap')
            dimensions = [int(value) for value in attrs['AppliesToMatrixDimension'].split(',')]
            mim = Cifti2MatrixIndicesMap(
                applies_to_matrix_dimension=dimensions,
                indices_map_to_data_type=attrs['IndicesMapToDataType'],
            )
            for key, dtype in (
                ('NumberOfSeriesPoints', int),
                ('SeriesExponent', int),
                ('SeriesStart', float),
                ('SeriesStep', float),
                ('SeriesUnit', str),
            ):
                if key in attrs:
                    setattr(mim, _underscore(key), dtype(attrs[key]))
            matrix = self.struct_state[-1]
            if not isinstance(matrix, Cifti2Matrix):
                raise Cifti2HeaderError(
                    'MatrixIndicesMap element can only be a child of the CIFTI-2 Matrix element'
                )
            matrix.append(mim)
            self.struct_state.append(mim)

        elif name == 'NamedMap':
            self.fsm_state.append('NamedMap')
            named_map = Cifti2NamedMap()
            mim = self.struct_state[-1]
            if not isinstance(mim, Cifti2MatrixIndicesMap):
                raise Cifti2HeaderError(
                    'NamedMap element can only be a child of the CIFTI-2 MatrixIndicesMap element'
                )
            self.struct_state.append(named_map)
            mim.append(named_map)

        elif name == 'LabelTable':
            named_map = self.struct_state[-1]
            mim = self.struct_state[-2]
            if mim.indices_map_to_data_type != 'CIFTI_INDEX_TYPE_LABELS':
                raise Cifti2HeaderError(
                    'LabelTable element can only be a child of a MatrixIndicesMap '
                    'with CIFTI_INDEX_TYPE_LABELS type'
                )
            lata = Cifti2LabelTable()
            if not isinstance(named_map, Cifti2NamedMap):
                raise Cifti2HeaderError(
                    'LabelTable element can only be a child of the CIFTI-2 NamedMap element'
                )
            self.fsm_state.append('LabelTable')
            self.struct_state.append(lata)
            named_map.label_table = lata

        elif name == 'Label':
            lata = self.struct_state[-1]
            if not isinstance(lata, Cifti2LabelTable):
                raise Cifti2HeaderError(
                    'Label element can only be a child of the CIFTI-2 LabelTable element'
                )
            label = Cifti2Label()
            label.key = int(attrs['Key'])
            label.red = float(attrs['Red'])
            label.green = float(attrs['Green'])
            label.blue = float(attrs['Blue'])
            label.alpha = float(attrs['Alpha'])
            self.write_to = 'Label'
            self.fsm_state.append('Label')
            self.struct_state.append(label)

        elif name == 'MapName':
            named_map = self.struct_state[-1]
            if not isinstance(named_map, Cifti2NamedMap):
                raise Cifti2HeaderError(
                    'MapName element can only be a child of the CIFTI-2 NamedMap element'
                )

            self.fsm_state.append('MapName')
            self.write_to = 'MapName'

        elif name == 'Surface':
            surface = Cifti2Surface()
            mim = self.struct_state[-1]
            if not isinstance(mim, Cifti2MatrixIndicesMap):
                raise Cifti2HeaderError(
                    'Surface element can only be a child of the CIFTI-2 MatrixIndicesMap element'
                )
            if mim.indices_map_to_data_type != 'CIFTI_INDEX_TYPE_PARCELS':
                raise Cifti2HeaderError(
                    'Surface element can only be a child of a MatrixIndicesMap '
                    'with CIFTI_INDEX_TYPE_PARCELS type'
                )
            surface.brain_structure = attrs['BrainStructure']
            surface.surface_number_of_vertices = int(attrs['SurfaceNumberOfVertices'])
            mim.append(surface)

        elif name == 'Parcel':
            parcel = Cifti2Parcel()
            mim = self.struct_state[-1]
            if not isinstance(mim, Cifti2MatrixIndicesMap):
                raise Cifti2HeaderError(
                    'Parcel element can only be a child of the CIFTI-2 MatrixIndicesMap element'
                )
            parcel.name = attrs['Name']
            mim.append(parcel)
            self.fsm_state.append('Parcel')
            self.struct_state.append(parcel)

        elif name == 'Vertices':
            vertices = Cifti2Vertices()
            parcel = self.struct_state[-1]
            if not isinstance(parcel, Cifti2Parcel):
                raise Cifti2HeaderError(
                    'Vertices element can only be a child of the CIFTI-2 Parcel element'
                )
            vertices.brain_structure = attrs['BrainStructure']
            if vertices.brain_structure not in CIFTI_BRAIN_STRUCTURES:
                raise Cifti2HeaderError('BrainStructure for this Vertices element is not valid')
            parcel.append_cifti_vertices(vertices)
            self.fsm_state.append('Vertices')
            self.struct_state.append(vertices)
            self.write_to = 'Vertices'

        elif name == 'VoxelIndicesIJK':
            parent = self.struct_state[-1]
            if not isinstance(parent, (Cifti2Parcel, Cifti2BrainModel)):
                raise Cifti2HeaderError(
                    'VoxelIndicesIJK element can only be a child of the CIFTI-2 '
                    'Parcel or BrainModel elements'
                )
            parent.voxel_indices_ijk = Cifti2VoxelIndicesIJK()
            self.write_to = 'VoxelIndices'

        elif name == 'Volume':
            mim = self.struct_state[-1]
            if not isinstance(mim, Cifti2MatrixIndicesMap):
                raise Cifti2HeaderError(
                    'Volume element can only be a child of the CIFTI-2 MatrixIndicesMap element'
                )
            dimensions = tuple(int(val) for val in attrs['VolumeDimensions'].split(','))
            volume = Cifti2Volume(volume_dimensions=dimensions)
            mim.append(volume)
            self.fsm_state.append('Volume')
            self.struct_state.append(volume)

        elif name == 'TransformationMatrixVoxelIndicesIJKtoXYZ':
            volume = self.struct_state[-1]
            if not isinstance(volume, Cifti2Volume):
                raise Cifti2HeaderError(
                    'TransformationMatrixVoxelIndicesIJKtoXYZ element can only be a child '
                    'of the CIFTI-2 Volume element'
                )
            transform = Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ()
            transform.meter_exponent = int(attrs['MeterExponent'])
            volume.transformation_matrix_voxel_indices_ijk_to_xyz = transform
            self.fsm_state.append('TransformMatrix')
            self.struct_state.append(transform)
            self.write_to = 'TransformMatrix'

        elif name == 'BrainModel':
            model = Cifti2BrainModel()
            mim = self.struct_state[-1]
            if not isinstance(mim, Cifti2MatrixIndicesMap):
                raise Cifti2HeaderError(
                    'BrainModel element can only be a child '
                    'of the CIFTI-2 MatrixIndicesMap element'
                )
            if mim.indices_map_to_data_type != 'CIFTI_INDEX_TYPE_BRAIN_MODELS':
                raise Cifti2HeaderError(
                    'BrainModel element can only be a child of a MatrixIndicesMap '
                    'with CIFTI_INDEX_TYPE_BRAIN_MODELS type'
                )
            for key, dtype in (
                ('IndexOffset', int),
                ('IndexCount', int),
                ('ModelType', str),
                ('BrainStructure', str),
                ('SurfaceNumberOfVertices', int),
            ):
                if key in attrs:
                    setattr(model, _underscore(key), dtype(attrs[key]))
            if model.brain_structure not in CIFTI_BRAIN_STRUCTURES:
                raise Cifti2HeaderError('BrainStructure for this BrainModel element is not valid')
            if model.model_type not in CIFTI_MODEL_TYPES:
                raise Cifti2HeaderError('ModelType for this BrainModel element is not valid')
            mim.append(model)
            self.fsm_state.append('BrainModel')
            self.struct_state.append(model)

        elif name == 'VertexIndices':
            index = Cifti2VertexIndices()
            model = self.struct_state[-1]
            if not isinstance(model, Cifti2BrainModel):
                raise Cifti2HeaderError(
                    'VertexIndices element can only be a child '
                    'of the CIFTI-2 BrainModel element'
                )
            self.fsm_state.append('VertexIndices')
            model.vertex_indices = index
            self.struct_state.append(index)
            self.write_to = 'VertexIndices'

    def EndElementHandler(self, name):
        self.flush_chardata()
        if self.verbose > 0:
            print('End element:\n\t', repr(name))

        if name == 'CIFTI':
            # remove last element of the list
            self.fsm_state.pop()
            self.struct_state.pop()

        elif name == 'Matrix':
            self.fsm_state.pop()
            self.struct_state.pop()

        elif name == 'MetaData':
            self.fsm_state.pop()
            meta = self.struct_state.pop()
            parent = self.struct_state[-1]
            parent.metadata = meta

        elif name == 'MD':
            self.fsm_state.pop()
            pair = self.struct_state.pop()
            meta = self.struct_state[-1]
            meta[pair[0]] = pair[1]

        elif name == 'Name':
            self.write_to = None

        elif name == 'Value':
            self.write_to = None

        elif name == 'MatrixIndicesMap':
            self.fsm_state.pop()
            self.struct_state.pop()

        elif name == 'NamedMap':
            self.fsm_state.pop()
            self.struct_state.pop()

        elif name == 'LabelTable':
            self.fsm_state.pop()
            self.struct_state.pop()

        elif name == 'Label':
            self.fsm_state.pop()
            label = self.struct_state.pop()
            lata = self.struct_state[-1]
            lata.append(label)
            self.write_to = None

        elif name == 'MapName':
            self.fsm_state.pop()
            self.write_to = None

        elif name == 'Parcel':
            self.fsm_state.pop()
            self.struct_state.pop()

        elif name == 'Vertices':
            self.fsm_state.pop()
            self.struct_state.pop()
            self.write_to = None

        elif name == 'VoxelIndicesIJK':
            self.write_to = None

        elif name == 'Volume':
            self.fsm_state.pop()
            self.struct_state.pop()

        elif name == 'TransformationMatrixVoxelIndicesIJKtoXYZ':
            self.fsm_state.pop()
            self.struct_state.pop()
            self.write_to = None

        elif name == 'BrainModel':
            self.fsm_state.pop()
            self.struct_state.pop()

        elif name == 'VertexIndices':
            self.fsm_state.pop()
            self.struct_state.pop()
            self.write_to = None

    def CharacterDataHandler(self, data):
        """Collect character data chunks pending collation

        The parser breaks the data up into chunks of size depending on the
        buffer_size of the parser.  A large bit of character data, with standard
        parser buffer_size (such as 8K) can easily span many calls to this
        function.  We thus collect the chunks and process them when we hit start
        or end tags.
        """
        if self._char_blocks is None:
            self._char_blocks = []
        self._char_blocks.append(data)

    def flush_chardata(self):
        """Collate and process collected character data"""
        if self._char_blocks is None:
            return
        # Just join the strings to get the data.  Maybe there are some memory
        # optimizations we could do by passing the list of strings to the
        # read_data_block function.
        data = ''.join(self._char_blocks)
        # Reset the char collector
        self._char_blocks = None
        # Process data
        if self.write_to == 'Name':
            data = data.strip()  # .decode('utf-8')
            pair = self.struct_state[-1]
            pair[0] = data

        elif self.write_to == 'Value':
            data = data.strip()  # .decode('utf-8')
            pair = self.struct_state[-1]
            pair[1] = data

        elif self.write_to == 'Vertices':
            # conversion to numpy array
            c = BytesIO(data.strip().encode('utf-8'))
            vertices = self.struct_state[-1]
            vertices.extend(np.loadtxt(c, dtype=int, ndmin=1))
            c.close()

        elif self.write_to == 'VoxelIndices':
            # conversion to numpy array
            c = BytesIO(data.strip().encode('utf-8'))
            parent = self.struct_state[-1]
            parent.voxel_indices_ijk.extend(np.loadtxt(c, dtype=int).reshape(-1, 3))
            c.close()

        elif self.write_to == 'VertexIndices':
            # conversion to numpy array
            c = BytesIO(data.strip().encode('utf-8'))
            index = self.struct_state[-1]
            index.extend(np.loadtxt(c, dtype=int, ndmin=1))
            c.close()

        elif self.write_to == 'TransformMatrix':
            # conversion to numpy array
            c = BytesIO(data.strip().encode('utf-8'))
            transform = self.struct_state[-1]
            matrix = np.loadtxt(c, dtype=np.float64)
            transform.matrix = matrix.reshape(4, 4)
            c.close()

        elif self.write_to == 'Label':
            label = self.struct_state[-1]
            label.label = data.strip()

        elif self.write_to == 'MapName':
            named_map = self.struct_state[-1]
            named_map.map_name = data.strip()  # .decode('utf-8')

    @property
    def pending_data(self):
        """True if there is character data pending for processing"""
        return self._char_blocks is not None


# class _Cifti2DenseDataSeriesNiftiHeader(_Cifti2AsNiftiHeader):
#
#    @classmethod
#    def _valid_intent_code(klass, intent_code):
#        """ Return True if `intent_code` matches our class `klass`
#        """
#        return intent_code == 3002
