# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from ..volumeutils import Recoder

# Translate dtype.kind char codes to XML text output strings
KIND2FMT = {'i': '%d', 'u': '%d', 'f': '%10.6f', 'c': '%10.6f', 'V': ''}

array_index_order_codes = Recoder(
    (
        (1, 'RowMajorOrder', 'C'),
        (2, 'ColumnMajorOrder', 'F'),
    ),
    fields=('code', 'label', 'npcode'),
)

gifti_encoding_codes = Recoder(
    (
        (0, 'undef', 'GIFTI_ENCODING_UNDEF', 'undef'),
        (1, 'ASCII', 'GIFTI_ENCODING_ASCII', 'ASCII'),
        (2, 'B64BIN', 'GIFTI_ENCODING_B64BIN', 'Base64Binary'),
        (3, 'B64GZ', 'GIFTI_ENCODING_B64GZ', 'GZipBase64Binary'),
        (4, 'External', 'GIFTI_ENCODING_EXTBIN', 'ExternalFileBinary'),
    ),
    fields=('code', 'label', 'giistring', 'specs'),
)

gifti_endian_codes = Recoder(
    (
        (0, 'GIFTI_ENDIAN_UNDEF', 'Undef', 'undef'),
        (1, 'GIFTI_ENDIAN_BIG', 'BigEndian', 'big'),
        (2, 'GIFTI_ENDIAN_LITTLE', 'LittleEndian', 'little'),
    ),
    fields=('code', 'giistring', 'specs', 'byteorder'),
)
