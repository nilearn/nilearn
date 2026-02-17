# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from collections.abc import MutableMapping

from . import xmlutils as xml


class CaretMetaData(xml.XmlSerializable, MutableMapping):
    """A list of name-value pairs used in various Caret-based XML formats

    * Description - Provides a simple method for user-supplied metadata that
      associates names with values.
    * Attributes: [NA]
    * Child Elements

        * MD (0...N)

    * Text Content: [NA]

    MD elements are a single metadata entry consisting of a name and a value.

    Attributes
    ----------
    data : mapping of {name: value} pairs

    >>> md = CaretMetaData()
    >>> md['key'] = 'val'
    >>> md
    <CaretMetaData {'key': 'val'}>
    >>> dict(md)
    {'key': 'val'}
    >>> md.to_xml()
    b'<MetaData><MD><Name>key</Name><Value>val</Value></MD></MetaData>'

    Objects may be constructed like any ``dict``:

    >>> md = CaretMetaData(key='val')
    >>> md.to_xml()
    b'<MetaData><MD><Name>key</Name><Value>val</Value></MD></MetaData>'
    """

    def __init__(self, *args, **kwargs):
        args, kwargs = self._sanitize(args, kwargs)
        self._data = dict(*args, **kwargs)

    @staticmethod
    def _sanitize(args, kwargs):
        """Override in subclasses to accept and warn on previous invocations"""
        return args, kwargs

    def __getitem__(self, key):
        """Get metadata entry by name

        >>> md = CaretMetaData({'key': 'val'})
        >>> md['key']
        'val'
        """
        return self._data[key]

    def __setitem__(self, key, value):
        """Set metadata entry by name

        >>> md = CaretMetaData({'key': 'val'})
        >>> dict(md)
        {'key': 'val'}
        >>> md['newkey'] = 'newval'
        >>> dict(md)
        {'key': 'val', 'newkey': 'newval'}
        >>> md['key'] = 'otherval'
        >>> dict(md)
        {'key': 'otherval', 'newkey': 'newval'}
        """
        self._data[key] = value

    def __delitem__(self, key):
        """Delete metadata entry by name

        >>> md = CaretMetaData({'key': 'val'})
        >>> dict(md)
        {'key': 'val'}
        >>> del md['key']
        >>> dict(md)
        {}
        """
        del self._data[key]

    def __len__(self):
        """Get length of metadata list

        >>> md = CaretMetaData({'key': 'val'})
        >>> len(md)
        1
        """
        return len(self._data)

    def __iter__(self):
        """Iterate over metadata entries

        >>> md = CaretMetaData({'key': 'val'})
        >>> for key in md:
        ...     print(key)
        key
        """
        return iter(self._data)

    def __repr__(self):
        return f'<{self.__class__.__name__} {self._data!r}>'

    def _to_xml_element(self):
        metadata = xml.Element('MetaData')

        for name_text, value_text in self._data.items():
            md = xml.SubElement(metadata, 'MD')
            name = xml.SubElement(md, 'Name')
            name.text = str(name_text)
            value = xml.SubElement(md, 'Value')
            value.text = str(value_text)
        return metadata
