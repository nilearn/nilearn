"""File-based images that have data arrays

The class:`DataObjImage` class defines an image that extends the
:class:`FileBasedImage` by adding an array-like object, named ``dataobj``.
This can either be an actual numpy array, or an object that:

* returns an array from ``numpy.asanyarray(obj)``;
* has an attribute or property ``shape``.
"""

from __future__ import annotations

import typing as ty

import numpy as np

from .deprecated import deprecate_with_version
from .filebasedimages import FileBasedHeader, FileBasedImage

if ty.TYPE_CHECKING:
    import numpy.typing as npt

    from .arrayproxy import ArrayLike
    from .fileholders import FileMap
    from .filename_parser import FileSpec

ArrayImgT = ty.TypeVar('ArrayImgT', bound='DataobjImage')


class DataobjImage(FileBasedImage):
    """Template class for images that have dataobj data stores"""

    _data_cache: np.ndarray | None
    _fdata_cache: np.ndarray[ty.Any, np.dtype[np.floating]] | None

    def __init__(
        self,
        dataobj: ArrayLike,
        header: FileBasedHeader | ty.Mapping | None = None,
        extra: ty.Mapping | None = None,
        file_map: FileMap | None = None,
    ):
        """Initialize dataobj image

        The datobj image is a combination of (dataobj, header), with optional
        metadata in `extra`, and filename / file-like objects contained in the
        `file_map` mapping.

        Parameters
        ----------
        dataobj : object
           Object containing image data.  It should be some object that returns
           an array from ``np.asanyarray``.  It should have ``shape`` and
           ``ndim`` attributes or properties
        header : None or mapping or header instance, optional
           metadata for this image format
        extra : None or mapping, optional
           metadata to associate with image that cannot be stored in the
           metadata of this image type
        file_map : mapping, optional
           mapping giving file information for this image format
        """
        super().__init__(header=header, extra=extra, file_map=file_map)
        self._dataobj = dataobj
        self._data_cache = None
        self._fdata_cache = None

    @property
    def dataobj(self) -> ArrayLike:
        return self._dataobj

    @deprecate_with_version(
        'get_data() is deprecated in favor of get_fdata(), which has a more predictable return '
        'type. To obtain get_data() behavior going forward, use numpy.asanyarray(img.dataobj).',
        '3.0',
        '5.0',
    )
    def get_data(self, caching='fill'):
        """Return image data from image with any necessary scaling applied

        .. WARNING::

            We recommend you use the ``get_fdata`` method instead of the
            ``get_data`` method, because it is easier to predict the return
            data type.  ``get_data`` will be deprecated around November 2019
            and removed around November 2021.

            If you don't care about the predictability of the return data type,
            and you want the minimum possible data size in memory, you can
            replicate the array that would be returned by ``img.get_data()`` by
            using ``np.asanyarray(img.dataobj)``.

        The image ``dataobj`` property can be an array proxy or an array.  An
        array proxy is an object that knows how to load the image data from
        disk.  An image with an array proxy ``dataobj`` is a *proxy image*; an
        image with an array in ``dataobj`` is an *array image*.

        The default behavior for ``get_data()`` on a proxy image is to read the
        data from the proxy, and store in an internal cache.  Future calls to
        ``get_data`` will return the cached array.  This is the behavior
        selected with `caching` == "fill".

        Once the data has been cached and returned from an array proxy, if you
        modify the returned array, you will also modify the cached array
        (because they are the same array).  Regardless of the `caching` flag,
        this is always true of an array image.

        Parameters
        ----------
        caching : {'fill', 'unchanged'}, optional
            See the Notes section for a detailed explanation.  This argument
            specifies whether the image object should fill in an internal
            cached reference to the returned image data array. "fill" specifies
            that the image should fill an internal cached reference if
            currently empty.  Future calls to ``get_data`` will return this
            cached reference.  You might prefer "fill" to save the image object
            from having to reload the array data from disk on each call to
            ``get_data``.  "unchanged" means that the image should not fill in
            the internal cached reference if the cache is currently empty.  You
            might prefer "unchanged" to "fill" if you want to make sure that
            the call to ``get_data`` does not create an extra (cached)
            reference to the returned array.  In this case it is easier for
            Python to free the memory from the returned array.

        Returns
        -------
        data : array
            array of image data

        See also
        --------
        uncache: empty the array data cache

        Notes
        -----
        All images have a property ``dataobj`` that represents the image array
        data.  Images that have been loaded from files usually do not load the
        array data from file immediately, in order to reduce image load time
        and memory use.  For these images, ``dataobj`` is an *array proxy*; an
        object that knows how to load the image array data from file.

        By default (`caching` == "fill"), when you call ``get_data`` on a
        proxy image, we load the array data from disk, store (cache) an
        internal reference to this array data, and return the array.  The next
        time you call ``get_data``, you will get the cached reference to the
        array, so we don't have to load the array data from disk again.

        Array images have a ``dataobj`` property that already refers to an
        array in memory, so there is no benefit to caching, and the `caching`
        keywords have no effect.

        For proxy images, you may not want to fill the cache after reading the
        data from disk because the cache will hold onto the array memory until
        the image object is deleted, or you use the image ``uncache`` method.
        If you don't want to fill the cache, then always use
        ``get_data(caching='unchanged')``; in this case ``get_data`` will not
        fill the cache (store the reference to the array) if the cache is empty
        (no reference to the array).  If the cache is full, "unchanged" leaves
        the cache full and returns the cached array reference.

        The cache can affect the behavior of the image, because if the cache is
        full, or you have an array image, then modifying the returned array
        will modify the result of future calls to ``get_data()``.  For example
        you might do this:

        >>> import os
        >>> import nibabel as nib
        >>> from nibabel.testing import data_path
        >>> img_fname = os.path.join(data_path, 'example4d.nii.gz')

        >>> img = nib.load(img_fname) # This is a proxy image
        >>> nib.is_proxy(img.dataobj)
        True

        The array is not yet cached by a call to "get_data", so:

        >>> img.in_memory
        False

        After we call ``get_data`` using the default `caching` == 'fill', the
        cache contains a reference to the returned array ``data``:

        >>> data = img.get_data()
        >>> img.in_memory
        True

        We modify an element in the returned data array:

        >>> data[0, 0, 0, 0]
        0
        >>> data[0, 0, 0, 0] = 99
        >>> data[0, 0, 0, 0]
        99

        The next time we call 'get_data', the method returns the cached
        reference to the (modified) array:

        >>> data_again = img.get_data()
        >>> data_again is data
        True
        >>> data_again[0, 0, 0, 0]
        99

        If you had *initially* used `caching` == 'unchanged' then the returned
        ``data`` array would have been loaded from file, but not cached, and:

        >>> img = nib.load(img_fname)  # a proxy image again
        >>> data = img.get_data(caching='unchanged')
        >>> img.in_memory
        False
        >>> data[0, 0, 0] = 99
        >>> data_again = img.get_data(caching='unchanged')
        >>> data_again is data
        False
        >>> data_again[0, 0, 0, 0]
        0
        """
        if caching not in ('fill', 'unchanged'):
            raise ValueError('caching value should be "fill" or "unchanged"')
        if self._data_cache is not None:
            return self._data_cache
        data = np.asanyarray(self._dataobj)
        if caching == 'fill':
            self._data_cache = data
        return data

    def get_fdata(
        self,
        caching: ty.Literal['fill', 'unchanged'] = 'fill',
        dtype: npt.DTypeLike = np.float64,
    ) -> np.ndarray[ty.Any, np.dtype[np.floating]]:
        """Return floating point image data with necessary scaling applied

        The image ``dataobj`` property can be an array proxy or an array.  An
        array proxy is an object that knows how to load the image data from
        disk.  An image with an array proxy ``dataobj`` is a *proxy image*; an
        image with an array in ``dataobj`` is an *array image*.

        The default behavior for ``get_fdata()`` on a proxy image is to read
        the data from the proxy, and store in an internal cache.  Future calls
        to ``get_fdata`` will return the cached array.  This is the behavior
        selected with `caching` == "fill".

        Once the data has been cached and returned from an array proxy, if you
        modify the returned array, you will also modify the cached array
        (because they are the same array).  Regardless of the `caching` flag,
        this is always true of an array image.

        Parameters
        ----------
        caching : {'fill', 'unchanged'}, optional
            See the Notes section for a detailed explanation.  This argument
            specifies whether the image object should fill in an internal
            cached reference to the returned image data array. "fill" specifies
            that the image should fill an internal cached reference if
            currently empty.  Future calls to ``get_fdata`` will return this
            cached reference.  You might prefer "fill" to save the image object
            from having to reload the array data from disk on each call to
            ``get_fdata``.  "unchanged" means that the image should not fill in
            the internal cached reference if the cache is currently empty.  You
            might prefer "unchanged" to "fill" if you want to make sure that
            the call to ``get_fdata`` does not create an extra (cached)
            reference to the returned array.  In this case it is easier for
            Python to free the memory from the returned array.
        dtype : numpy dtype specifier
            A numpy dtype specifier specifying a floating point type.  Data is
            returned as this floating point type.  Default is ``np.float64``.

        Returns
        -------
        fdata : array
            Array of image data of data type `dtype`.

        See also
        --------
        uncache: empty the array data cache

        Notes
        -----
        All images have a property ``dataobj`` that represents the image array
        data.  Images that have been loaded from files usually do not load the
        array data from file immediately, in order to reduce image load time
        and memory use.  For these images, ``dataobj`` is an *array proxy*; an
        object that knows how to load the image array data from file.

        By default (`caching` == "fill"), when you call ``get_fdata`` on a
        proxy image, we load the array data from disk, store (cache) an
        internal reference to this array data, and return the array.  The next
        time you call ``get_fdata``, you will get the cached reference to the
        array, so we don't have to load the array data from disk again.

        Array images have a ``dataobj`` property that already refers to an
        array in memory, so there is no benefit to caching, and the `caching`
        keywords have no effect.

        For proxy images, you may not want to fill the cache after reading the
        data from disk because the cache will hold onto the array memory until
        the image object is deleted, or you use the image ``uncache`` method.
        If you don't want to fill the cache, then always use
        ``get_fdata(caching='unchanged')``; in this case ``get_fdata`` will not
        fill the cache (store the reference to the array) if the cache is empty
        (no reference to the array).  If the cache is full, "unchanged" leaves
        the cache full and returns the cached array reference.

        The cache can effect the behavior of the image, because if the cache is
        full, or you have an array image, then modifying the returned array
        will modify the result of future calls to ``get_fdata()``.  For example
        you might do this:

        >>> import os
        >>> import nibabel as nib
        >>> from nibabel.testing import data_path
        >>> img_fname = os.path.join(data_path, 'example4d.nii.gz')

        >>> img = nib.load(img_fname) # This is a proxy image
        >>> nib.is_proxy(img.dataobj)
        True

        The array is not yet cached by a call to "get_fdata", so:

        >>> img.in_memory
        False

        After we call ``get_fdata`` using the default `caching` == 'fill', the
        cache contains a reference to the returned array ``data``:

        >>> data = img.get_fdata()
        >>> img.in_memory
        True

        We modify an element in the returned data array:

        >>> data[0, 0, 0, 0]
        0.0
        >>> data[0, 0, 0, 0] = 99
        >>> data[0, 0, 0, 0]
        99.0

        The next time we call 'get_fdata', the method returns the cached
        reference to the (modified) array:

        >>> data_again = img.get_fdata()
        >>> data_again is data
        True
        >>> data_again[0, 0, 0, 0]
        99.0

        If you had *initially* used `caching` == 'unchanged' then the returned
        ``data`` array would have been loaded from file, but not cached, and:

        >>> img = nib.load(img_fname)  # a proxy image again
        >>> data = img.get_fdata(caching='unchanged')
        >>> img.in_memory
        False
        >>> data[0, 0, 0] = 99
        >>> data_again = img.get_fdata(caching='unchanged')
        >>> data_again is data
        False
        >>> data_again[0, 0, 0, 0]
        0.0
        """
        if caching not in ('fill', 'unchanged'):
            raise ValueError('caching value should be "fill" or "unchanged"')
        dtype = np.dtype(dtype)
        if not issubclass(dtype.type, np.inexact):
            raise ValueError(f'{dtype} should be floating point type')
        # Return cache if cache present and of correct dtype.
        if self._fdata_cache is not None:
            if self._fdata_cache.dtype.type == dtype.type:
                return self._fdata_cache
        # Always return requested data type
        # For array proxies, will attempt to confine data array to dtype
        # during scaling
        data = np.asanyarray(self._dataobj, dtype=dtype)
        if caching == 'fill':
            self._fdata_cache = data
        return data

    @property
    def in_memory(self) -> bool:
        """True when any array data is in memory cache

        There are separate caches for `get_data` reads and `get_fdata` reads.
        This property is True if either of those caches are set.
        """
        return (
            isinstance(self._dataobj, np.ndarray)
            or self._fdata_cache is not None
            or self._data_cache is not None
        )

    def uncache(self) -> None:
        """Delete any cached read of data from proxied data

        Remember there are two types of images:

        * *array images* where the data ``img.dataobj`` is an array
        * *proxy images* where the data ``img.dataobj`` is a proxy object

        If you call ``img.get_fdata()`` on a proxy image, the result of reading
        from the proxy gets cached inside the image object, and this cache is
        what gets returned from the next call to ``img.get_fdata()``.  If you
        modify the returned data, as in::

            data = img.get_fdata()
            data[:] = 42

        then the next call to ``img.get_fdata()`` returns the modified array,
        whether the image is an array image or a proxy image::

            assert np.all(img.get_fdata() == 42)

        When you uncache an array image, this has no effect on the return of
        ``img.get_fdata()``, but when you uncache a proxy image, the result of
        ``img.get_fdata()`` returns to its original value.
        """
        self._fdata_cache = None
        self._data_cache = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self._dataobj.shape

    @property
    def ndim(self) -> int:
        return self._dataobj.ndim

    @classmethod
    def from_file_map(
        klass: type[ArrayImgT],
        file_map: FileMap,
        *,
        mmap: bool | ty.Literal['c', 'r'] = True,
        keep_file_open: bool | None = None,
    ) -> ArrayImgT:
        """Class method to create image from mapping in ``file_map``

        Parameters
        ----------
        file_map : dict
            Mapping with (key, value) pairs of (``file_type``, FileHolder
            instance giving file-likes for each file needed for this image
            type.
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        keep_file_open : { None, True, False }, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed.
            If ``file_map`` refers to an open file handle, this setting has no
            effect. The default value (``None``) will result in the value of
            ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT`` being used.

        Returns
        -------
        img : DataobjImage instance
        """
        raise NotImplementedError

    @classmethod
    def from_filename(
        klass: type[ArrayImgT],
        filename: FileSpec,
        *,
        mmap: bool | ty.Literal['c', 'r'] = True,
        keep_file_open: bool | None = None,
    ) -> ArrayImgT:
        """Class method to create image from filename `filename`

        Parameters
        ----------
        filename : str
            Filename of image to load
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        keep_file_open : { None, True, False }, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed.
            The default value (``None``) will result in the value of
            ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT`` being used.

        Returns
        -------
        img : DataobjImage instance
        """
        if mmap not in (True, False, 'c', 'r'):
            raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
        file_map = klass.filespec_to_file_map(filename)
        return klass.from_file_map(file_map, mmap=mmap, keep_file_open=keep_file_open)

    load = from_filename
