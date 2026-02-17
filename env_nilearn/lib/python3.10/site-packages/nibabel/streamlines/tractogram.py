import copy
import numbers
import types
from collections.abc import Iterable, MutableMapping
from warnings import warn

import numpy as np

from nibabel.affines import apply_affine

from .array_sequence import ArraySequence


def is_data_dict(obj):
    """True if `obj` seems to implement the :class:`DataDict` API"""
    return hasattr(obj, 'store')


def is_lazy_dict(obj):
    """True if `obj` seems to implement the :class:`LazyDict` API"""
    return is_data_dict(obj) and callable(list(obj.store.values())[0])


class SliceableDataDict(MutableMapping):
    r"""Dictionary for which key access can do slicing on the values.

    This container behaves like a standard dictionary but extends key access to
    allow keys for key access to be indices slicing into the contained ndarray
    values.

    Parameters
    ----------
    \*args :
    \*\*kwargs :
        Positional and keyword arguments, passed straight through the ``dict``
        constructor.
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        try:
            return self.store[key]
        except (KeyError, TypeError, IndexError):
            pass  # Maybe it is an integer or a slicing object

        # Try to interpret key as an index/slice for every data element, in
        # which case we perform (maybe advanced) indexing on every element of
        # the dictionary.
        idx = key
        new_dict = type(self)()
        try:
            for k, v in self.items():
                new_dict[k] = v[idx]
        except (TypeError, ValueError, IndexError):
            pass
        else:
            return new_dict

        # Key was not a valid index/slice after all.
        return self.store[key]  # Will raise the proper error.

    def __contains__(self, key):
        return key in self.store

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


class PerArrayDict(SliceableDataDict):
    r"""Dictionary for which key access can do slicing on the values.

    This container behaves like a standard dictionary but extends key access to
    allow keys for key access to be indices slicing into the contained ndarray
    values. The elements must also be ndarrays.

    In addition, it makes sure the amount of data contained in those ndarrays
    matches the number of streamlines given at the instantiation of this
    instance.

    Parameters
    ----------
    n_rows : None or int, optional
        Number of rows per value in each key, value pair or None for not
        specified.
    \*args :
    \*\*kwargs :
        Positional and keyword arguments, passed straight through the ``dict``
        constructor.
    """

    def __init__(self, n_rows=0, *args, **kwargs):
        self.n_rows = n_rows
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        dtype = np.float64

        if isinstance(value, types.GeneratorType):
            value = list(value)

        if isinstance(value, np.ndarray):
            dtype = value.dtype
        elif not all(len(v) == len(value[0]) for v in value[1:]):
            dtype = object

        value = np.asarray(value, dtype=dtype)

        if value.ndim == 1 and value.dtype != object:
            # Reshape without copy
            value.shape = (len(value), 1)

        if value.ndim != 2 and value.dtype != object:
            raise ValueError('data_per_streamline must be a 2D array.')

        if value.dtype == object and not all(isinstance(v, Iterable) for v in value):
            raise ValueError('data_per_streamline must be a 2D array')

        # We make sure there is the right amount of values
        if 0 < self.n_rows != len(value):
            msg = f'The number of values ({len(value)}) should match n_elements ({self.n_rows}).'
            raise ValueError(msg)

        self.store[key] = value

    def _extend_entry(self, key, value):
        """Appends the `value` to the entry specified by `key`."""
        self[key] = np.concatenate([self[key], value])

    def extend(self, other):
        """Appends the elements of another :class:`PerArrayDict`.

        That is, for each entry in this dictionary, we append the elements
        coming from the other dictionary at the corresponding entry.

        Parameters
        ----------
        other : :class:`PerArrayDict` object
            Its data will be appended to the data of this dictionary.

        Returns
        -------
        None

        Notes
        -----
        The keys in both dictionaries must be the same.
        """
        if len(self) > 0 and len(other) > 0 and sorted(self.keys()) != sorted(other.keys()):
            msg = (
                'Entry mismatched between the two PerArrayDict objects. '
                f"This PerArrayDict contains '{sorted(self.keys())}' "
                f"whereas the other contains '{sorted(other.keys())}'."
            )
            raise ValueError(msg)

        self.n_rows += other.n_rows
        for key in other.keys():
            if key not in self:
                self[key] = other[key]
            else:
                self._extend_entry(key, other[key])


class PerArraySequenceDict(PerArrayDict):
    """Dictionary for which key access can do slicing on the values.

    This container behaves like a standard dictionary but extends key access to
    allow keys for key access to be indices slicing into the contained ndarray
    values.  The elements must also be :class:`ArraySequence`.

    In addition, it makes sure the amount of data contained in those array
    sequences matches the number of elements given at the instantiation
    of the instance.
    """

    def __setitem__(self, key, value):
        value = ArraySequence(value)

        # We make sure there is the right amount of data.
        if 0 < self.n_rows != value.total_nb_rows:
            msg = f'The number of values ({value.total_nb_rows}) should match ({self.n_rows}).'
            raise ValueError(msg)

        self.store[key] = value

    def _extend_entry(self, key, value):
        """Appends the `value` to the entry specified by `key`."""
        self[key].extend(value)


class LazyDict(MutableMapping):
    """Dictionary of generator functions.

    This container behaves like a dictionary but it makes sure its elements are
    callable objects that it assumes are generator functions yielding values.
    When getting the element associated with a given key, the element (i.e. a
    generator function) is first called before being returned.
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        # Use the 'update' method to set the keys.
        if len(args) == 1:
            if args[0] is None:
                return

            if isinstance(args[0], LazyDict):
                self.update(**args[0].store)  # Copy the generator functions.
                return

        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]()

    def __setitem__(self, key, value):
        if not callable(value):
            msg = (
                'Values in a `LazyDict` must be generator functions.'
                ' These are functions which, when called, return an'
                ' instantiated generator.'
            )
            raise TypeError(msg)
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


class TractogramItem:
    """Class containing information about one streamline.

    :class:`TractogramItem` objects have three public attributes: `streamline`,
    `data_for_streamline`, and `data_for_points`.

    Parameters
    ----------
    streamline : ndarray shape (N, 3)
        Points of this streamline represented as an ndarray of shape (N, 3)
        where N is the number of points.
    data_for_streamline : dict
        Dictionary containing some data associated with this particular
        streamline. Each key ``k`` is mapped to a ndarray of shape (Pt,), where
        ``Pt`` is the dimension of the data associated with key ``k``.
    data_for_points : dict
        Dictionary containing some data associated to each point of this
        particular streamline. Each key ``k`` is mapped to a ndarray of shape
        (Nt, Mk), where ``Nt`` is the number of points of this streamline and
        ``Mk`` is the dimension of the data associated with key ``k``.
    """

    def __init__(self, streamline, data_for_streamline, data_for_points):
        self.streamline = np.asarray(streamline)
        self.data_for_streamline = data_for_streamline
        self.data_for_points = data_for_points

    def __iter__(self):
        return iter(self.streamline)

    def __len__(self):
        return len(self.streamline)


class Tractogram:
    """Container for streamlines and their data information.

    Streamlines of a tractogram can be in any coordinate system of your
    choice as long as you provide the correct `affine_to_rasmm` matrix, at
    construction time. When applied to streamlines coordinates, that
    transformation matrix should bring the streamlines back to world space
    (RAS+ and mm space) [#]_.

    Moreover, when streamlines are mapped back to voxel space [#]_, a
    streamline point located at an integer coordinate (i,j,k) is considered
    to be at the center of the corresponding voxel. This is in contrast with
    other conventions where it might have referred to a corner.

    Attributes
    ----------
    streamlines : :class:`ArraySequence` object
        Sequence of $T$ streamlines. Each streamline is an ndarray of
        shape ($N_t$, 3) where $N_t$ is the number of points of
        streamline $t$.
    data_per_streamline : :class:`PerArrayDict` object
        Dictionary where the items are (str, 2D array).  Each key represents a
        piece of information $i$ to be kept alongside every streamline, and its
        associated value is a 2D array of shape ($T$, $P_i$) where $T$ is the
        number of streamlines and $P_i$ is the number of values to store for
        that particular piece of information $i$.
    data_per_point : :class:`PerArraySequenceDict` object
        Dictionary where the items are (str, :class:`ArraySequence`).  Each key
        represents a piece of information $i$ to be kept alongside every point
        of every streamline, and its associated value is an iterable of
        ndarrays of shape ($N_t$, $M_i$) where $N_t$ is the number of points
        for a particular streamline $t$ and $M_i$ is the number values to store
        for that particular piece of information $i$.

    References
    ----------
    .. [#] http://nipy.org/nibabel/coordinate_systems.html#naming-reference-spaces
    .. [#] http://nipy.org/nibabel/coordinate_systems.html#voxel-coordinates-are-in-voxel-space
    """

    def __init__(
        self, streamlines=None, data_per_streamline=None, data_per_point=None, affine_to_rasmm=None
    ):
        """
        Parameters
        ----------
        streamlines : iterable of ndarrays or :class:`ArraySequence`, optional
            Sequence of $T$ streamlines. Each streamline is an ndarray of
            shape ($N_t$, 3) where $N_t$ is the number of points of
            streamline $t$.
        data_per_streamline : dict of iterable of ndarrays, optional
            Dictionary where the items are (str, iterable).
            Each key represents an information $i$ to be kept alongside every
            streamline, and its associated value is an iterable of ndarrays of
            shape ($P_i$,) where $P_i$ is the number of scalar values to store
            for that particular information $i$.
        data_per_point : dict of iterable of ndarrays, optional
            Dictionary where the items are (str, iterable).
            Each key represents an information $i$ to be kept alongside every
            point of every streamline, and its associated value is an iterable
            of ndarrays of shape ($N_t$, $M_i$) where $N_t$ is the number of
            points for a particular streamline $t$ and $M_i$ is the number
            scalar values to store for that particular information $i$.
        affine_to_rasmm : ndarray of shape (4, 4) or None, optional
            Transformation matrix that brings the streamlines contained in
            this tractogram to *RAS+* and *mm* space where coordinate (0,0,0)
            refers to the center of the voxel. By default, the streamlines
            are in an unknown space, i.e. affine_to_rasmm is None.
        """
        self._set_streamlines(streamlines)
        self.data_per_streamline = data_per_streamline
        self.data_per_point = data_per_point
        self.affine_to_rasmm = affine_to_rasmm

    @property
    def streamlines(self):
        return self._streamlines

    def _set_streamlines(self, value):
        self._streamlines = ArraySequence(value)

    @property
    def data_per_streamline(self):
        return self._data_per_streamline

    @data_per_streamline.setter
    def data_per_streamline(self, value):
        self._data_per_streamline = PerArrayDict(
            len(self.streamlines), {} if value is None else value
        )

    @property
    def data_per_point(self):
        return self._data_per_point

    @data_per_point.setter
    def data_per_point(self, value):
        self._data_per_point = PerArraySequenceDict(
            self.streamlines.total_nb_rows, {} if value is None else value
        )

    @property
    def affine_to_rasmm(self):
        """Affine bringing streamlines in this tractogram to RAS+mm."""
        return copy.deepcopy(self._affine_to_rasmm)

    @affine_to_rasmm.setter
    def affine_to_rasmm(self, value):
        if value is not None:
            value = np.array(value)
            if value.shape != (4, 4):
                msg = (
                    'Affine matrix has a shape of (4, 4) but a ndarray with '
                    f'shape {value.shape} was provided instead.'
                )
                raise ValueError(msg)

        self._affine_to_rasmm = value

    def __iter__(self):
        for i in range(len(self.streamlines)):
            yield self[i]

    def __getitem__(self, idx):
        pts = self.streamlines[idx]

        data_per_streamline = {}
        for key in self.data_per_streamline:
            data_per_streamline[key] = self.data_per_streamline[key][idx]

        data_per_point = {}
        for key in self.data_per_point:
            data_per_point[key] = self.data_per_point[key][idx]

        if isinstance(idx, (numbers.Integral, np.integer)):
            return TractogramItem(pts, data_per_streamline, data_per_point)

        return Tractogram(
            pts, data_per_streamline, data_per_point, affine_to_rasmm=self.affine_to_rasmm
        )

    def __len__(self):
        return len(self.streamlines)

    def copy(self):
        """Returns a copy of this :class:`Tractogram` object."""
        return copy.deepcopy(self)

    def apply_affine(self, affine, lazy=False):
        """Applies an affine transformation on the points of each streamline.

        If `lazy` is not specified, this is performed *in-place*.

        Parameters
        ----------
        affine : ndarray of shape (4, 4)
            Transformation that will be applied to every streamline.
        lazy : {False, True}, optional
            If True, streamlines are *not* transformed in-place and a
            :class:`LazyTractogram` object is returned. Otherwise, streamlines
            are modified in-place.

        Returns
        -------
        tractogram : :class:`Tractogram` or :class:`LazyTractogram` object
            Tractogram where the streamlines have been transformed according
            to the given affine transformation. If the `lazy` option is true,
            it returns a :class:`LazyTractogram` object, otherwise it returns a
            reference to this :class:`Tractogram` object with updated
            streamlines.
        """
        if lazy:
            lazy_tractogram = LazyTractogram.from_tractogram(self)
            return lazy_tractogram.apply_affine(affine)

        if len(self.streamlines) == 0:
            return self

        if np.all(affine == np.eye(4)):
            return self  # No transformation.

        if self.streamlines.is_sliced_view:
            # Apply affine only on the selected streamlines.
            for i in range(len(self.streamlines)):
                self.streamlines[i] = apply_affine(affine, self.streamlines[i])
        else:
            self.streamlines._data = apply_affine(affine, self.streamlines._data, inplace=True)

        if self.affine_to_rasmm is not None:
            # Update the affine that brings back the streamlines to RASmm.
            self.affine_to_rasmm = np.dot(self.affine_to_rasmm, np.linalg.inv(affine))

        return self

    def to_world(self, lazy=False):
        """Brings the streamlines to world space (i.e. RAS+ and mm).

        If `lazy` is not specified, this is performed *in-place*.

        Parameters
        ----------
        lazy : {False, True}, optional
            If True, streamlines are *not* transformed in-place and a
            :class:`LazyTractogram` object is returned. Otherwise, streamlines
            are modified in-place.

        Returns
        -------
        tractogram : :class:`Tractogram` or :class:`LazyTractogram` object
            Tractogram where the streamlines have been sent to world space.
            If the `lazy` option is true, it returns a :class:`LazyTractogram`
            object, otherwise it returns a reference to this
            :class:`Tractogram` object with updated streamlines.
        """
        if self.affine_to_rasmm is None:
            msg = (
                'Streamlines are in a unknown space. This error can be'
                " avoided by setting the 'affine_to_rasmm' property."
            )
            raise ValueError(msg)

        return self.apply_affine(self.affine_to_rasmm, lazy=lazy)

    def extend(self, other):
        """Appends the data of another :class:`Tractogram`.

        Data that will be appended includes the streamlines and the content
        of both dictionaries `data_per_streamline` and `data_per_point`.

        Parameters
        ----------
        other : :class:`Tractogram` object
            Its data will be appended to the data of this tractogram.

        Returns
        -------
        None

        Notes
        -----
        The entries in both dictionaries `self.data_per_streamline` and
        `self.data_per_point` must match respectively those contained in
        the other tractogram.
        """
        self.streamlines.extend(other.streamlines)
        self.data_per_streamline.extend(other.data_per_streamline)
        self.data_per_point.extend(other.data_per_point)

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __add__(self, other):
        tractogram = self.copy()
        tractogram += other
        return tractogram


class LazyTractogram(Tractogram):
    """Lazy container for streamlines and their data information.

    This container behaves lazily as it uses generator functions to manage
    streamlines and their data information. This container is thus memory
    friendly since it doesn't require having all this data loaded in memory.

    Streamlines of a tractogram can be in any coordinate system of your
    choice as long as you provide the correct `affine_to_rasmm` matrix, at
    construction time. When applied to streamlines coordinates, that
    transformation matrix should bring the streamlines back to world space
    (RAS+ and mm space) [#]_.

    Moreover, when streamlines are mapped back to voxel space [#]_, a
    streamline point located at an integer coordinate (i,j,k) is considered
    to be at the center of the corresponding voxel. This is in contrast with
    other conventions where it might have referred to a corner.

    Attributes
    ----------
    streamlines : generator function
        Generator function yielding streamlines. Each streamline is an
        ndarray of shape ($N_t$, 3) where $N_t$ is the number of points of
        streamline $t$.
    data_per_streamline : instance of :class:`LazyDict`
        Dictionary where the items are (str, instantiated generator).
        Each key represents a piece of information $i$ to be kept alongside
        every streamline, and its associated value is a generator function
        yielding that information via ndarrays of shape ($P_i$,) where $P_i$ is
        the number of values to store for that particular piece of information
        $i$.
    data_per_point : :class:`LazyDict` object
        Dictionary where the items are (str, instantiated generator).  Each key
        represents a piece of information $i$ to be kept alongside every point
        of every streamline, and its associated value is a generator function
        yielding that information via ndarrays of shape ($N_t$, $M_i$) where
        $N_t$ is the number of points for a particular streamline $t$ and $M_i$
        is the number of values to store for that particular piece of
        information $i$.

    Notes
    -----
    LazyTractogram objects do not support indexing currently.
    LazyTractogram objects are suited for operations that can be linearized
    such as applying an affine transformation or converting streamlines from
    one file format to another.

    References
    ----------
    .. [#] http://nipy.org/nibabel/coordinate_systems.html#naming-reference-spaces
    .. [#] http://nipy.org/nibabel/coordinate_systems.html#voxel-coordinates-are-in-voxel-space
    """

    def __init__(
        self, streamlines=None, data_per_streamline=None, data_per_point=None, affine_to_rasmm=None
    ):
        """
        Parameters
        ----------
        streamlines : generator function, optional
            Generator function yielding streamlines. Each streamline is an
            ndarray of shape ($N_t$, 3) where $N_t$ is the number of points of
            streamline $t$.
        data_per_streamline : dict of generator functions, optional
            Dictionary where the items are (str, generator function).
            Each key represents an information $i$ to be kept alongside every
            streamline, and its associated value is a generator function
            yielding that information via ndarrays of shape ($P_i$,) where
            $P_i$ is the number of values to store for that particular
            information $i$.
        data_per_point : dict of generator functions, optional
            Dictionary where the items are (str, generator function).
            Each key represents an information $i$ to be kept alongside every
            point of every streamline, and its associated value is a generator
            function yielding that information via ndarrays of shape
            ($N_t$, $M_i$) where $N_t$ is the number of points for a particular
            streamline $t$ and $M_i$ is the number of values to store for
            that particular information $i$.
        affine_to_rasmm : ndarray of shape (4, 4) or None, optional
            Transformation matrix that brings the streamlines contained in
            this tractogram to *RAS+* and *mm* space where coordinate (0,0,0)
            refers to the center of the voxel. By default, the streamlines
            are in an unknown space, i.e. affine_to_rasmm is None.
        """
        super().__init__(streamlines, data_per_streamline, data_per_point, affine_to_rasmm)
        self._nb_streamlines = None
        self._data = None
        self._affine_to_apply = np.eye(4)

    @classmethod
    def from_tractogram(cls, tractogram):
        """Creates a :class:`LazyTractogram` object from a :class:`Tractogram` object.

        Parameters
        ----------
        tractogram : :class:`Tractgogram` object
            Tractogram from which to create a :class:`LazyTractogram` object.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            New lazy tractogram.
        """
        lazy_tractogram = cls(lambda: tractogram.streamlines.copy())

        # Set data_per_streamline using data_func
        def _gen(key):
            return lambda: iter(tractogram.data_per_streamline[key])

        for k in tractogram.data_per_streamline:
            lazy_tractogram._data_per_streamline[k] = _gen(k)

        # Set data_per_point using data_func
        def _gen(key):
            return lambda: iter(tractogram.data_per_point[key])

        for k in tractogram.data_per_point:
            lazy_tractogram._data_per_point[k] = _gen(k)

        lazy_tractogram._nb_streamlines = len(tractogram)
        lazy_tractogram.affine_to_rasmm = tractogram.affine_to_rasmm
        return lazy_tractogram

    @classmethod
    def from_data_func(cls, data_func):
        """Creates an instance from a generator function.

        The generator function must yield :class:`TractogramItem` objects.

        Parameters
        ----------
        data_func : generator function yielding :class:`TractogramItem` objects
            Generator function that whenever is called starts yielding
            :class:`TractogramItem` objects that will be used to instantiate a
            :class:`LazyTractogram`.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            New lazy tractogram.
        """
        if not callable(data_func):
            raise TypeError('`data_func` must be a generator function.')

        lazy_tractogram = cls()
        lazy_tractogram._data = data_func

        try:
            first_item = next(data_func())

            # Set data_per_streamline using data_func
            def _gen(key):
                return lambda: (t.data_for_streamline[key] for t in data_func())

            data_per_streamline_keys = first_item.data_for_streamline.keys()
            for k in data_per_streamline_keys:
                lazy_tractogram._data_per_streamline[k] = _gen(k)

            # Set data_per_point using data_func
            def _gen(key):
                return lambda: (t.data_for_points[key] for t in data_func())

            data_per_point_keys = first_item.data_for_points.keys()
            for k in data_per_point_keys:
                lazy_tractogram._data_per_point[k] = _gen(k)

        except StopIteration:
            pass

        return lazy_tractogram

    @property
    def streamlines(self):
        streamlines_gen = iter([])
        if self._streamlines is not None:
            streamlines_gen = self._streamlines()
        elif self._data is not None:
            streamlines_gen = (t.streamline for t in self._data())

        # Check if we need to apply an affine.
        if not np.allclose(self._affine_to_apply, np.eye(4)):

            def _apply_affine():
                for s in streamlines_gen:
                    yield apply_affine(self._affine_to_apply, s)

            return _apply_affine()

        return streamlines_gen

    def _set_streamlines(self, value):
        if value is not None and not callable(value):
            msg = (
                '`streamlines` must be a generator function. That is a'
                ' function which, when called, returns an instantiated'
                ' generator.'
            )
            raise TypeError(msg)
        self._streamlines = value

    @property
    def data_per_streamline(self):
        return self._data_per_streamline

    @data_per_streamline.setter
    def data_per_streamline(self, value):
        self._data_per_streamline = LazyDict(value)

    @property
    def data_per_point(self):
        return self._data_per_point

    @data_per_point.setter
    def data_per_point(self, value):
        self._data_per_point = LazyDict(value)

    @property
    def data(self):
        if self._data is not None:
            return self._data()

        def _gen_data():
            data_per_streamline_generators = {}
            for k, v in self.data_per_streamline.items():
                data_per_streamline_generators[k] = iter(v)

            data_per_point_generators = {}
            for k, v in self.data_per_point.items():
                data_per_point_generators[k] = iter(v)

            for s in self.streamlines:
                data_for_streamline = {}
                for k, v in data_per_streamline_generators.items():
                    data_for_streamline[k] = next(v)

                data_for_points = {}
                for k, v in data_per_point_generators.items():
                    data_for_points[k] = next(v)

                yield TractogramItem(s, data_for_streamline, data_for_points)

        return _gen_data()

    def __getitem__(self, idx):
        raise NotImplementedError('LazyTractogram does not support indexing.')

    def extend(self, other):
        msg = 'LazyTractogram does not support concatenation.'
        raise NotImplementedError(msg)

    def __iter__(self):
        count = 0
        for tractogram_item in self.data:
            yield tractogram_item
            count += 1

        # Keep how many streamlines there are in this tractogram.
        self._nb_streamlines = count

    def __len__(self):
        # Check if we know how many streamlines there are.
        if self._nb_streamlines is None:
            warn(
                'Number of streamlines will be determined manually by looping'
                ' through the streamlines. If you know the actual number of'
                ' streamlines, you might want to set it beforehand via'
                ' `self.header.nb_streamlines`.',
                Warning,
            )
            # Count the number of streamlines.
            self._nb_streamlines = sum(1 for _ in self.streamlines)

        return self._nb_streamlines

    def copy(self):
        """Returns a copy of this :class:`LazyTractogram` object."""
        tractogram = LazyTractogram(
            self._streamlines,
            self._data_per_streamline,
            self._data_per_point,
            self.affine_to_rasmm,
        )
        tractogram._nb_streamlines = self._nb_streamlines
        tractogram._data = self._data
        tractogram._affine_to_apply = self._affine_to_apply.copy()
        return tractogram

    def apply_affine(self, affine, lazy=True):
        """Applies an affine transformation to the streamlines.

        The transformation given by the `affine` matrix is applied after any
        other pending transformations to the streamline points.

        Parameters
        ----------
        affine : 2D array (4,4)
            Transformation matrix that will be applied on each streamline.
        lazy : True, optional
            Should always be True for :class:`LazyTractogram` object. Doing
            otherwise will raise a ValueError.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            A copy of this :class:`LazyTractogram` instance but with a
            transformation to be applied on the streamlines.
        """
        if not lazy:
            msg = 'LazyTractogram only supports lazy transformations.'
            raise ValueError(msg)

        tractogram = self.copy()  # New instance.

        # Update the affine that will be applied when returning streamlines.
        tractogram._affine_to_apply = np.dot(affine, self._affine_to_apply)

        if tractogram.affine_to_rasmm is not None:
            # Update the affine that brings back the streamlines to RASmm.
            tractogram.affine_to_rasmm = np.dot(self.affine_to_rasmm, np.linalg.inv(affine))
        return tractogram

    def to_world(self, lazy=True):
        """Brings the streamlines to world space (i.e. RAS+ and mm).

        The transformation is applied after any other pending transformations
        to the streamline points.

        Parameters
        ----------
        lazy : True, optional
            Should always be True for :class:`LazyTractogram` object. Doing
            otherwise will raise a ValueError.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            A copy of this :class:`LazyTractogram` instance but with a
            transformation to be applied on the streamlines.
        """
        if self.affine_to_rasmm is None:
            msg = (
                'Streamlines are in a unknown space. This error can be'
                " avoided by setting the 'affine_to_rasmm' property."
            )
            raise ValueError(msg)

        return self.apply_affine(self.affine_to_rasmm, lazy=lazy)
