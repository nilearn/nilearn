import nibabel as nib
import numpy as np
from nibabel.fileslice import fill_slicer

import nilearn.coordimage.pointset as ps


class CoordinateImage:
    """
    Attributes
    ----------
    header : a file-specific header
    coordaxis : ``CoordinateAxis``
    dataobj : array-like
    """

    def __init__(self, data, coordaxis, header=None):
        self.data = data
        self.coordaxis = coordaxis
        self.header = header

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, slicer):
        if isinstance(slicer, str):
            slicer = self.coordaxis.get_indices(slicer)
        elif isinstance(slicer, list):
            slicer = np.hstack([self.coordaxis.get_indices(sub) for sub in slicer])

        if isinstance(slicer, range):
            slicer = slice(slicer.start, slicer.stop, slicer.step)

        data = self.data
        if not isinstance(slicer, slice):
            data = np.asanyarray(data)
        return self.__class__(data[slicer], self.coordaxis[slicer], header=self.header.copy())

    @classmethod
    def from_image(klass, img):
        coordaxis = CoordinateAxis.from_header(img.header)
        if isinstance(img, nib.Cifti2Image):
            if img.ndim != 2:
                raise ValueError('Can only interpret 2D images')
            for i in img.header.mapped_indices:
                if isinstance(img.header.get_axis(i), nib.cifti2.BrainModelAxis):
                    break
            # Reinterpret data ordering based on location of coordinate axis
            data = img.dataobj.copy()
            data.order = ['F', 'C'][i]
            if i == 1:
                data._shape = data._shape[::-1]
        return klass(data, coordaxis, img.header)


class CoordinateAxis:
    """
    Attributes
    ----------
    parcels : list of ``Parcel`` objects
    """

    def __init__(self, parcels):
        self.parcels = parcels

    def load_structures(self, mapping):
        """
        Associate parcels to ``Pointset`` structures
        """
        raise NotImplementedError

    def __getitem__(self, slicer):
        """
        Return a sub-sampled CoordinateAxis containing structures
        matching the indices provided.
        """
        if slicer is Ellipsis or isinstance(slicer, slice) and slicer == slice(None):
            return self
        elif isinstance(slicer, slice):
            slicer = fill_slicer(slicer, len(self))
            start, stop, step = slicer.start, slicer.stop, slicer.step
        else:
            raise TypeError(f'Indexing type not supported: {type(slicer)}')

        subparcels = []
        pstop = 0
        for parcel in self.parcels:
            pstart, pstop = pstop, pstop + len(parcel)
            if pstop < start:
                continue
            if pstart >= stop:
                break
            if start < pstart:
                substart = (start - pstart) % step
            else:
                substart = start - pstart
            subparcels.append(parcel[substart : stop - pstart : step])
        return CoordinateAxis(subparcels)

    def get_indices(self, parcel, indices=None):
        """
        Return the indices in the full axis that correspond to the
        requested parcel. If indices are provided, further subsample
        the requested parcel.
        """
        subseqs = []
        idx = 0
        for p in self.parcels:
            if p.name == parcel:
                subseqs.append(range(idx, idx + len(p)))
            idx += len(p)
        if not subseqs:
            return ()
        if indices:
            return np.hstack(subseqs)[indices]
        if len(subseqs) == 1:
            return subseqs[0]
        return np.hstack(subseqs)

    def __len__(self):
        return sum(len(parcel) for parcel in self.parcels)

    # Hacky factory method for now
    @classmethod
    def from_header(klass, hdr):
        parcels = []
        if isinstance(hdr, nib.Cifti2Header):
            axes = [hdr.get_axis(i) for i in hdr.mapped_indices]
            for ax in axes:
                if isinstance(ax, nib.cifti2.BrainModelAxis):
                    break
            else:
                raise ValueError('No BrainModelAxis, cannot create CoordinateAxis')
            for name, slicer, struct in ax.iter_structures():
                if struct.volume_shape:
                    substruct = ps.NdGrid(struct.volume_shape, struct.affine)
                    indices = struct.voxel
                else:
                    substruct = None
                    indices = struct.vertex
                parcels.append(Parcel(name, substruct, indices))

        return klass(parcels)


class Parcel:
    """
    Attributes
    ----------
    name : str
    structure : ``Pointset``
    indices : object that selects a subset of coordinates in structure
    """

    def __init__(self, name, structure, indices):
        self.name = name
        self.structure = structure
        self.indices = indices

    def __repr__(self):
        return f'<Parcel {self.name}({len(self.indices)})>'

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, slicer):
        return self.__class__(self.name, self.structure, self.indices[slicer])


class GeometryCollection:
    """
    Attributes
    ----------
    structures : dict
        Mapping from structure names to ``Pointset``
    """

    def __init__(self, structures):
        self.structures = structures

    @classmethod
    def from_spec(klass, pathlike):
        """Load a collection of geometries from a specification."""
        raise NotImplementedError
