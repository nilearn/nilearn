import operator as op
from functools import reduce

import numpy as np
from nibabel.affines import apply_affine


class Pointset:
    def __init__(self, coords):
        self._coords = coords

    @property
    def n_coords(self):
        """Number of coordinates.

        Subclasses should override with more efficient implementations.
        """
        return self.get_coords().shape[0]

    def get_coords(self, name=None):
        """Nx3 array of coordinates in RAS+ space."""
        return self._coords


class TriangularMesh(Pointset):
    def __init__(self, mesh):
        if isinstance(mesh, tuple) and len(mesh) == 2:
            coords, self._triangles = mesh
        elif hasattr(mesh, "coords") and hasattr(mesh, "triangles"):
            coords = mesh.coords
            self._triangles = mesh.triangles
        elif hasattr(mesh, "get_mesh"):
            coords, self._triangles = mesh.get_mesh()
        else:
            raise ValueError("Cannot interpret input as triangular mesh")
        super().__init__(coords)

    @property
    def n_triangles(self):
        """Number of faces.

        Subclasses should override with more efficient implementations.
        """
        return self._triangles.shape[0]

    def get_triangles(self):
        """Mx3 array of indices into coordinate table."""
        return self._triangles

    def get_mesh(self, name=None):
        return self.get_coords(name=name), self.get_triangles()

    def get_names(self):
        """Get list of surface names.

        These can be passed to ``get_{coords,triangles,mesh}``.
        """
        raise NotImplementedError

    # This method is called for by the BIAP, but it now seems simpler to wait
    # to provide it until there are any proposed implementations
    # def decimate(self, *, n_coords=None, ratio=None):
    #     """ Return a TriangularMesh with a smaller number of vertices that
    #     preserves the geometry of the original """
    #     # To be overridden when a format provides optimization opportunities
    #     raise NotImplementedError


class TriMeshFamily(TriangularMesh):
    def __init__(self, mapping, default=None):
        self._triangles = None
        self._coords = {}
        for name, mesh in dict(mapping).items():
            coords, triangles = TriangularMesh(mesh).get_mesh()
            if self._triangles is None:
                self._triangles = triangles
            self._coords[name] = coords

        if default is None:
            default = next(iter(self._coords))
        self._default = default

    def get_names(self):
        return list(self._coords)

    def get_coords(self, name=None):
        if name is None:
            name = self._default
        return self._coords[name]


class NdGrid(Pointset):
    """.

    Attributes
    ----------
    shape : 3-tuple
        number of coordinates in each dimension of grid
    """

    def __init__(self, shape, affines):
        self.shape = tuple(shape)
        try:
            self._affines = dict(affines)
        except (TypeError, ValueError):
            self._affines = {"world": np.array(affines)}
        if "voxels" not in self._affines:
            self._affines["voxels"] = np.eye(4, dtype=np.uint8)

    def get_affine(self, name=None):
        """4x4 array."""
        if name is None:
            name = next(iter(self._affines))
        return self._affines[name]

    def get_coords(self, name=None):
        if name is None:
            name = next(iter(self._affines))
        aff = self.get_affine(name)
        dt = np.result_type(*(np.min_scalar_type(dim) for dim in self.shape))
        # This is pretty wasteful; we almost certainly want instead an
        # object that will retrieve a coordinate when indexed, but where
        # np.array(obj) returns this
        ijk_coords = np.array(list(np.ndindex(self.shape)), dtype=dt)
        return apply_affine(aff, ijk_coords)

    @property
    def n_coords(self):
        return reduce(op.mul, self.shape)
