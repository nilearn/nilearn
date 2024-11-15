"""Surface API."""

from nilearn import surface


class PolyData(surface.PolyData):
    """A collection of data arrays."""


class SurfaceMesh(surface.SurfaceMesh):
    """A surface :term:`mesh` having vertex, \
    coordinates and faces (triangles).
    """


class InMemoryMesh(surface.SurfaceMesh):
    """A surface mesh stored as in-memory numpy arrays."""


class FileMesh(surface.SurfaceMesh):
    """A surface mesh stored in a Gifti or Freesurfer file."""


class PolyMesh(surface.PolyMesh):
    """A collection of meshes."""


class SurfaceImage(surface.SurfaceImage):
    """Surface image, usually containing meshes & data for both hemispheres."""
