from pathlib import Path
from unittest import skipUnless

import numpy as np
from nibabel.arrayproxy import ArrayProxy
from nibabel.onetime import auto_attr
from nibabel.optpkg import optional_package

from nilearn._coordimage import pointset as ps

h5, has_h5py, _ = optional_package("h5py")

FS_DATA = Path("nilearn/_coordimage/tests/data/fsaverage")


class H5ArrayProxy:
    def __init__(self, file_like, dataset_name):
        self.file_like = file_like
        self.dataset_name = dataset_name
        with h5.File(file_like, "r") as h5f:
            arr = h5f[dataset_name]
            self._shape = arr.shape
            self._dtype = arr.dtype

    @property
    def is_proxy(self):
        return True

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._dtype

    def __array__(self, dtype=None):
        with h5.File(self.file_like, "r") as h5f:
            return np.asanyarray(h5f[self.dataset_name], dtype)

    def __getitem__(self, slicer):
        with h5.File(self.file_like, "r") as h5f:
            return h5f[self.dataset_name][slicer]


class H5Geometry(ps.TriMeshFamily):
    """Simple Geometry file structure that combines a single topology
    with one or more coordinate sets
    """

    @classmethod
    def from_filename(klass, pathlike):
        meshes = {}
        with h5.File(pathlike, "r") as h5f:
            triangles = H5ArrayProxy(pathlike, "/topology")
            for name in h5f["coordinates"]:
                meshes[name] = (
                    H5ArrayProxy(pathlike, f"/coordinates/{name}"),
                    triangles,
                )
        return klass(meshes)

    def to_filename(self, pathlike):
        with h5.File(pathlike, "w") as h5f:
            h5f.create_dataset("/topology", data=self.get_triangles())
            for name, coord in self._coords.items():
                h5f.create_dataset(f"/coordinates/{name}", data=coord)


class FSGeometryProxy:
    def __init__(self, pathlike):
        self._file_like = str(Path(pathlike))
        self._offset = None
        self._vnum = None
        self._fnum = None

    def _peek(self):
        from nibabel.freesurfer.io import _fread3

        with open(self._file_like, "rb") as fobj:
            magic = _fread3(fobj)
            if magic != 16777214:
                raise NotImplementedError("Triangle files only!")
            fobj.readline()
            fobj.readline()
            self._vnum = np.fromfile(fobj, ">i4", 1)[0]
            self._fnum = np.fromfile(fobj, ">i4", 1)[0]
            self._offset = fobj.tell()

    @property
    def vnum(self):
        if self._vnum is None:
            self._peek()
        return self._vnum

    @property
    def fnum(self):
        if self._fnum is None:
            self._peek()
        return self._fnum

    @property
    def offset(self):
        if self._offset is None:
            self._peek()
        return self._offset

    @auto_attr
    def coords(self):
        ap = ArrayProxy(self._file_like, ((self.vnum, 3), ">f4", self.offset))
        ap.order = "C"
        return ap

    @auto_attr
    def triangles(self):
        offset = self.offset + 12 * self.vnum
        ap = ArrayProxy(self._file_like, ((self.fnum, 3), ">i4", offset))
        ap.order = "C"
        return ap


class FreeSurferHemisphere(ps.TriMeshFamily):
    @classmethod
    def from_filename(klass, pathlike):
        path = Path(pathlike)
        hemi, default = path.name.split(".")
        mesh_names = (
            "orig",
            "white",
            "smoothwm",
            "pial",
            "inflated",
            "sphere",
            "midthickness",
            "graymid",
        )  # Often created
        if default not in mesh_names:
            mesh_names.append(default)
        meshes = {}
        for mesh in mesh_names:
            fpath = path.parent / f"{hemi}.{mesh}"
            if fpath.exists():
                meshes[mesh] = FSGeometryProxy(fpath)
        hemi = klass(meshes)
        hemi._default = default
        return hemi


def test_FreeSurferHemisphere():
    lh = FreeSurferHemisphere.from_filename(
        FS_DATA / "surf/lh.white"
    )
    assert lh.n_coords == 10242
    assert lh.n_triangles == 20480


@skipUnless(has_h5py, reason="Test requires h5py")
def test_make_H5Geometry(tmp_path):
    lh = FreeSurferHemisphere.from_filename(
        FS_DATA / "surf/lh.white"
    )
    h5geo = H5Geometry({name: lh.get_mesh(name) for name in ("white", "pial")})
    h5geo.to_filename(tmp_path / "geometry.h5")

    rt_h5geo = H5Geometry.from_filename(tmp_path / "geometry.h5")
    assert set(h5geo._coords) == set(rt_h5geo._coords)
    assert np.array_equal(lh.get_coords("white"), rt_h5geo.get_coords("white"))
    assert np.array_equal(lh.get_triangles(), rt_h5geo.get_triangles())
