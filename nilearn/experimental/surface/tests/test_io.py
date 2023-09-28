import nibabel as nib
import numpy as np

from nilearn.experimental.surface import _io


def test_read_array(tmp_path):
    gifti_img = nib.gifti.GiftiImage()
    a = np.arange(5)
    gifti_array = nib.gifti.GiftiDataArray(a, datatype="float32")
    gifti_img.add_gifti_data_array(gifti_array)
    gifti_file = tmp_path / "a.gii"
    nib.save(gifti_img, gifti_file)
    read_a = _io.read_array(gifti_file)
    assert np.array_equal(a, read_a)


def test_read_mesh(tmp_path, mini_mesh):
    left = mini_mesh["left_hemisphere"]
    gifti_file = tmp_path / "img.gii"
    _io.mesh_to_gifti(left.coordinates, left.faces, gifti_file)
    read_mesh = _io.read_mesh(gifti_file)
    assert np.array_equal(left.coordinates, read_mesh["coordinates"])
    assert np.array_equal(left.faces, read_mesh["faces"])
