import numpy as np
import nibabel as nb
import tempfile
import os
from nose.tools import assert_true, assert_false, assert_raises
from numpy.testing import assert_array_equal, assert_equal


from nilearn.plotting.surf_plotting import load_surf_data, load_surf_mesh

# def _generate_surf():
#    data_positive = np.zeros((7, 7, 3))
#    rng = np.random.RandomState(42)
#    data_rng = rng.rand(7, 7, 3)
#    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
#    return nibabel.Nifti1Image(data_positive, mni_affine)

# test cases:

# load_surf_data
# numpy array return numpy array of same size squeezed
# files return numpy array
# raise value error if none of the above

# load_surf_mesh
# list return same list
# list that doesnt have exactly two entries
# files return list
# raise value error if none of the above


def _generate_surf():
    rng = np.random.RandomState(42)
    coords = rng.rand(20, 3)
    faces = rng.randint(coords.shape[0], size=(30, 3))
    return [coords, faces]


def test_load_surf_data_array():
    data_flat = np.zeros((20,))
    data_squeeze = np.zeros((20, 1, 3))
    assert_array_equal(load_surf_data(data_flat), np.zeros((20,)))
    assert_array_equal(load_surf_data(data_squeeze), np.zeros((20, 3)))


def test_load_surf_data_file_nii_gii():
    filename_gii = tempfile.mktemp(suffix='.gii')
    darray = nb.gifti.GiftiDataArray(data=np.zeros((20,)))
    gii = nb.gifti.GiftiImage(darrays=[darray])
    nb.gifti.write(gii, filename_gii)
    assert_array_equal(load_surf_data(filename_gii), np.zeros((20,)))
    os.remove(filename_gii)

    filename_nii = tempfile.mktemp(suffix='.nii')
    filename_niigz = tempfile.mktemp(suffix='.nii.gz')
    nii = nb.Nifti1Image(np.zeros((20,)), affine=None)
    nb.save(nii, filename_nii)
    nb.save(nii, filename_niigz)
    assert_array_equal(load_surf_data(filename_nii), np.zeros((20,)))
    assert_array_equal(load_surf_data(filename_niigz), np.zeros((20,)))
    os.remove(filename_nii)
    os.remove(filename_niigz)


def test_load_surf_data_file_freesurfer():
    data = np.zeros((20,))

    filename_sulc = tempfile.mktemp(suffix='.sulc')
    nb.freesurfer.io.write_morph_data(filename_sulc, data)
    assert_array_equal(load_surf_data(filename_sulc), np.zeros((20,)))
    os.remove(filename_sulc)

    filename_thick = tempfile.mktemp(suffix='.thickness')
    nb.freesurfer.io.write_morph_data(filename_thick, data)
    assert_array_equal(load_surf_data(filename_thick), np.zeros((20,)))
    os.remove(filename_thick)

    label_start = np.array([5900, 5899, 5901, 5902, 2638])
    label_end = np.array([8756, 6241, 8757, 1896, 6243])
    label = load_surf_data('data/test.label')
    assert_array_equal(label[:5], label_start)
    assert_array_equal(label[-5:], label_end)
    assert_equal(label.shape, (326,))
    del label, label_start, label_end

    annot_start = np.array([24, 29, 28, 27, 24, 31, 11, 25, 0, 12])
    annot_end = np.array([16, 16, 16, 16, 16, 16, 16, 16, 16, 16])
    annot = load_surf_data('data/test.annot')
    assert_array_equal(annot[:10], annot_start)
    assert_array_equal(annot[-10:], annot_end)
    assert_equal(annot.shape, (10242,))
    del annot, annot_start, annot_end

    # filename_annot = tempfile.mktemp(suffix='.annot')
    # rng = np.random.RandomState(42)
    # labels = rng.randint(1, 75, size=(20,))
    # ctab = rng.randint(255, size=(76, 5))
    # names = 76*['test']
    # nb.freesurfer.io.write_annot(filename_annot, labels, ctab, names)
    # assert_array_equal(load_surf_data(filename_annot), labels)
    # os.remove(filename_annot)
