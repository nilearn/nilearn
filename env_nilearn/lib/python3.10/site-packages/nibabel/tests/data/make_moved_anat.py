"""Make anatomical image with altered affine

* Add some rotations and translations to affine;
* Save as ``.nii`` file so SPM can read it.

See ``resample_using_spm.m`` for processing of this generated image by SPM.
"""

import numpy as np

import nibabel as nib
from nibabel.affines import from_matvec
from nibabel.eulerangles import euler2mat

if __name__ == '__main__':
    img = nib.load('anatomical.nii')
    some_rotations = euler2mat(0.1, 0.2, 0.3)
    extra_affine = from_matvec(some_rotations, [3, 4, 5])
    moved_anat = nib.Nifti1Image(img.dataobj, extra_affine.dot(img.affine), img.header)
    moved_anat.set_data_dtype(np.float32)
    nib.save(moved_anat, 'anat_moved.nii')
