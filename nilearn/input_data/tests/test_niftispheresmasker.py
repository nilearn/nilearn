import numpy as np
import nibabel

from nilearn.input_data import NiftiSpheresMasker
#Preparing img with nans inside it 
def generate_imgs():
    affine = np.eye(4)

    data_with_nans = np.zeros((10, 10, 10), dtype=np.float64)
    data_with_nans[:, :, :] = np.nan

    data_without_nans = np.random.random((9, 9, 9))
    indices = np.nonzero(data_without_nans)

    # Leaving nans outside of some data
    data_with_nans[indices] = data_without_nans[indices]
    img = nibabel.Nifti1Image(data_with_nans, affine)
    return img

def test_is_nifti_spheres_masker_give_nans():
    seed = [(7, 7, 7)]
    img = generate_imgs()
    # Interaction of seed with nans
    masker = NiftiSpheresMasker(seeds=seed, radius=2.)
    assert not np.isnan(np.sum(masker.fit_transform(img)))
    
def test_nifti_spheres_masker_with_mask_img():
    affine = np.eye(4)
    mask = np.ones((9, 9, 9))
    seed = [(7, 7, 7)] 
    mask_img = nibabel.Nifti1Image(mask, affine)
    img = generate_imgs()
    # When mask image is provided, the seed interacts within the brain, so no nans.
    masker = NiftiSpheresMasker(seeds=seed, radius=2., mask_img=mask_img)
    assert not np.isnan(np.sum(masker.fit_transform(img)))