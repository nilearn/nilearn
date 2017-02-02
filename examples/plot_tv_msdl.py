from matplotlib import pyplot as plt
import nibabel
from nilearn import datasets, input_data

from nilearn.decomposition.tv_msdl import TVMSDL


# Fetch ADHD dataset
adhd = datasets.fetch_adhd(n_subjects=13)

# Fetch init atlas
atlas = datasets.fetch_atlas_smith_2009().rsn20

# Mask data
masker = input_data.MultiNiftiMasker(memory_level=1, memory='adhd_cache')
data = masker.fit_transform(adhd.func)
atlas = masker.transform(atlas)

# Create TV-MSDL
msdl = TVMSDL(
    masker.mask_img_, 20, alpha=0.5, mu=2.,
    l1_ratio=0.3, do_ica=False, max_iter=10, verbose=10)

msdl.fit(data, V_init=atlas)

maps = masker.inverse_transform(msdl.maps_)
nibabel.save(maps, 'tv-msdl_adhd.nii.gz')
plt.figure()
plt.plot(msdl.E)
plt.xlabel('Iteration')
plt.ylabel('Cost function')
plt.show()
