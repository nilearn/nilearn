"""
Group analysis of resting-state fMRI with dictionary learning: DictLearning
=====================================================

An example applying dictionary learning to resting-state data. This example applies it
to 10 subjects of the ADHD200 datasets.

Dictionary learning is a sparsity based decomposition method for extracting spatial maps.

    * Gael Varoquaux et al.
    Multi-subject dictionary learning to segment an atlas of brain spontaneous activity
    Information Processing in Medical Imaging, 2011, pp. 562-573, Lecture Notes in Computer Science

Pre-prints for paper is available on hal
(http://hal.archives-ouvertes.fr)
"""

### Load ADHD rest dataset ####################################################
from nilearn import datasets
# For linear assignment (should be moved in non user space...)

adhd_dataset = datasets.fetch_adhd(n_subjects=20, data_dir='/media/data/neuro')
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

### Apply DictLearning ########################################################
from nilearn.decomposition.dict_learning import DictLearning
from nilearn.decomposition.canica import CanICA

n_components = 10

dict_learning = DictLearning(n_components=n_components, smoothing_fwhm=6.,
                             memory="/media/data/nilearn_cache", memory_level=5, verbose=2, random_state=0,
                             n_jobs=1, alpha=6, n_iter=1000)
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="/media/data/nilearn_cache", memory_level=5, verbose=2, random_state=0,
                n_jobs=1, n_init=1, threshold=3.)

estimators = [canica, dict_learning]


for estimator in estimators:
    estimator.fit(func_filenames)

print('[Example] Dumping results')

components_imgs = []
# Retrieve learned spatial maps in brain space
for i, estimator in enumerate(estimators):
    components_img = estimator.masker_.inverse_transform(estimator.components_[indices[:, i]])
    components_img.to_filename('%s_resting_state.nii.gz' % type(estimator).__name__)
    components_imgs.append(components_img)

### Visualize the results #####################################################
# Show some interesting components
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords
from nilearn.image import index_img

for i in range(n_components):
    if i % 2 == 0:
        fig, axes = plt.subplots(nrows=3)
        cut_coords = find_xyz_cut_coords(index_img(components_imgs[1], i))
        for estimator, cur_img, ax in zip(estimators, components_imgs, axes):
            plot_stat_map(index_img(cur_img, i), title="Component %d" % i, axes=ax,
                          cut_coords=cut_coords, colorbar=False)

plt.show()