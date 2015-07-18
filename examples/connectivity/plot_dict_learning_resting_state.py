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
https://hal.inria.fr/inria-00588898/en/
"""
from sklearn.externals.joblib import Memory

### Load ADHD rest dataset ####################################################
from nilearn import datasets
# For linear assignment (should be moved in non user space...)

adhd_dataset = datasets.fetch_adhd(n_subjects=20)
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

### Apply DictLearning ########################################################
from nilearn.decomposition import DictLearning, CanICA

n_components = 10

dict_learning = DictLearning(n_components=n_components, smoothing_fwhm=6.,
                             memory="nilearn_cache", memory_level=5, verbose=2, random_state=0,
                             n_jobs=1, alpha=6, n_iter=1000)
canica = CanICA(n_components=n_components, smoothing_fwhm=6., memory_level=5, verbose=2, random_state=0,
                memory="nilearn_cache", n_jobs=1, n_init=1, threshold=3.)

estimators = [canica, dict_learning]

components_imgs = []

for estimator in estimators:
    print('[Example] Learning maps using %s model' % type(estimator).__name__)
    estimator.fit(func_filenames)
    print('[Example] Dumping results')
    components_img = estimator.masker_.inverse_transform(estimator.components_)
    components_img.to_filename('%s_resting_state.nii.gz' % type(estimator).__name__)
    components_imgs.append(components_img)

### Visualize the results #####################################################
# Show some interesting components
import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas, find_xyz_cut_coords
from nilearn.image import index_img

mem = Memory(cachedir='~/nilearn_cache')

print('[Example] Displaying')

fig, axes = plt.subplots(nrows=2)
cut_coords = find_xyz_cut_coords(index_img(components_imgs[1], 1))
for estimator, cur_img, ax in zip(estimators, components_imgs, axes):
    plot_prob_atlas(cur_img, title="%s" % estimator.__class__.__name__, axes=ax,
                  cut_coords=cut_coords, colorbar=False)

plt.show()