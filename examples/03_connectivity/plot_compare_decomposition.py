"""
Dictionary Learning and ICA for doing group analysis of fMRI
============================================================

This example applies dictionary learning and ICA to
movie-watching fMRI data,
visualizing resulting components using atlas plotting tools.

Dictionary learning is a sparsity based decomposition method for extracting
spatial maps. It extracts maps that are naturally sparse and usually cleaner
than ICA

   * Arthur Mensch et al. `Compressed online dictionary learning for fast resting-state fMRI decomposition
     <https://hal.archives-ouvertes.fr/hal-01271033/>`_,
     ISBI 2016, Lecture Notes in Computer Science

.. note::

    The use of the attribute `components_img_` from decomposition
    estimators is implemented from version 0.4.1.
    For older versions, unmask the deprecated attribute `components_` to
    get the components image using attribute `masker_` embedded in estimator.
    See the :ref:`section Inverse transform: unmasking data <unmasking_step>`.
"""
###############################################################################
# Load brain development fmri dataset
# -----------------------------------
from nilearn import datasets

rest_dataset = datasets.fetch_development_fmri(n_subjects=30)
func_filenames = rest_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      rest_dataset.func[0])  # 4D data

###############################################################################
# Create two decomposition estimators
# ------------------------------------
from nilearn.decomposition import DictLearning, CanICA

n_components = 40

###############################################################################
# Dictionary learning
# --------------------
#
# We use as "template" as a strategy to compute the mask, as this leads
# to slightly faster and more reproducible results. However, the images
# need to be in MNI template space
dict_learning = DictLearning(n_components=n_components,
                             memory="nilearn_cache", memory_level=2,
                             verbose=1,
                             random_state=0,
                             n_epochs=1,
                             mask_strategy='template')
###############################################################################
# CanICA
# ------
canica = CanICA(n_components=n_components,
                memory="nilearn_cache", memory_level=2,
                threshold=3.,
                n_init=1,
                verbose=1,
                mask_strategy='template')

###############################################################################
# Fit both estimators
# --------------------
estimators = [dict_learning, canica]
names = {dict_learning: 'DictionaryLearning', canica: 'CanICA'}
components_imgs = []

for estimator in estimators:
    print('[Example] Learning maps using %s model' % names[estimator])
    estimator.fit(func_filenames)
    print('[Example] Saving results')
    # Grab extracted components umasked back to Nifti image.
    # Note: For older versions, less than 0.4.1. components_img_
    # is not implemented. See Note section above for details.
    components_img = estimator.components_img_
    components_img.to_filename('%s_resting_state.nii.gz' %
                               names[estimator])
    components_imgs.append(components_img)

###############################################################################
# Visualize the results
# ----------------------
from nilearn.plotting import (plot_prob_atlas, find_xyz_cut_coords, show,
                              plot_stat_map)
from nilearn.image import index_img

# Selecting specific maps to display: maps were manually chosen to be similar
indices = {dict_learning: 25, canica: 33}
# We select relevant cut coordinates for displaying
cut_component = index_img(components_imgs[0], indices[dict_learning])
cut_coords = find_xyz_cut_coords(cut_component)
for estimator, components in zip(estimators, components_imgs):
    # 4D plotting
    plot_prob_atlas(components, view_type="filled_contours",
                    title="%s" % names[estimator],
                    cut_coords=cut_coords, colorbar=False)
    # 3D plotting
    plot_stat_map(index_img(components, indices[estimator]),
                  title="%s" % names[estimator],
                  cut_coords=cut_coords, colorbar=False)
show()
