"""
Deriving spatial maps from group fMRI data using Dictionary Learning and ICA
============================================================================

Many different approaches exist to derive spatial maps or networks from 
group fmri data. The methods seek to find distributed brain regions that 
exhibit similar changes in BOLD fluctuations over time. Decomposition
methods allow for generation of many independent maps simultaneously 
without the need to provide a priori information (e.g. seeds or priors.) 

This example will apply two popular decomposition methods, ICA and 
Dictionary Learning, to fMRI data measured while children and young adults 
watch movies. The resulting maps will be visualized using atlas plotting 
tools. 

CanICA is an ICA method for group-level analysis of fMRI data. Compared
to other strategies, it brings a well-controlled group model, as well as a
thresholding algorithm controlling for specificity and sensitivity with
an explicit model of the signal. The reference papers are:

    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299

    * G. Varoquaux et al. "ICA-based sparse features recovery from fMRI
      datasets", IEEE ISBI 2010, p. 1177

Pre-prints for both papers are available on hal
(http://hal.archives-ouvertes.fr)

.. note::

    The use of the attribute `components_img_` from decomposition
    estimators is implemented from version 0.4.1.
    For older versions, unmask the deprecated attribute `components_`
    to get the components image using attribute `masker_` embedded in
    estimator.
    See the :ref:`section Inverse transform: unmasking data <unmasking_step>`
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


####################################################################
# Here we apply CanICA on the data
# ---------------------------------
# We use as "template" as a strategy to compute the mask, as this leads
# to slightly faster and more reproducible results. However, the images
# need to be in MNI template space

from nilearn.decomposition import CanICA

canica = CanICA(n_components=20,
                memory="nilearn_cache", memory_level=2,
                threshold=3.,
                n_init=1,
                verbose=10,
                mask_strategy='template')
canica.fit(func_filenames)

# Retrieve the independent components in brain space. Directly
# accesible through attribute `components_img_`. Note that this
# attribute is implemented from version 0.4.1. For older versions,
# see note section above for details.
canica_components_img = canica.components_img_
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
canica_components_img.to_filename('canica_resting_state.nii.gz')


####################################################################
# To visualize we plot the outline of all components on one figure
# -----------------------------------------------------------------
from nilearn.plotting import plot_prob_atlas

# Plot all ICA components together
plot_prob_atlas(canica_components_img, title='All ICA components')


####################################################################
# Finally, we plot the map for each ICA component separately
# -----------------------------------------------------------
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show

for i, cur_img in enumerate(iter_img(canica_components_img)):
    plot_stat_map(cur_img, display_mode="z", title="IC %d" % i,
                  cut_coords=1, colorbar=False)

show()


####################################################################
# Compare CanICA to dictionary learning
# -------------------------------------------------------------
# Dictionary learning is a sparsity based decomposition method for extracting
# spatial maps. It extracts maps that are naturally sparse and usually cleaner
# than ICA. Here, we will compare Dictionary learning to CanICA.
#
#   * Arthur Mensch et al. `Compressed online dictionary learning for fast resting-state fMRI decomposition
#    <https://hal.archives-ouvertes.fr/hal-01271033/>`_,
#    ISBI 2016, Lecture Notes in Computer Science


###############################################################################
# Create a dictionary learning estimator
# ---------------------------------------------------------------
from nilearn.decomposition import DictLearning

dict_learning = DictLearning(n_components=20,
                             memory="nilearn_cache", memory_level=2,
                             verbose=1,
                             random_state=0,
                             n_epochs=1,
                             mask_strategy='template')

print('[Example] Fitting dicitonary learning model')
dict_learning.fit(func_filenames)
print('[Example] Saving results')
# Grab extracted components umasked back to Nifti image.
# Note: For older versions, less than 0.4.1. components_img_
# is not implemented. See Note section above for details.
dictlearning_components_img = dict_learning.components_img_
dictlearning_components_img.to_filename('dictionary_learning_resting_state.nii.gz')


###############################################################################
# Visualize the results
# ----------------------
from nilearn.plotting import find_xyz_cut_coords
from nilearn.image import index_img

names = {dict_learning: 'DictionaryLearning', canica: 'CanICA'}
estimators = [canica, dict_learning]
components_imgs = [canica_components_img, dictlearning_components_img]

# Selecting specific maps to display: maps were manually chosen to be similar
indices = {dict_learning: 8, canica: 14}
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

# .. note::
#     To see how to extract subject-level timeseries' from regions
#     created using Dictionary Learning, see :ref:`example Regions 
#     extraction using Dictionary Learning and functional connectomes
#     <sphx_glr_auto_examples_03_connectivitiy_plot_extract_regions_dictlearning_maps.py>'.
#