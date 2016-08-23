"""
Extracting brain signal from spheres
====================================

This example shows how to extract brain signals from spheres described by the
coordinates of their **center in MNI space** and a given
**radius in millimeters**.

In particular, it creates 4 ROIs part of the default mode network (DMN) as
8-mm-radius spheres centered at coordinates published in [1] and estimates
**direct connectivity** between them.

**References**

[1] Vincent J.L. et al.,  "Evidence for a frontoparietal control system
revealed by intrinsic functional connectivity", J Neurophysiol 100 (2008).

"""

###############################################################################
# Loding fMRI data and giving spheres centers
# -------------------------------------------
#
# We retrieve the first subject data of the ADHD dataset.
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd(n_subjects=1)
func_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: {0}'.format(
    func_filename))  # 4D data


###############################################################################
# We give coordinates of the posterior cingulate cortex,  the medial prefrontal
# cortex and the left and right angular gyrus, all part of the DMN.
pcc_coords = (1, -55, 17)
mpfc_coords = (0, 51, -7)
lag_coords = (-47, -71, 29)
rag_coords = (50, -64, 27)

dmn_coords = [pcc_coords, mpfc_coords, lag_coords, rag_coords]
labels = ['PCC', 'MPFC', 'lAG', 'rAG']

###############################################################################
# It is advised to **mask the spheres with grey matter**. We don't have
# subject's grey matter mask, we resort to a less precise group-level one.
gm_mask_img = datasets.fetch_icbm152_brain_gm_mask()
print(type(gm_mask_img))

###############################################################################
# This is a Nifti image, characterized by a data array, an affine and a header.

###############################################################################
# Creating the ROIs
# -----------------
#
# Data in the gray mask image is stored in a 3D array.

gm_mask_array = gm_mask_img.get_data()
print('GM image data of shape {0}.'.format(gm_mask_array.shape))

###############################################################################
# We identify indices of GM voxels and compute their MNI coordinates.
import numpy as np
from nilearn.image.resampling import coord_transform
i, j, k = np.where(gm_mask_array != 0)
gm_voxels_coords = np.array(coord_transform(i, j, k, gm_mask_img.affine)).T

print('Computed MNI coordinates of {0} GM voxels.'.format(
    gm_voxels_coords.shape[0]))

###############################################################################
# We use the obtained coordinates to mask GM voxels more than 8mm from each
# ROI center,
gm_voxels_in_pcc = np.linalg.norm(gm_voxels_coords - pcc_coords, axis=1) < 8.
gm_voxels_in_mpfc = np.linalg.norm(gm_voxels_coords - mpfc_coords, axis=1) < 8.
gm_voxels_in_lag = np.linalg.norm(gm_voxels_coords - lag_coords, axis=1) < 8.
gm_voxels_in_rag = np.linalg.norm(gm_voxels_coords - rag_coords, axis=1) < 8.

print(gm_voxels_in_pcc)
###############################################################################
# and combine the masks to mark GM voxels part of the whole DMN.
gm_voxels_in_dmn = \
    gm_voxels_in_pcc + gm_voxels_in_mpfc + gm_voxels_in_lag + gm_voxels_in_rag

print('{0} voxels from the {1} GM voxels are part of our DMN.'.format(
    gm_voxels_in_dmn.sum(), gm_voxels_in_dmn.shape[0]))
###############################################################################
# Finally, we need to go back from the mask array to the ROIs mask image.
from nilearn import masking

dmn_mask_img = masking.unmask(gm_voxels_in_dmn, gm_mask_img)
print(type(dmn_mask_img))

###############################################################################
# We can visualize the ROIs.
from nilearn import plotting

plotting.plot_roi(dmn_mask_img, cut_coords=[0, -68, 28],
                  title="DMN spheres intersected with GM")

###############################################################################
# Computing average signals on the spheres
# ----------------------------------------
#
# We make use of a spheric ROIs dedicated object, the
# `:class:nilearn.input_data.NiftiSpheresMasker`. We define it by specifying
# the **spheres centers and radius**, as well as optional arguments:
# The grey matter mask image, the detrending, signal normalization and
# filtering choices.
from nilearn import input_data

dmn_masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=8., mask_img=gm_mask_img,
    low_pass=0.1, t_r=2.5, high_pass=None, detrend=True, standardize=True,
    memory='nilearn_cache', memory_level=1, verbose=1)

###############################################################################
# Time-series are computed once functional and confounds filenames given.
dmn_time_series = dmn_masker.fit_transform(
    func_filename, confounds=[confound_filename])

print('Computed ROIs average signals: {0} timepoints for {1} ROIs.'.format(
    dmn_time_series.shape[0], dmn_time_series.shape[1]))

###############################################################################
# We display the time series and check visually their synchronization.
import matplotlib.pyplot as plt

plt.figure()
for time_serie, label in zip(dmn_time_series.T, labels):
    plt.plot(time_serie, label=label)

plt.title('Default Mode Network Average Time Series')
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')
plt.legend()
plt.tight_layout()


###############################################################################
# Estimating direct connectivity
# ------------------------------
#
# **Direct connections** between ROIs are revealed by the ROI-to-ROI
# **partial correlation coefficients**. A non-normalized version of these
# coefficients can be obtained from the inverse covariance or precision matrix
# of the signals. We estimate that matrix with **Ledoit-Wolf** estimator,
# well suited to **small number of ROIs /large number of scans** cases.
from sklearn.covariance import LedoitWolf

estimator = LedoitWolf()

###############################################################################
# We just need to fit the estimator with the DMN timeseries.
estimator.fit(dmn_time_series)

# negated precision coefficients are proportional to partial correlations.
negated_precision_matrix = -estimator.precision_

###############################################################################
# We can check that we got a square `n_spheres` by `n_spheres` matrix.
print('Precision matrix has shape {0}'.format(negated_precision_matrix.shape))

###############################################################################
# Visualizing the connections
# ---------------------------

###############################################################################
# We visualize the connectivity graph using the connectome dedicated function
# `nilearn.plotting.plot_connectome`. Connectivity values are reflected by
# edges colors.
figure = plt.figure(figsize=(6, 6))
display = plotting.plot_connectome(
    negated_precision_matrix, dmn_coords, figure=figure,
    title='DMN connectivity', display_mode='z', node_size=0)

# We overlay the graph by ROIs contours, and label them.
display.add_contours(dmn_mask_img, levels=[0])
plt.text(pcc_coords[0] + 8., pcc_coords[1] + 8., 'PCC')
plt.text(mpfc_coords[0] + 8., mpfc_coords[1] + 8., 'MPFC')
plt.text(lag_coords[0] + 8., lag_coords[1] + 8., 'lAG')
plt.text(rag_coords[0] + 8., rag_coords[1] + 8., 'rAG')

###############################################################################
# We can also display more synthetic connectome with hemispheric projections.
plotting.plot_connectome(
    negated_precision_matrix, dmn_coords,
    title='Connectivity projected on hemispheres', display_mode='lyrz')
# Notice MPFC (0, 51, -7) is included in both hemispheres since x == 0.

plotting.show()
