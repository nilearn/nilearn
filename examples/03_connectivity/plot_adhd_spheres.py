"""
Extracting brain signal from spheres
====================================

This example shows how to extract brain signals from spheres described by the
coordinates of their center in MNI space and a given radius in millimeters.
In particular, it creates 4 ROIs part of the default mode network (DMN) as
10-mm-radius spheres centered at coordinates from [1] and estimates partial
correlation strength between them. The example concludes with a more advanced
part dedicated to spheres radius choice.


** References **

[1] Vincent JL, Kahn I, Snyder AZ, Raichle ME, Buckner RL (2008) Evidence
for a frontoparietal control system revealed by intrinsic functional con-
nectivity. J Neurophysiol 100:3328-3342.

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
# It is advised to mask the spheres with grey matter. We don't have subject's
# grey matter mask, we resort to a less precise group-level one.
gm_mask_img = datasets.fetch_icbm152_brain_gm_mask()

###############################################################################
# Creating the ROIs
#------------------
#
# We compute voxels indices within the grey mask
# data in the gray mask image is stored in a 3D array
import numpy as np

gm_mask_array = gm_mask_img.get_data()
i, j, k = np.where(gm_mask_array != 0)


from nilearn.image.resampling import coord_transform
gm_mask_mni_coords = np.array(coord_transform(i, j, k, gm_mask_img.affine)).T

###############################################################################
# We use the obtained coordinates to identify voxels lying inside the seeds.
pcc_10mm_mask = np.linalg.norm(gm_mask_mni_coords - pcc_coords, axis=1) < 10.
mpfc_10mm_mask = np.linalg.norm(gm_mask_mni_coords - mpfc_coords, axis=1) < 10.
lag_10mm_mask = np.linalg.norm(gm_mask_mni_coords - lag_coords, axis=1) < 10.
rag_10mm_mask = np.linalg.norm(gm_mask_mni_coords - rag_coords, axis=1) < 10.

dmn_10mm_mask = pcc_10mm_mask + mpfc_10mm_mask + lag_10mm_mask + rag_10mm_mask

from nilearn import masking
dmn_10mm_mask_img = masking.unmask(dmn_10mm_mask, gm_mask_img)

from nilearn import plotting
plotting.plot_roi(dmn_10mm_mask_img, cut_coords=[0, -68, 28])

###############################################################################
# We can visualize 

###############################################################################
# Computing average signals on 10mm radius spheres
# ------------------------------------------------
#
# We make use of a spheric ROIs dedicated object, the
# `:class:nilearn.input_data.NiftiSpheresMasker`. We define it by specifying
# the spheres centers and radius, as well as optional arguments:
# The grey matter mask image, the detrending, signal normalization and
# filtering choices.
from nilearn import input_data

dmn_10mm_masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=10., mask_img=gm_mask_img,
    low_pass=0.1, t_r=2.5, high_pass=None, detrend=True, standardize=True,
    memory='nilearn_cache', memory_level=1, verbose=1)

###############################################################################
# Time-series are computed once functional and confounds filenames given.
dmn_10mm_average_time_series = dmn_10mm_masker.fit_transform(
    func_filename, confounds=[confound_filename])

print('Saved seeds time-series in array of shape {0}'.format(
    dmn_10mm_average_time_series.shape))

###############################################################################
# We display the time series and check visually their synchronization.
import matplotlib.pyplot as plt

plt.figure()
for time_serie, label in zip(dmn_10mm_average_time_series.T, labels):
    plt.plot(time_serie, label=label, lw=3.)

plt.title('Default Mode Network Average Time Series')
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')
plt.legend()
plt.tight_layout()


###############################################################################
# Estimating connectivity
# -----------------------
#
# Direct connections between ROIs are revealed by the signals precision
# (ie inverse covariance) matrix. We compute this matrix using the Ledoit-Wolf
# estimator, well suited to small number of ROIs /large number of scans cases.
from sklearn.covariance import LedoitWolf

estimator = LedoitWolf()

###############################################################################
# We just need to fit the estimator with the DMN timeseries.
estimator.fit(dmn_10mm_average_time_series)

# negated precision coefficients are proportional to partial correlations.
connectivity_matrix = -estimator.precision_  # no it's not connectivity matrix

###############################################################################
# We can check that we got a square (n_spheres, n_spheres) connectivity matrix.
print('connectivity matrix has shape {0}'.format(
    connectivity_matrix.shape))

###############################################################################
# Visualizing the connections
# ---------------------------

##########################################################################
# We display the connectivity graph with hemispheric projections using
# the connectome dedicated function `nilearn.plotting.plot_connectome`.
# Connectivity values are reflected by edges colors.
from nilearn import plotting

title = "Connectivity projected on hemispheres"
plotting.plot_connectome(connectivity_matrix, dmn_coords, title=title,
                         display_mode='lyrz')


title = 'DMN 10mm seed-to-voxel correlation maps within-seeds'
figure = plt.figure(figsize=(6, 6))
display = plotting.plot_connectome(
    connectivity_matrix, dmn_coords, node_size=0,
    title=title, display_mode='z', figure=figure, edge_vmax=.6)

display.add_overlay(dmn_10mm_mask_img)#, levels=[0, 1])
plt.text(pcc_coords[0] + 10., pcc_coords[1] + 10., 'PCC')
plt.text(mpfc_coords[0] + 10., mpfc_coords[1] + 10., 'MPFC')
plt.text(lag_coords[0] + 10., lag_coords[1] + 10., 'lAG')
plt.text(rag_coords[0] + 10., rag_coords[1] + 10., 'rAG')

plotting.show()
