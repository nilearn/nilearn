"""
Demonstrate basic loading and plotting of cortical surface data
===============================================================
The dataset that is a subset of the enhanced NKI Rockland sample
(http://fcon_1000.projects.nitrc.org/indi/enhanced/, Nooner et al, 2012)

Resting state fMRI scans (TR=645ms) of 102 subjects were preprocessed
(https://github.com/fliem/nki_nilearn) and projected onto the Freesurfer
fsaverage5 template (Dale et al, 1999, Fischl et al, 1999). For this example
we use the time series of a single subject's left hemisphere.

The Destrieux parcellation (Destrieux et al, 2010) in fsaverage5 space as
distributed with Freesurfer is used to select a seed region in the posterior
cingulate cortex.

Functional connectivity of the seed region to all other cortical nodes in the
same hemisphere is calculated using Pearson product-moment correlation
coefficient.

The :func:`nilearn.plotting.plot_surf_stat_map` function is used
to plot the resulting statistical map on the (inflated) pial surface.

See :ref:`plotting` for more details.

References
----------

Nooner et al, (2012). The NKI-Rockland Sample: A model for accelerating the
pace of discovery science in psychiatry. Frontiers in neuroscience 6, 152.
URL http://dx.doi.org/10.3389/fnins.2012.00152

Dale et al, (1999). Cortical surface-based analysis.I. Segmentation and
surface reconstruction. Neuroimage 9.
URL http://dx.doi.org/10.1006/nimg.1998.0395

Fischl et al, (1999). Cortical surface-based analysis. II: Inflation,
flattening, and a surface-based coordinate system. Neuroimage 9.
http://dx.doi.org/10.1006/nimg.1998.0396

Destrieux et al, (2010). Automatic parcellation of human cortical gyri and
sulci using standard anatomical nomenclature. NeuroImage, 53, 1.
URL http://dx.doi.org/10.1016/j.neuroimage.2010.06.010.
"""

###############################################################################
from nilearn import plotting
from nilearn import datasets
from scipy import stats
import numpy as np

###############################################################################
# Retrieve the data
nki_dataset = datasets.fetch_surf_nki_enhanced(n_subjects=1)

# NKI resting state data set of one subject left hemisphere in fsaverage5 space
resting_state = nki_dataset['func_left'][0]

# Destrieux parcellation left hemisphere in fsaverage5 space
destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
parcellation = destrieux_atlas['map_left']
labels = destrieux_atlas['labels']

# Retrieve fsaverage data
fsaverage = datasets.fetch_surf_fsaverage5()

# Fsaverage5 left hemisphere surface mesh files
fsaverage5_pial = fsaverage['pial_left'][0]
fsaverage5_inflated = fsaverage['infl_left'][0]
sulcal_depth_map = fsaverage['sulc_left'][0]

###############################################################################
# Load resting state time series and parcellation
timeseries = plotting.surf_plotting.load_surf_data(resting_state)

# Extract seed region: dorsal posterior cingulate gyrus
pcc_region = 'G_cingul-Post-dorsal'
pcc_labels = np.where(parcellation == labels.index(pcc_region))[0]

# Extract time series from seed region
seed_timeseries = np.mean(timeseries[pcc_labels], axis=0)

# Calculate Pearson product-moment correlation coefficient between seed
# time series and timeseries of all cortical nodes of the hemisphere
stat_map = np.zeros(timeseries.shape[0])
for i in range(timeseries.shape[0]):
    stat_map[i] = stats.pearsonr(seed_timeseries, timeseries[i])[0]

# Re-mask previously masked nodes (medial wall)
stat_map[np.where(np.mean(timeseries, axis=1) == 0)] = 0

###############################################################################
# Display ROI on surface
plotting.plot_surf_roi(fsaverage5_pial, roi_map=pcc_labels, hemi='left',
                       view='medial', bg_map=sulcal_depth_map, bg_on_data=True,
                       title='PCC Seed')

# Display unthresholded stat map  with dimmed background
plotting.plot_surf_stat_map(fsaverage5_pial, stat_map=stat_map, hemi='left',
                            view='medial', bg_map=sulcal_depth_map,
                            bg_on_data=True, darkness=.5,
                            title='Correlation map')

# Display unthresholded stat map without background map, transparency is
# automatically set to .5, but can also be controlled with the alpha parameter
plotting.plot_surf_stat_map(fsaverage5_pial, stat_map=stat_map, hemi='left',
                            view='medial', title='Plotting without background')

# Many different options are available for plotting, for example thresholding,
# or using custom colormaps
plotting.plot_surf_stat_map(fsaverage5_pial, stat_map=stat_map, hemi='left',
                            view='medial', bg_map=sulcal_depth_map,
                            bg_on_data=True, cmap='Spectral', threshold=.5,
                            title='Threshold and colormap')

# The plots can be saved to file, in which case the display is closed after
# creating the figure
plotting.plot_surf_stat_map(fsaverage5_inflated, stat_map=stat_map,
                            hemi='left', bg_map=sulcal_depth_map,
                            bg_on_data=True, threshold=.6,
                            output_file='plot_surf_stat_map.png')

plotting.show()
