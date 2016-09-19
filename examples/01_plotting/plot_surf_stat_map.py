"""
Demonstrate basic loading and plotting of cortical surface data
on the example of seed-based resting state functional connectivity.
===================================================================
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
import nibabel as nb
import numpy as np

###############################################################################
# Retrieve the data
nki_dataset = datasets.fetch_surf_nki_enhanced(n_subjects=1)

# Fsaverage5 left hemisphere surface mesh files
fsaverage5_pial = nki_dataset['fsaverage5_pial_left'][0]
fsaverage5_inflated = nki_dataset['fsaverage5_infl_left'][0]
sulcal_depth_map = nki_dataset['fsaverage5_sulc_left'][0]

# Destrieux parcellation left hemisphere in fsaverage5 space
# TO DO: include in nki_dataset for fetching
destrieux = '/home/julia/nilearn_testing/lh.aparc.a2009s.annot'

# NKI resting state data set of one subject left hemisphere in fsaverage5 space
resting_state = nki_dataset['func_left'][0]

###############################################################################
# Load resting state time series and parcellation
timeseries = plotting.surf_plotting.check_surf_data(resting_state)
parcellation = nb.freesurfer.read_annot(destrieux)

# Extract seed region: dorsal posterior cingulate gyrus
region = 'G_cingul-Post-dorsal'
labels = np.where(parcellation[0] == parcellation[2].index(region))[0]

# Extract time series from seed region
seed_timeseries = np.mean(timeseries[labels], axis=0)

# Calculate Pearson product-moment correlation coefficient between seed
# time series and timeseries of all cortical nodes of the hemisphere
stat_map = np.zeros(timeseries.shape[0])
for i in range(timeseries.shape[0]):
    stat_map[i] = stats.pearsonr(seed_timeseries, timeseries[i])[0]

# Re-mask previously masked nodes (medial wall)
stat_map[np.where(np.mean(timeseries, axis=1) == 0)] = 0

###############################################################################
# Display parcellation
plotting.plot_surf_roi(fsaverage5_inflated, roi_map=parcellation[0],
                       hemi='left', view='lateral', bg_map=sulcal_depth_map,
                       bg_on_data=True, darkness=.5, cmap='gist_ncar')

# Display ROI on surface
plotting.plot_surf_roi(fsaverage5_pial, roi_map=labels, hemi='left',
                       view='medial', bg_map=sulcal_depth_map, bg_on_data=True)

# Display unthresholded stat map in lateral and medial view
# dimmed background
plotting.plot_surf_stat_map(fsaverage5_pial, stat_map=stat_map, hemi='left',
                            bg_map=sulcal_depth_map, bg_on_data=True,
                            darkness=.5)
plotting.plot_surf_stat_map(fsaverage5_pial, stat_map=stat_map, hemi='left',
                            view='medial', bg_map=sulcal_depth_map,
                            bg_on_data=True, darkness=.5)

# Threshold stat_map
plotting.plot_surf_stat_map(fsaverage5_pial, stat_map=stat_map, hemi='left',
                            bg_map=sulcal_depth_map, bg_on_data=True,
                            darkness=.5, threshold=.6)
plotting.plot_surf_stat_map(fsaverage5_pial, stat_map=stat_map, hemi='left',
                            bg_map=sulcal_depth_map, bg_on_data=True,
                            darkness=.5, threshold=.6, view='medial')

# Display stat map with on inflated surface
plotting.plot_surf_stat_map(fsaverage5_inflated, stat_map=stat_map,
                            hemi='left', bg_map=sulcal_depth_map,
                            bg_on_data=True, threshold=.6)
plotting.plot_surf_stat_map(fsaverage5_inflated, stat_map=stat_map,
                            hemi='left', bg_map=sulcal_depth_map,
                            bg_on_data=True, view='medial', threshold=.6)

# changing the colormap and alpha
plotting.plot_surf_stat_map(fsaverage5_pial, stat_map=stat_map, hemi='left',
                            bg_map=sulcal_depth_map, bg_on_data=True,
                            cmap='Spectral', threshold=.6, alpha=.5)

# saving plots to file
plotting.plot_surf_stat_map(fsaverage5_pial, stat_map=stat_map, hemi='left',
                            bg_map=sulcal_depth_map,
                            bg_on_data=True, darkness=.5,
                            output_file='/tmp/plot_surf_stat_map.png')

plotting.show()
