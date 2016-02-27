"""
Demonstrate basic surface plotting with plot_surf_stat_map
==========================================================

See :ref:`plotting` for more details.
"""
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import datasets

# Retrieve the data
nki_dataset = datasets.fetch_nki_enhanced_surface(n_subjects=1)

fsaverage5_pial = nki_dataset['fsaverage_pial'][0][0]
fsaverage5_inflated = nki_dataset['fsaverage_inflated'][0][0]
sulcal_depth_map = nki_dataset['fsaverage_sulc'][0][0]
stat_map = nki_dataset['resting_lh'][0][0]


# display unthresholded stat map in lateral and medial view
plotting.plot_surf_stat_map(fsaverage5_pial, 'left', stat_map=stat_map)

# display thresholded stat map without background map
plotting.plot_surf_stat_map(fsaverage5_pial, 'left', stat_map=stat_map,
                            threshold=.5)

# display thresholded stat map with background map lateral and medial
plotting.plot_surf_stat_map(fsaverage5_pial, 'left', stat_map=stat_map,
                            bg_map=sulcal_depth_map, threshold=.5)
plotting.plot_surf_stat_map(fsaverage5_pial, 'left', stat_map=stat_map,
                            bg_map=sulcal_depth_map, threshold=.5,
                            view='medial')

# display thresholded stat map with background map with background on statmap
plotting.plot_surf_stat_map(fsaverage5_pial, 'left', stat_map=stat_map,
                            bg_map=sulcal_depth_map, bg_on_stat=True,
                            threshold=.5)

# display thresholded stat map with background map, background on stat_map
# with less dark background
plotting.plot_surf_stat_map(fsaverage5_pial, 'left', stat_map=stat_map,
                            bg_map=sulcal_depth_map, bg_on_stat=True,
                            darkness=0.5, threshold=.5)

# display thresholded stat map without background map on inflated surface
plotting.plot_surf_stat_map(fsaverage5_inflated, 'left', stat_map=stat_map,
                            bg_map=sulcal_depth_map, threshold=.5)

# changing the colormap and alpha
plotting.plot_surf_stat_map(fsaverage5_pial, 'left', stat_map=stat_map,
                            bg_map=sulcal_depth_map, cmap='Spectral',
                            threshold=.5, alpha=0.7)

# saving plots to file
plotting.plot_surf_stat_map(fsaverage5_pial, 'left', stat_map=stat_map,
                            output_file='/tmp/plot_surf_stat_map.png')

plt.show()
