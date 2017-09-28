from matplotlib import pyplot as plt

import nilearn.image
import nilearn.datasets
from nilearn.plotting import surf_plotting

brainpedia = nilearn.datasets.fetch_neurovault_ids(image_ids=(32015,))
image = brainpedia.images[0]

fsaverage = nilearn.datasets.fetch_surf_fsaverage5()
pial_left = surf_plotting.load_surf_mesh(fsaverage.pial_left)[0]

texture = surf_plotting.niimg_to_surf_data(image, pial_left)

surf_plotting.plot_surf_stat_map(fsaverage.infl_left, texture, cmap='bwr')

plt.show()
