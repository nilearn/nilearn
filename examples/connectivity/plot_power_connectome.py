import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets


#fetch power atlas coords
power = datasets.fetch_atlas_power_2011()
power_coords = power.rois[['x', 'y', 'z']]

#fetch dataset

adhd = datasets.fetch_adhd(n_subjects=1)

# Preprocess image and extract signals from Power-264
from nilearn import input_data
masker = input_data.NiftiSpheresMasker(seeds = power_coords,
                                       radius = 5.,
                                       standardize = True,
                                       smoothing_fwhm = 4,
                                       detrend = True,
                                       low_pass = 0.1,
                                       high_pass = 0.01,
                                       t_r = 2.5,
                                      verbose = 5)

timeseries = masker.fit_transform(adhd.func[0], confounds = adhd.confounds[0])

# calculate connectivity and plot Power-264 correlation matrix
connectivity = np.arctanh(np.corrcoef(timeseries.T))
plt.imshow(connectivity, vmin = -1., vmax = 1.)
plt.colorbar()
plt.title('Power 264 Connectivity')

# Plot the connectome
from nilearn import plotting
plotting.plot_connectome(connectivity,
                         power_coords,
                         edge_threshold ='99.8%',
                         node_size = 20)

#############################################################################
# Compute the sparse inverse covariance
from sklearn.covariance import GraphLassoCV
estimator = GraphLassoCV()
estimator.fit(timeseries)

# Display the covariance
plt.figure(figsize=(5, 5))
plt.imshow(estimator.covariance_, interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
plt.title('Covariance')

# display the corresponding graph
plotting.plot_connectome(estimator.covariance_,
                         power_coords,
                         title='Covariance',
                         edge_threshold ='99.8%',
                         node_size = 20)

# Display the sparse inverse covariance
plt.figure(figsize=(5, 5))
plt.imshow(-estimator.precision_, interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
# And display the labels
plt.title('Sparse inverse covariance')

# And now display the corresponding graph
plotting.plot_connectome(-estimator.precision_, power_coords,
                         title='Sparse inverse covariance',
                         edge_threshold="99.8%",
                         node_size = 20)

plt.show()