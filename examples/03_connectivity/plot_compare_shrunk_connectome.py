"""
Population shrinkage of covariance estimator (PoSCE) for connectome analysis
============================================================================

This examples applies population-based shrinkage of covariance to build a
functional connectome.

We use the `MSDL atlas
<https://team.inria.fr/parietal/18-2/spatial_patterns/spatial-patterns-in-resting-state/>`_
of functional regions in movie watching, and the
:class:`nilearn.input_data.NiftiMapsMasker` to extract time series.

    * Mehdi Rahim et al. `Population shrinkage of covariance (PoSCE) for better 
    individual brain functional-connectivity 
    estimation<https://hal.inria.fr/hal-02068389>`_, 
    in Medical Image Analysis (2019).

"""
###############################################################################
# Fetch 20 subjects from COBRE  the dataset.
from nilearn import datasets

rest_data = datasets.fetch_cobre(n_subjects=20)

###############################################################################
# Use regions of interest from the MSDL atlas.
atlas = datasets.fetch_atlas_msdl()

###############################################################################
# Extract time series
from nilearn import input_data

masker = input_data.NiftiMapsMasker(atlas.maps, standardize=True, 
                         memory="nilearn_cache", verbose=5)
time_series = [masker.fit_transform(f) for f in rest_data.func]

##############################################################################
# Compute Pearson correlation
# ---------------------------
from nilearn.connectome import ConnectivityMeasure

corr = ConnectivityMeasure(kind="correlation")
corr_connectivities = corr.fit_transform(time_series)

###############################################################################
# compute partial correlation

pcorr = ConnectivityMeasure(kind="partial correlation")
pcorr_connectivities = pcorr.fit_transform(time_series)

###############################################################################
# compute tangent embedding
tangent = ConnectivityMeasure(kind="tangent")
tangent_connectivities = tangent.fit_transform(time_series)

###############################################################################
# compute PoSCE
from nilearn.connectome import PopulationShrunkCovariance
from nilearn.connectome import vec_to_sym_matrix

posce = PopulationShrunkCovariance(shrinkage=1e-2)
posce.fit(time_series)
shrunk_embeddings = posce.transform(time_series)
shrunk_connectivities = [vec_to_sym_matrix(c) for c in shrunk_embeddings]

###############################################################################
# plot first subject
from nilearn import plotting
plotting.plot_matrix(corr_connectivities[0], title="Correlation")
plotting.plot_matrix(pcorr_connectivities[0], title="Partial Correlation")
plotting.plot_matrix(tangent_connectivities[0], title="Tangent Embedding")
plotting.plot_matrix(shrunk_connectivities[0], title="PoSCE")
