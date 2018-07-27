"""
Comparing connectomes on different reference atlases
====================================================

An important part of the example, is to turn a parcellation into
connectome for visualization. This requires choosing centers for each parcel
or network, via :func:`nilearn.plotting.find_parcellation_cut_coords` for
parcellation based on labels and
:func:`nilearn.plotting.find_probabilistic_atlas_cut_coords` for
parcellation based on probabilistic values.

In the intermediary steps, we make use of
:class:`nilearn.input_data.NiftiLabelsMasker` and
:class:`nilearn.input_data.NiftiMapsMasker` to extract time series from nifti
objects using different parcellation atlases.
The time series of all subjects of the ADHD Dataset are concatenated and
given directly to :class:`nilearn.connectome.ConnectivityMeasure` for
computing parcel-wise correlation matrices for each atlas across all subjects.

Mean correlation matrix is displayed on glass brain on extracted coordinates.

# author: Amadeus Kanaan

"""
# General imports
import numpy as np

####################################################################
# Load atlases
# -------------
from nilearn import datasets

destrieux = datasets.fetch_atlas_destrieux_2009()
yeo = datasets.fetch_atlas_yeo_2011()
harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
# with split into left and right hemispheres
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',
                                                       symmetric_split=True)

print('Destrieux atlas nifti image (3D) is located at: %s' % destrieux['maps'])
print('Yeo atlas nifti image (3D) with 17 parcels and liberal mask is located '
      'at: %s' % yeo['thick_17'])
print('Harvard Oxford atlas nifti image (3D) thresholded at .25 is located '
      'at: %s' % harvard_oxford['maps'])
# Store atlases in a dictionary. Atlases are for extracting coordinates.
atlases = {
    'Destrieux Atlas (struct)': destrieux['maps'],
    'Yeo Atlas 17 thick (func)': yeo['thick_17'],
    'Harvard Oxford > 25% (struct one hemisphere)': harvard_oxford['maps'],
    'Harvard Oxford > 25% (struct two hemispheres)': harvard_oxford_s['maps']}

#########################################################################
# Load functional data
# --------------------
data = datasets.fetch_adhd(n_subjects=10)

print('Functional nifti images (4D, e.g., one subject) are located at : %r'
      % data['func'][0])
print('Counfound csv files (of same subject) are located at : %r'
      % data['confounds'][0])

##########################################################################
# Iterate over fetched atlases to extract coordinates - parcellations
# -------------------------------------------------------------------
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

# ConenctivityMeasure from Nilearn uses simple 'correlation' to compute
# connectivity matrices for all subjects in a list
connectome_measure = ConnectivityMeasure(kind='correlation')

# useful for plotting connectivity interactions on glass brain
from nilearn import plotting

for name, atlas in sorted(atlases.items()):
    # create masker to extract functional data within atlas parcels
    masker = NiftiLabelsMasker(labels_img=atlas,
                               standardize=True,
                               memory='nilearn_cache')

    # extract time series from all subjects and concatenate them
    time_series = []
    for func, confounds in zip(data.func, data.confounds):
        time_series.append(masker.fit_transform(func, confounds=confounds))

    # calculate correlation matrices across subjects and display
    correlation_matrices = connectome_measure.fit_transform(time_series)

    # Mean correlation matrix across 10 subjects can be grabbed like this,
    # using connectome measure object
    mean_correlation_matrix = connectome_measure.mean_

    # grab center coordinates for atlas labels
    coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas)

    # plot connectome with 80% edge strength in the connectivity
    plotting.plot_connectome(mean_correlation_matrix, coordinates,
                             edge_threshold="80%", title=name)

##########################################################################
# Load probabilistic atlases - extracting coordinates on brain maps
# -----------------------------------------------------------------

msdl = datasets.fetch_atlas_msdl()
smith_rsn = datasets.fetch_atlas_smith_2009()

# Store atlases in a dictionary. Atlases are for extracting coordinates.
atlases = {'MSDL (probabilistic)': msdl['maps'],
           'Smith RSN 20 (probabilistic)': smith_rsn['rsn20']}

##########################################################################
# Iterate over fetched atlases to extract coordinates - probabilistic
# -------------------------------------------------------------------
from nilearn.input_data import NiftiMapsMasker

for name, atlas in sorted(atlases.items()):
    # create masker to extract functional data within atlas parcels
    masker = NiftiMapsMasker(maps_img=atlas, standardize=True,
                             memory='nilearn_cache')

    # extract time series from all subjects and concatenate them
    time_series = []
    for func, confounds in zip(data.func, data.confounds):
        time_series.append(masker.fit_transform(func, confounds=confounds))

    # calculate correlation matrices across subjects and display
    correlation_matrices = connectome_measure.fit_transform(time_series)

    # Mean correlation matrix across 10 subjects can be grabbed like this,
    # using connectome measure object
    mean_correlation_matrix = connectome_measure.mean_

    # grab center coordinates for probabilistic atlas
    coordinates = plotting.find_probabilistic_atlas_cut_coords(maps_img=atlas)

    # plot connectome with 80% edge strength in the connectivity
    plotting.plot_connectome(mean_correlation_matrix, coordinates,
                             edge_threshold="80%", title=name)
plotting.show()
