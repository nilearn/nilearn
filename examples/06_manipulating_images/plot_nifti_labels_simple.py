"""
Simple example of NiftiLabelsMasker use
=======================================

Here is a simple example of automatic signal extraction using the nifti
labels masker.
"""

###########################################################################
# Retrieve the brain development functional dataset

from nilearn import datasets
dataset = datasets.fetch_development_fmri(n_subjects=1)
func_filename = dataset.func[0]

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' % func_filename)

###########################################################################
# Load an atlas
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

# The first label correspond to the background
print('The atlas contains {} non-overlapping regions'.format(
                                len(atlas.labels)-1))

###########################################################################
# Instantiate the mask and visualize atlas
from nilearn.input_data import NiftiLabelsMasker

# Instantiate the masker with label image and label values
masker = NiftiLabelsMasker(atlas.maps,
                           labels=atlas.labels)

# Visualize the atlas
# Note that we need to call fit prior to generating the mask
masker.fit()

# At this point, no functional image has been provided to the masker.
# We can still generate a report which can be displayed in a Jupyter
# Notebook, opened in a browser using the .open_in_browser() method,
# or saved to a file using the .save_as_html(output_filepath) mathod.
report = masker.generate_report()
report

##########################################################################
# Fitting the mask and generating a report
masker.fit(func_filename)

# We can again generate a report, but this time, the provided functional
# image is displayed with the ROI of the atlas.
# The report also contains a summary table giving the region sizes in mm3
report = masker.generate_report()
report

###########################################################################
# Preprocess the data with the NiftiLablesMasker
fmri_masked = masker.transform(func_filename)
# fmri_masked is now a 2D matrix, (n_time_points x n_regions)
fmri_masked.shape

