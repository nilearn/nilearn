"""
Massively univariate analysis of face vs house recognition
==========================================================

A permuted Ordinary Least Squares algorithm is run at each voxel in
order to detemine whether or not it behaves differently under a "face
viewing" condition and a "house viewing" condition.
We consider the mean image per session and per condition.
Otherwise, the observations cannot be exchanged at random because
a time dependance exists between observations within a same session
(see [1] for more detailed explanations).

The example shows the small differences that exist between
Bonferroni-corrected p-values and family-wise corrected p-values obtained
from a permutation test combined with a max-type procedure [2].
Bonferroni correction is a bit conservative, as revealed by the presence of
a few false negative.

References
----------
[1] Winkler, A. M. et al. (2014).
    Permutation inference for the general linear model. Neuroimage.

[2] Anderson, M. J. & Robinson, J. (2001).
    Permutation tests for linear models.
    Australian & New Zealand Journal of Statistics, 43(1), 75-88.
    (http://avesbiodiv.mncn.csic.es/estadistica/permut2.pdf)

"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Feb. 2014

##############################################################################
# Load Haxby dataset
from nilearn import datasets, image
haxby_dataset = datasets.fetch_haxby(subjects=[2])

# print basic information on the dataset
print('Mask nifti image (3D) is located at: %s' % haxby_dataset.mask)
print('Functional nifti image (4D) is located at: %s' % haxby_dataset.func[0])

##############################################################################
# Mask data
mask_filename = haxby_dataset.mask

from nilearn.input_data import NiftiMasker
nifti_masker = NiftiMasker(
    smoothing_fwhm=8,
    mask_img=mask_filename,
    memory='nilearn_cache', memory_level=1)
func_filename = haxby_dataset.func[0]
func_reduced = image.index_img(
    func_filename, slice(0,1300))
print(func_reduced)
fmri_masked = nifti_masker.fit(func_reduced)
print('fitted')
series = fmri_masked.transform(func_reduced)
print('Data masked.')
print(series.shape)

