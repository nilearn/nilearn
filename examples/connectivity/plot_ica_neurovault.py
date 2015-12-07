"""
NeuroVault cross-study ICA maps.
================================

This example shows how to download statistical maps from
NeuroVault, label them with NeuroSynth terms,
and compute ICA components across all the maps.

See :func:`nilearn.datasets.fetch_neurovault` documentation for more details.
"""
# Author: Ben Cipollini
# License: BSD
# Ported from code authored by Chris Filo Gorgolewski, Gael Varoquaux
# https://github.com/NeuroVault/neurovault_analysis

import json
import io
import os
import warnings
warnings.simplefilter('error', RuntimeWarning)

import numpy as np
from joblib import Memory
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.feature_extraction import DictVectorizer

import nibabel as nib
from nilearn import datasets
from nilearn.datasets.utils import _fetch_files, _get_dataset_dir
from nilearn.image import resample_img, iter_img
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_background_mask, _extrapolate_out_mask
from nilearn.plotting import plot_stat_map


# Use for caching results to disk, so running twice will be *fast*
mem = Memory('cache', verbose=0)


@mem.cache
def resample_img_with_cache(src_img, target_affine, target_shape=None):
    """Cache wrapper around resample_img"""
    return resample_img(img, target_affine=target_img.affine,
                        target_shape=target_img.shape)

@mem.cache
def get_neurosynth_terms(images, data_dir, print_frequency=100):
    """ Grab terms for each image, decoded with neurosynth"""

    terms = list()
    vectorizer = DictVectorizer()
    for ii, img in enumerate(images):
        if ii % print_frequency == 0:
            print("Fetching terms for images %d-%d of %d" % (
                ii + 1, min(ii + print_frequency, len(images)), len(images)))

        url = 'http://neurosynth.org/api/v2/decode/?neurovault=%d' % img['id']
        fil = 'terms-for-image-%d.json' % img['id']
        elevations = _fetch_files(data_dir, ((fil, url, {'move': fil}),),
                                  verbose=2)[0]

        try:
            with io.open(elevations, 'r', encoding='utf8') as fp:
                data = json.load(fp)['data']
        except Exception as e:
            if os.path.exists(elevations):
                os.remove(elevations)
            terms.append({})
        else:
            data = data['values']
            terms.append(data)
    X = vectorizer.fit_transform(terms).toarray()
    return dict([(name, X[:, idx])
                 for name, idx in vectorizer.vocabulary_.items()])



# -------------------------------------------
# Define pre-download filters
bad_collects = [16]
bad_image_ids = [
    96, 97, 98,                    # The following maps are not brain maps
    338, 339,                      # And the following are crap
    335,                           # 335 is a duplicate of 336
    3360, 3362, 3364,              # These are mean images, and not Z maps
    1202, 1163, 1931, 1101, 1099]  # Ugly / obviously not Z maps
imfilts = [lambda im: im.get('perc_bad_voxels', 0.) < 100.]

# Download 100 matching images
ss_all = datasets.fetch_neurovault(max_images=100,  # Use np.inf for all imgs.
                                   collection_ids=[-bid for bid in bad_collects],
                                   image_ids=[-bid for bid in bad_image_ids],
                                   map_types=['F map', 'T map', 'Z map'],
                                   image_filters=imfilts)
images, collections = ss_all['images'], ss_all['collections']

### Resample the images #######################################################
target_img = datasets.load_mni152_template()
resampled_niis = []
for ii, image in enumerate(images):
    if ii % 100 == 0:  # Report progress, this can be slow.
        print("Resampling image %d-%d of %d..." % (
            ii + 1, min(ii + 100, len(images)), len(images)))
    for img in iter_img(image['local_path']):  # Works for 3d or 4d images
        resampled_niis.append(resample_img_with_cache(img))

# Report to the user any expansion from 4D images.
if len(images) != len(resampled_niis):
    print("After resampling, %d images => %d "
          "(each time point became an image)" % (
              len(images), len(resampled_niis)))

### Get neurosynth terms ######################################################
# Returns a dict of terms, with
#   a vector of values (1 per image)
terms = get_neurosynth_terms(images, _get_dataset_dir('neurosynth'))
good_terms = dict([(t, v) for t, v in terms.items()
                   if np.sum(v[v > 0]) > 0.])

print("Top 100 neurosynth terms:")
sort_idx = np.argsort([np.sum(v[v > 0]) for v in good_terms.values()])
for term_idx in sort_idx[-100:]:
    # Eliminate negative values
    term = list(good_terms.keys())[term_idx]
    vec = good_terms[term]
    vec[vec < 0] = 0
    print('\t%-25s: %.4e' % (term, np.sum(vec)))

### Get the grey matter mask ##################################################
print("Downloading and resampling grey matter mask.")
url = 'https://github.com/NeuroVault/neurovault_analysis/raw/master/gm_mask.nii.gz'
mask = _fetch_files(_get_dataset_dir('neurovault'),
                    (('gm_mask.nii.gz', url, {}),))[0]
mask = resample_and_expand_image(nib.load(mask), target_img)[0]
mask = new_image_like((mask.get_data() >= 0.5).astype(int),
                      mask.affine, mask.header)

### Get the ICA maps ##########################################################
# Do the ICA transform.
masker = NiftiMasker(mask_img=mask, memory=mem)
X = masker.fit_transform(resampled_niis)

print("Running ICA; may take time...")
fast_ica = FastICA(n_components=20, random_state=42)
ica_maps = fast_ica.fit_transform(X.T).T

### Map ICA components to terms ###############################################
term_matrix = np.asarray(list(good_terms.values()))
# Don't use the transform method as it centers the data
ica_terms = np.dot(term_matrix, fast_ica.components_.T).T
col_names = list(good_terms.keys())

### Generate figures ##########################################################
for idx, (ic, ic_terms) in enumerate(zip(ica_maps, ica_terms)):
    if -ic.min() > ic.max():
        # Flip the map's sign for prettiness
        ic = -ic
        ic_terms = -ic_terms

    ic_thr = stats.scoreatpercentile(np.abs(ic), 90)
    ic_img = masker.inverse_transform(ic)
    # Use the 4 terms weighted most as a title
    important_terms = np.array(col_names)[np.argsort(ic_terms)[-4:]]
    display = plot_stat_map(ic_img, threshold=ic_thr, colorbar=False,
                            bg_img=target_img)
    display.title(', '.join(important_terms[::-1]), size=16)

# Done.
plt.show()
