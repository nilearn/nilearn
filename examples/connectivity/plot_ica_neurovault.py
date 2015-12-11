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


### Get data ##################################################################
target_img = datasets.load_mni152_template()

# Download 100 matching images
ss_all = datasets.fetch_neurovault(max_images=100,  # Use np.inf for all imgs.
                                   map_types=['F map', 'T map', 'Z map'],
                                   fetch_terms=True)
images, collections = ss_all['images'], ss_all['collections']

### Show neurosynth terms ######################################################
term_scores = ss_all['terms']
terms = list(term_scores.keys())
total_scores = [np.sum(sc[sc > 0]) for sc in term_scores.values()]

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
masker = NiftiMasker(mask_img=mask, target_affine=target_img.affine,
                     target_shape=target_img.shape, memory='nilearn_cache')

# Make a list of 3D images (4D => list of 3D).
decimated_imgs = concat_niimgs([target_img] + [im['local_path'] for im in images])
X = masker.fit_transform(decimated_imgs)

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
