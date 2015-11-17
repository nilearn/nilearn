"""
NeuroVault cross-study ICA maps.
=====================================================================

This example shows how to download statistical maps from
NeuroVault, label them with NeuroSynth terms,
and compute ICA components across all the maps.

See :func:`nilearn.datasets.fetch_neurovault` documentation for more details.
"""
# Outline


""" Code to grab the data from NeuroVault, and compute a map of
frequency of activation in the brain.
"""
# Authors: Chris Filo Gorgolewski, Gael Varoquaux
# License: BSD

import json
import io
import os

import numpy as np
from joblib import Memory
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.feature_extraction import DictVectorizer

import nibabel as nib
from nilearn import datasets
from nilearn.datasets.utils import _fetch_files, _get_dataset_dir
from nilearn.image import resample_img
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_background_mask, _extrapolate_out_mask
from nilearn.plotting import plot_stat_map


mem = Memory('cache', verbose=0)


@mem.cache
def faulty_ids():
    # The following maps are not brain maps
    faulty_ids = [96, 97, 98]
    # And the following are crap
    faulty_ids.extend([338, 339])
    # 335 is a duplicate of 336
    faulty_ids.extend([335, ])
    # Now remove -3360, -3362, and -3364, that are mean images, and not Z
    # scores
    faulty_ids.extend([-3360, -3362, -3364])
    # Now remove images that are ugly, or obviously not z maps:
    faulty_ids.extend([1202, 1163, 1931, 1101, 1099])

    return faulty_ids


@mem.cache
def splitext(p):
    if p.endswith('.nii.gz'):
        return p[:-7], '.nii.gz'
    return os.path.splitext(p)


@mem.cache
def resample_and_expand_image(src_nii, target_nii, out_path=None, force=False):
    """Resample src_img to target_img affine, turn 4D into list of 3D.

    Optionally save to out_path.
    """
    data = src_nii.get_data().squeeze()
    data[np.isnan(data)] = 0
    assert np.abs(data).max() > 0

    # print('working on %s, %s' % (src_nii.get_filename(), len(np.unique(data))))
    if out_path is not None:
        out_stem, out_ext = splitext(out_path)
        out_paths = [out_path] if len(data.shape) == 3 else \
                    ['%s-%d%s' % (out_stem, ii, out_ext)
                     for ii in range(data.shape[-1])]
        if np.all([os.path.exists(p) for p in out_paths]):
            return [nib.load(p) for p in out_paths]

    # Copy the image and mkae a background mask.
    src_nii = nib.Nifti1Image(data, src_nii.get_affine(),
                              header=src_nii.get_header())
    bg_mask = compute_background_mask(src_nii).get_data()

    # Test if the image has been masked:
    out_of_mask = data[np.logical_not(bg_mask)]
    if np.all(np.isnan(out_of_mask)) or len(np.unique(out_of_mask)) == 1:
        # Need to extrapolate
        data = _extrapolate_out_mask(data.astype(np.float), bg_mask,
                                     iterations=3)[0]
    src_nii = nib.Nifti1Image(data, src_nii.get_affine(),
                              header=src_nii.get_header())
    del out_of_mask, bg_mask

    # Resampling the file to target and saving the output in the "resampled"
    # folder
    resampled_nii = resample_img(src_nii, target_nii.get_affine(),
                                 target_nii.shape)
    resampled_nii = nib.Nifti1Image(resampled_nii.get_data().squeeze(),
                                    resampled_nii.get_affine(),
                                    header=src_nii.get_header())

    # Decimate 4D files.
    if len(resampled_nii.shape) == 3:
        resampled_niis = [resampled_nii]
    else:
        resampled_niis = []
        resampled_data = resampled_nii.get_data()
        for index in range(resampled_nii.shape[3]):
            # First save the files separately
            this_nii = nib.Nifti1Image(resampled_data[..., index],
                                       resampled_nii.get_affine())
            resampled_niis.append(this_nii)

    # Save the result
    if out_path:
        for nii, nii_path in zip(resampled_niis, out_paths):
            nib.save(nii, nii_path)

    return resampled_niis


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
cfilts = [lambda col: col.get('id') not in [16]]  # [lambda col: col['DOI'] is not None]
imfilts = [lambda im: np.any([im.get('map_type', '').startswith(stat_type)
                              for stat_type in ["Z", "F", "T"]]),
           lambda im: im.get('id') not in faulty_ids(),
           lambda im: im.get('perc_bad_voxels', 0.) < 100.]

# Download up to 100 matches
ss_all = datasets.fetch_neurovault(max_images=np.inf,
                                   collection_filters=cfilts,
                                   image_filters=imfilts)
images, collections = ss_all['images'], ss_all['collections']

# -------------------------------------------
# Resample the images
target_nii = datasets.load_mni152_template()
@mem.cache
def rne_images(images, target_nii):
    resampled_niis = []
    for ii, image in enumerate(images):
        if ii % 100 == 0:
            print("Resampling image %d-%d of %d..." % (
                ii + 1, min(ii + 100, len(images)), len(images)))
        src_nii = nib.load(image['local_path'])
        file_path, ext = splitext(image['local_path'])
        resample_path = '%s-resampled%s' % (file_path, ext)
        resampled_niis += resample_and_expand_image(src_nii, target_nii, resample_path)
    if len(images) != len(resampled_niis):
        print("After resampling, %d images => %d "
              "(each time point became an image)" % (
                  len(images), len(resampled_niis)))
    return resampled_niis
resampled_niis = rne_images(images, target_nii)

# -------------------------------------------
# Get the neurosynth terms; returns a dict of terms, with
#   a vector of values (1 per image)
@mem.cache
def get_terms(images):
    terms = get_neurosynth_terms(images, _get_dataset_dir('neurosynth'))
    good_terms = dict([(t, v) for t, v in terms.items()
                       if np.sum(v[v > 0]) > 0.])
    return good_terms
good_terms = get_terms(images)

print("Top 100 neurosynth terms:")
sort_idx = np.argsort([np.sum(v[v > 0]) for v in good_terms.values()])
for term_idx in sort_idx[-100:]:
    # Eliminate negative values
    term = list(good_terms.keys())[term_idx]
    vec = good_terms[term]
    vec[vec < 0] = 0
    print('\t%-25s: %.4e' % (term, np.sum(vec)))

# -------------------------------------------
# Get the grey matter mask. Cheat by pulling from github :)
@mem.cache
def get_mask():
    print("Downloading and resampling grey matter mask.")
    url = 'https://github.com/NeuroVault/neurovault_analysis/raw/master/gm_mask.nii.gz'
    mask = _fetch_files(_get_dataset_dir('neurovault'),
                        (('gm_mask.nii.gz', url, {}),))[0]
    mask = resample_and_expand_image(nib.load(mask), target_nii)[0]
    mask = nib.Nifti1Image((mask.get_data() >= 0.5).astype(int),
                           mask.get_affine(), mask.get_header())
    return mask
mask = get_mask()

# -------------------------------------------
# Do the ICA transform.
masker = NiftiMasker(mask_img=mask, memory=mem)
@mem.cache
def mask_images(resampled_niis, masker):
    X = masker.fit_transform(resampled_niis)
    return X
X = mask_images(resampled_niis, masker)

@mem.cache
def run_ica(X):
    print("Running ICA; may take time...")
    fast_ica = FastICA(n_components=20, random_state=42)
    ica_maps = fast_ica.fit_transform(X.T).T
    return fast_ica, ica_maps
fast_ica, ica_maps = run_ica(X)
# ica_img = masker.inverse_transform(ica_maps)
# ica_img.to_filename('ica.nii.gz')

# -------------------------------------------
# Relate ICA to terms
term_matrix = np.asarray(list(good_terms.values()))
# Don't use the transform method as it centers the data
ica_terms = np.dot(term_matrix, fast_ica.components_.T).T
col_names = list(good_terms.keys())

# -------------------------------------------
# Generate figures
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
                            bg_img=target_nii)
    display.title(', '.join(important_terms[::-1]), size=16)

# Done.
plt.show()
