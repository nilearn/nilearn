"""
NeuroVault cross-study ICA maps
===============================

This example shows how to download statistical maps from
NeuroVault, label them with NeuroSynth terms,
and compute :term:`ICA` components across all the maps.

See :func:`nilearn.datasets.fetch_neurovault`
documentation for more details.

.. include:: ../../../examples/masker_note.rst

"""
# Author: Ben Cipollini
# License: BSD
# Ported from code authored by Chris Filo Gorgolewski, Gael Varoquaux
# https://github.com/NeuroVault/neurovault_analysis
import warnings

import numpy as np
from nilearn import plotting
from nilearn.datasets import fetch_neurovault, load_mni152_brain_mask
from nilearn.image import smooth_img
from nilearn.maskers import NiftiMasker
from scipy import stats
from sklearn.decomposition import FastICA

######################################################################
# Get image and term data
# -----------------------

# Download images
# Here by default we only download 80 images to save time,
# but for better results I recommend using at least 200.
print(
    "Fetching Neurovault images; "
    "if you haven't downloaded any Neurovault data before "
    "this will take several minutes."
)
nv_data = fetch_neurovault(max_images=30, fetch_neurosynth_words=True)

images = nv_data["images"]
term_weights = nv_data["word_frequencies"]
vocabulary = nv_data["vocabulary"]
if term_weights is None:
    term_weights = np.ones((len(images), 2))
    vocabulary = np.asarray(["Neurosynth is down", "Please try again later"])

# Clean and report term scores
term_weights[term_weights < 0] = 0
total_scores = np.mean(term_weights, axis=0)

print("\nTop 10 neurosynth terms from downloaded images:\n")

for term_idx in np.argsort(total_scores)[-10:][::-1]:
    print(vocabulary[term_idx])

######################################################################
# Reshape and mask images
# -----------------------

print("\nReshaping and masking images.\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", DeprecationWarning)

    mask_img = load_mni152_brain_mask(resolution=2)
    masker = NiftiMasker(
        mask_img=mask_img, memory="nilearn_cache", memory_level=1
    )
    masker = masker.fit()

    # Images may fail to be transformed, and are of different shapes,
    # so we need to transform one-by-one and keep track of failures.
    X = []
    is_usable = np.ones((len(images),), dtype=bool)

    for index, image_path in enumerate(images):
        # load image and remove nan and inf values.
        # applying smooth_img to an image with fwhm=None simply cleans up
        # non-finite values but otherwise doesn't modify the image.
        image = smooth_img(image_path, fwhm=None)
        try:
            X.append(masker.transform(image))
        except Exception as e:
            meta = nv_data["images_meta"][index]
            print(
                f"Failed to mask/reshape image: id: {meta.get('id')}; "
                f"name: '{meta.get('name')}'; "
                f"collection: {meta.get('collection_id')}; error: {e}"
            )
            is_usable[index] = False

# Now reshape list into 2D matrix, and remove failed images from terms
X = np.vstack(X)
term_weights = term_weights[is_usable, :]

######################################################################
# Run ICA and map components to terms
# -----------------------------------

print("Running ICA; may take time...")
# We use a very small number of components as we have downloaded only 80
# images. For better results, increase the number of images downloaded
# and the number of components
n_components = 8
fast_ica = FastICA(n_components=n_components, random_state=0)
ica_maps = fast_ica.fit_transform(X.T).T

term_weights_for_components = np.dot(fast_ica.components_, term_weights)
print("Done, plotting results.")

######################################################################
# Generate figures
# ----------------

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)

    for index, (ic_map, ic_terms) in enumerate(
        zip(ica_maps, term_weights_for_components)
    ):
        if -ic_map.min() > ic_map.max():
            # Flip the map's sign for prettiness
            ic_map = -ic_map
            ic_terms = -ic_terms

        ic_threshold = stats.scoreatpercentile(np.abs(ic_map), 90)
        ic_img = masker.inverse_transform(ic_map)
        important_terms = vocabulary[np.argsort(ic_terms)[-3:]]
        title = f"IC{int(index)}  {', '.join(important_terms[::-1])}"

        plotting.plot_stat_map(
            ic_img, threshold=ic_threshold, colorbar=False, title=title
        )

######################################################################
# As we can see, some of the components capture cognitive or neurological
# maps, while other capture noise in the database. More data, better
# filtering, and better cognitive labels would give better maps

# Done.
plotting.show()
