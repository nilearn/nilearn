"""
NeuroVault meta-analysis of stop-go paradigm studies.
=====================================================================

This example shows how to download statistical maps from
NeuroVault

See :func:`nilearn.datasets.fetch_neurovault` documentation for more details.
"""

import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets

# Define pre-download filters
cfilts = []  # [lambda col: col['DOI'] is not None]
imfilts = [lambda im: "stop signal" in (im.get('cognitive_paradigm_cogatlas') or "").lower(),
           lambda im: np.all([s in (im.get("contrast_definition") or "")
                              for s in ["succ", "stop", "go"]])]


# Download up to 100 matches
ss_all = datasets.fetch_neurovault(collection_filters=cfilts,
                                   image_filters=imfilts)
images, collections = ss_all['images'], ss_all['collections']
print("Paradigms we've downloaded:")
for im in images:
    print("\t%04d: %s" % (im['id'],
                          im['cognitive_paradigm_cogatlas']))

# Show all contrast definitions
print("Contrast definitions for downloaded images:")
for cd in np.unique([im['contrast_definition'] for im in images]):
    print("\t%s" % cd)

# Visualize the content
from nilearn.plotting import plot_glass_brain
for im in images:
    plot_glass_brain(im['local_path'], title='%04d' % im['id'])

# Convert t values to z values
def t_to_z(t_scores, df):  # df == degrees of freedom
    import scipy
    p_values = scipy.stats.t.sf(t_scores, df=df)
    z_values = scipy.stats.norm.isf(p_values)
    return z_values, p_values

# Compute z values
from nilearn.image import mean_img
import nibabel as nb
analysis_dir = '.'

mean_maps = []
p_datas = []
z_datas = []
all_collection_ids = np.unique([im['collection_id'] for im in images])
for collection_id in all_collection_ids:
    print("Collection %d" % collection_id)
    df = collections[collection_id]['number_of_subjects'] - 2
    print("Degrees of freedom = %d" % df)

    # convert t to z
    cur_imgs = [im for im in images if im['collection_id'] == collection_id]
    image_z_niis = []
    for im in cur_imgs:
        # Load and validate the downloaded image.
        nii = nb.load(im['local_path'])
        assert im['map_type'] == 'T map', "We assume T-maps"
        if image_z_niis:
            assert np.allclose(image_z_niis[0].get_affine(),
                               nii.get_affine()), \
                   "We assume all images are in the same space."

        # Convert data, create new image.
        data_z, data_p = t_to_z(nii.get_data(), df=df)
        p_datas.append(data_p)
        z_datas.append(data_z)
        nii_z = nb.Nifti1Image(data_z, nii.get_affine())
        image_z_niis.append(nii)


    mean_map = mean_img(image_z_niis)
    plot_glass_brain(mean_map,
                     title="Collection %04d mean map" % collection_id)
    mean_maps.append(mean_map)

# Fisher's z-score on all maps
from nilearn.plotting import plot_stat_map
def z_map(z_data, affine):
    import math
    cut_coords = [-15, -8, 6, 30, 46, 62]
    z_meta_data = np.array(z_data).sum(axis=0) / math.sqrt(len(z_data))
    nii = nb.Nifti1Image(z_meta_data, affine)
    plot_stat_map(nii, display_mode='z', threshold=6, cut_coords=cut_coords, vmax=12)
z_map(z_datas, mean_maps[0].get_affine())

# Fisher's z-score on combined maps
z_input_datas = [nii.get_data() for nii in mean_maps]
z_map(z_input_datas, mean_maps[0].get_affine())

plt.show()
