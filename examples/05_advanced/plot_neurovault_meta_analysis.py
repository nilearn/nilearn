"""
NeuroVault meta-analysis of stop-go paradigm studies.
=====================================================

This example shows how to download statistical maps from
NeuroVault

See :func:`nilearn.datasets.fetch_neurovault_ids`
documentation for more details.

"""
# Author: Ben Cipollini
# License: BSD
import scipy

from nilearn.datasets import fetch_neurovault_ids
from nilearn import plotting
from nilearn.image import new_img_like, load_img, math_img


######################################################################
# Fetch images for "successful stop minus go"-like protocols.
# -----------------------------------------------------------

# These are the images we are interested in,
# in order to save time we specify their ids explicitly.
stop_go_image_ids = (151, 3041, 3042, 2676, 2675, 2818, 2834)

# These ids were determined by querying neurovault like this:

# from nilearn.datasets import fetch_neurovault, neurovault

# nv_data = fetch_neurovault(
#     max_images=7,
#     cognitive_paradigm_cogatlas=neurovault.Contains('stop signal'),
#     contrast_definition=neurovault.Contains('succ', 'stop', 'go'),
#     map_type='T map')

# print([meta['id'] for meta in nv_data['images_meta']])


nv_data = fetch_neurovault_ids(image_ids=stop_go_image_ids)

images_meta = nv_data['images_meta']
collections = nv_data['collections_meta']

######################################################################
# Visualize the data
# ------------------

print('\nplotting glass brain for collected images\n')

for im in images_meta:
    plotting.plot_glass_brain(
        im['absolute_path'],
        title='image {0}: {1}'.format(im['id'], im['contrast_definition']))

######################################################################
# Compute statistics
# ------------------


def t_to_z(t_scores, deg_of_freedom):
    p_values = scipy.stats.t.sf(t_scores, df=deg_of_freedom)
    z_values = scipy.stats.norm.isf(p_values)
    return z_values


# Compute z values
mean_maps = []
z_imgs = []
current_collection = None

print("\nComputing maps...")


# convert t to z for all images
for this_meta in images_meta:
    if this_meta['collection_id'] != current_collection:
        print("\n\nCollection {0}:".format(this_meta['id']))
        current_collection = this_meta['collection_id']

    # Load and validate the downloaded image.
    t_img = load_img(this_meta['absolute_path'])
    deg_of_freedom = this_meta['number_of_subjects'] - 2
    print("     Image {1}: degrees of freedom: {2}".format(
        "", this_meta['id'], deg_of_freedom))

    # Convert data, create new image.
    z_img = new_img_like(
        t_img, t_to_z(t_img.get_data(), deg_of_freedom=deg_of_freedom))

    z_imgs.append(z_img)


######################################################################
# Plot the combined z maps
# ------------------------

cut_coords = [-15, -8, 6, 30, 46, 62]
meta_analysis_img = math_img(
    'np.sum(z_imgs, axis=3) / np.sqrt(z_imgs.shape[3])',
    z_imgs=z_imgs)

plotting.plot_stat_map(meta_analysis_img, display_mode='z', threshold=6,
                       cut_coords=cut_coords, vmax=12)


plotting.show()
