"""
NeuroVault meta-analysis of stop-go paradigm studies.
=====================================================

This example shows how to download statistical maps from
NeuroVault

See :func:`nilearn.datasets.fetch_neurovault`
documentation for more details.

"""
import scipy

from nilearn.datasets import neurovault as nv
from nilearn.image import new_img_like, load_img, math_img


######################################################################
# Fetch images for "successful stop minus go"-like protocols.

# Only 7 images match our critera; set max_images to 7
# so that if we already have them we won't look for more.
nv_data = nv.fetch_neurovault(
    max_images=7,
    cognitive_paradigm_cogatlas=nv.Contains('stop signal'),
    contrast_definition=nv.Contains('succ', 'stop', 'go'),
    map_type='T map')

images_meta = nv_data['images_meta']
collections = nv_data['collections_meta']

######################################################################
# Visualize the data

from nilearn import plotting

for im in images_meta:
    plotting.plot_glass_brain(im['absolute_path'],
                              title='image {0}: {1}'.format(im['id'],
                                    im['contrast_definition']))

######################################################################
# Compute statistics


def t_to_z(t_scores, deg_of_freedom):
    p_values = scipy.stats.t.sf(t_scores, df=deg_of_freedom)
    z_values = scipy.stats.norm.isf(p_values)
    return z_values


# Compute z values
mean_maps = []
z_imgs = []
ids = set()
print("\nComputing maps...")
for collection in [col for col in collections
                   if not(col['id'] in ids or ids.add(col['id']))]:
    print("\n\nCollection {0}:".format(collection['id']))

    # convert t to z
    image_z_niis = []
    for this_meta in images_meta:
        if this_meta['collection_id'] != collection['id']:
            # We don't want to load this image
            continue
        # Load and validate the downloaded image.
        nii = load_img(this_meta['absolute_path'])
        deg_of_freedom = this_meta['number_of_subjects'] - 2
        print("     Image {1}: degrees of freedom: {2}".format(
            "", this_meta['id'], deg_of_freedom))

        # Convert data, create new image.
        z_img = new_img_like(nii,
                    t_to_z(nii.get_data(), deg_of_freedom=deg_of_freedom))

        z_imgs.append(z_img)
        image_z_niis.append(nii)


######################################################################
# Plot the combined z maps

cut_coords = [-15, -8, 6, 30, 46, 62]
nii = math_img('np.sum(z_imgs, axis=3) / np.sqrt(z_imgs.shape[3])',
               z_imgs=z_imgs)

plotting.plot_stat_map(nii, display_mode='z', threshold=6,
                        cut_coords=cut_coords, vmax=12)


plotting.show()
