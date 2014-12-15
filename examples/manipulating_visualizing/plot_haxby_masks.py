"""
Plot Haxby masks
=================

Small script to plot the masks of the Haxby dataset.
"""
from scipy import linalg
import matplotlib.pyplot as plt

from nilearn import datasets
data = datasets.fetch_haxby()

# Build the mean image because we have no anatomic data
from nilearn import image
mean_img = image.mean_img(data.func[0])

z_slice = -24
from nilearn.image.resampling import coord_transform
affine = mean_img.get_affine()
_, _, k_slice = coord_transform(0, 0, z_slice,
                                linalg.inv(affine))
k_slice = round(k_slice)

fig = plt.figure(figsize=(4, 5.4), facecolor='k')

from nilearn.plotting import plot_anat
display = plot_anat(mean_img, display_mode='z', cut_coords=[z_slice],
                    figure=fig)
display.add_contours(data.mask_vt[0], contours=1, antialiased=False,
                     linewidths=4., levels=[0], colors=['red'])
display.add_contours(data.mask_house[0], contours=1, antialiased=False,
                     linewidths=4., levels=[0], colors=['blue'])
display.add_contours(data.mask_face[0], contours=1, antialiased=False,
                     linewidths=4., levels=[0], colors=['limegreen'])

# We generate a legend using the trick described on
# http://matplotlib.sourceforge.net/users/legend_guide.httpml#using-proxy-artist
from matplotlib.patches import Rectangle
p_v = Rectangle((0, 0), 1, 1, fc="red")
p_h = Rectangle((0, 0), 1, 1, fc="blue")
p_f = Rectangle((0, 0), 1, 1, fc="limegreen")
plt.legend([p_v, p_h, p_f], ["vt", "house", "face"])

plt.show()
