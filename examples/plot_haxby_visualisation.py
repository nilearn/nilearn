"""
============================
Extract from haxby fMRI data
============================

This is a simple extract from data with no meaning and no filter.
"""
print __doc__

from matplotlib import pyplot as plt

from nisl import datasets

haxby = datasets.fetch_haxby_data()
img = haxby.data[:, 29, :, 0] * haxby.mask[:, 29, :]
plt.imshow(img, cmap=plt.cm.spectral)
