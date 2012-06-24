""" Simple example to show data manipulation and visualization.
"""

### NIfTI: haxby ##############################################################

# Haxby: Fetch data
from nisl import datasets
haxby = datasets.fetch_haxby()

# Haxby: Get the files relative to this dataset
files = haxby.files
bold = files[1]

# Haxby: Load the NIfTI data
import nibabel
data = nibabel.load(bold).get_data()

# Haxby: Visualization
import numpy as np
import pylab as pl
pl.figure()
pl.subplot(131)
pl.axis('off')
pl.title('Coronal')
pl.imshow(np.rot90(data[:, 32, :, 100]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.subplot(132)
pl.axis('off')
pl.title('Sagittal')
pl.imshow(np.rot90(data[15, :, :, 100]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.subplot(133)
pl.axis('off')
pl.title('Axial')
pl.imshow(np.rot90(data[:, :, 32, 100]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.show()


