# Fetch dataset and load data
from nilearn import datasets
import nibabel
import numpy as np


z = 26

# Fetch dataset and restrict labels to face and houses
haxby = datasets.fetch_haxby(n_subjects=1)
img = nibabel.load(haxby.func[0])
data = img.get_data()
affine = img.get_affine()
labels = np.loadtxt(haxby.session_target[0], delimiter=' ',
                    usecols=[0], dtype=basestring, skiprows=1)

# Create a function to display an axial slice
import matplotlib.pyplot as plt
import numpy as np


def display_axial(brain, i, cmap='hot'):
    plt.figure()
    plt.imshow(brain[:, :, i].T, origin='lower', interpolation='nearest',
               cmap=cmap)
    plt.axis('off')
    plt.show()


# Show mean slice
display_axial(data.mean(axis=-1), z)

# Smoothing
from nilearn.masking import _smooth_array
data = _smooth_array(data, affine, fwhm=6)
display_axial(data.mean(axis=-1), z)

# Run a T-test for face and houses
from scipy import stats
tv, pv = stats.ttest_ind(data[..., labels == 'face'],
                         data[..., labels == 'house'], axis=-1)
pv = - np.log10(pv)
pv[np.isnan(pv)] = 0.
pv[pv > 10] = 10
display_axial(pv, z)

# Thresholding
pv[pv < 4] = 0
display_axial(pv, z)

# Binarization and intersection with VT mask
pv = (pv != 0)
vt = nibabel.load(haxby.mask_vt[0]).get_data().astype(bool)
pv = np.logical_and(pv, vt)
display_axial(pv, z)

# Dilation
from scipy.ndimage import binary_dilation
pv = binary_dilation(pv)
display_axial(pv, z)

# Identification of connected components
from scipy.ndimage import label
labels, _ = label(pv)
display_axial(labels, z, 'gnuplot')
