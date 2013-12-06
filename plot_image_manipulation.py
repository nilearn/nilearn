# Fetch dataset and load data
from nilearn import datasets
import nibabel
import numpy


z = 26

# Fetch dataset
haxby = datasets.fetch_haxby(n_subjects=1)
labels = numpy.genfromtxt(haxby.session_target[0], skip_header=1, usecols=[0],
                          dtype=basestring)

# Create a function to display an axial slice
import matplotlib.pyplot as plt


def display_axial(brain, i, title, cmap='hot'):
    plt.figure()
    plt.imshow(brain[:, :, i].T.copy(), origin='lower', interpolation='nearest',
               cmap=cmap)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()


# Smoothing
from nilearn.image import smooth
img = smooth(haxby.func[0], fwhm=6)
data = img.get_data()
affine = img.get_affine()
display_axial(data.mean(axis=-1), z, 'Mean EPI')

# Run a T-test for face and houses
from scipy import stats
tv, pv = stats.ttest_ind(data[..., labels == 'face'],
                         data[..., labels == 'house'], axis=-1)
pv = - numpy.log10(pv)
pv[numpy.isnan(pv)] = 0.
pv[pv > 10] = 10
display_axial(pv, z, 'p-values')

# Thresholding
pv[pv < 5] = 0
display_axial(pv, z, 'Thresholded p-values')

# Binarization and intersection with VT mask
pv = (pv != 0)
vt = nibabel.load(haxby.mask_vt[0]).get_data().astype(bool)
pv = numpy.logical_and(pv, vt)
display_axial(pv, z, 'Intersection with ventral temporal mask')

# Dilation
from scipy.ndimage import binary_dilation
pv = binary_dilation(pv)
display_axial(pv, z, 'Dilated mask')

# Identification of connected components
from scipy.ndimage import label
labels, n_labels = label(pv)
display_axial(labels, z, 'Connected components', 'gnuplot')

# Save the result
nibabel.save(nibabel.Nifti1Image(labels, affine), 'mask.nii')
plt.show()
