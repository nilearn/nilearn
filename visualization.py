### NIfTI: haxby ##############################################################

# Haxby: Fetch data
from nisl import datasets
haxby = datasets.fetch_haxby()

# Haxby: Get the files relative to this dataset
files = haxby.files
bold = files[1]

# Haxby: Load the NIfTI data
import nibabel as ni
data = ni.load(bold).get_data()

# Haxby: Visualization
import numpy as np
from matplotlib import pyplot as pl
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

### Matlab: kamitani ##########################################################

# Kamitani: Fetch data
from nisl import datasets
kami = datasets.fetch_kamitani()

# Kamitani: Load matlab data
from scipy import io
mat = io.loadmat(kami.files[0], struct_as_record=True)
# Kamitani: get the scans for random images (20 sessions)
data = mat['D']['data'].flat[0]['random'].flat[0].squeeze()
# Kamitani: take the data of the first scan of the first session
scan = data[0][:, 0]

# Kamitani: voxels are flattened, and there is an index (volInd) to get their
# 3D coordinates. We have to use volInd and xyz (which gives MNI coordinates)
# to get back to a 3D matrix
ijk = mat['D']['xyz'].flat[0] / 3 - 0.5 + [[32], [32], [15]]
volInd = mat['D']['volInd'].flat[0].squeeze()
fullscan = np.zeros((64, 64, 30))
for i, v in enumerate(scan):
    fullscan[tuple(ijk[:, volInd[i]])] = v

# Kamitani: Visualization
import numpy as np
from matplotlib import pyplot as pl
pl.figure()
pl.subplot(131)
pl.axis('off')
pl.title('Coronal')
pl.imshow(np.rot90(fullscan[:, 8, :]), interpolation='nearest',
    cmap=pl.cm.gray)
pl.subplot(132)
pl.axis('off')
pl.title('Sagittal')
pl.imshow(np.rot90(fullscan[21, :, :]), interpolation='nearest',
    cmap=pl.cm.gray)
pl.subplot(133)
pl.axis('off')
pl.title('Axial')
pl.imshow(np.rot90(fullscan[:, :, 12]), interpolation='nearest',
    cmap=pl.cm.gray)
pl.show()
