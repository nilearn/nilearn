"""
Example of automatic mask computation
"""

import pylab as pl
from nisl import datasets, io

# Load Haxby dataset
dataset_files = datasets.fetch_haxby()

masker = io.NiftiMasker()
masker.fit(dataset_files.func)

pl.imshow(masker.mask_.get_data()[..., 20])
pl.show()

