"""
Cortical surface-based searchlight decoding
===========================================

This is a demo for surface-based searchlight decoding, as described in:
Chen, Y., Namburi, P., Elliott, L.T., Heinzle, J., Soon, C.S., 
Chee, M.W.L., and Haynes, J.-D. (2011). Cortical surface-based 
searchlight decoding. NeuroImage 56, 582–592.

The decoding question addressed here aims at guessing whether the audio
stimulus heard by the subject was vocal or non-vocal. The data is taken
from the InterTVA experiment: https://openneuro.org/datasets/ds001771/
"""

import numpy as np
import nibabel.gifti as ng
import nibabel as nb
import nilearn.surface as ns
import os.path as op
import pandas
import tarfile
import glob


### Getting the data ###
# probably temporary until a better way to get the data is set up
# get the archive
import urllib
intertva_url = 'https://cloud.int.univ-amu.fr/index.php/s/j8Z84GRXRGBMqec/download'
tar_fname = 'surf_data.tar.gz'
urllib.request.urlretrieve(intertva_url,tar_fname)

# unpack the data
tar = tarfile.open(tar_fname, "r:gz")
tar.extractall()
tar.close()

### Define the labels to be decoded ###
subdir = 'InterTVA_sub-41_surfdata/'
labels_df = pandas.read_csv(op.join(subdir,'labels_voicelocalizer_voice_vs_nonvoice.tsv'), sep='\t')
y = np.array(labels_df['label'])


### Load the mesh which will serve to define the searchlight spheres ###
# Here, we use the freesurfer sphere, sampled at a coarse resolution
# to limit the computational cost
mesh_path = op.join(subdir, 'fsaverage5.lh.sphere.surf.gii')
mesh = nb.load(mesh_path)


### Prepare the matrix containing the functional data ###

# build list of beta maps
beta_flist = glob.glob(op.join(subdir,'fsaverage5.lh.beta*.gii'))
beta_flist.sort()

# read the surfacic beta maps
# each beta was computed to estimate the amplitude of the response for a single trial
alldata_singletrials = []
for tex_path in beta_flist:
    # tex = nb.load(tex_path)
    # alldata_singletrials.append(tex.darrays[0].data)
    tex_data = ns.load_surf_data(tex_path)
    alldata_singletrials.append(tex_data)
X = np.array(alldata_singletrials).T
print(X.shape)


### Compute the analysis mask

# compute a "brain" mask (i.e the set of vertices where there is data)
surfmask = X.sum(1).astype(np.float32)
surfmask[surfmask!=0] = 1.
# save this brain mask (this is a dirty way to do it, to be changed...)
surfmask_tex = nb.load(tex_path)
surfmask_tex.darrays[0].data = surfmask
print(np.flatnonzero(surfmask).size)
#surfmask_path = op.join(subdir, 'fsaverage5.lh.brainmask.gii')
#ng.write(surfmask_tex,surfmask_path)

### Setting up the surfacic searchlight ###

# Make processing parallel
n_jobs = 2

# Define the cross-validation scheme used for validation.
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.15)

# Define the classifier
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# Set the radius of the searchlight sphere that will scan the mesh
radius = 3

# Define the searchlight "estimator"
import nilearn.decoding
searchlight = nilearn.decoding.SurfSearchLight(mesh,
                                               surfmask_tex,
                                               estimator=logreg,
                                               process_surfmask_tex=surfmask_tex,
                                               radius=radius,
                                               verbose=1,
                                               n_jobs=n_jobs,
                                               cv=sss)

### Run the searchlight decoding ###

# this can take time, depending mostly on the size of the mesh, the number of cross-validation splits and the radius
searchlight.fit(X, y)


### Save the searchlight accuracy map ###

# here also, this is a dirty way to do it, to be changed...
slscores_tex = nb.load(tex_path)
slscores_tex.darrays[0].data = searchlight.scores_.astype(np.float32)
slscores_path = op.join(subdir, 'fsaverage5.lh.searchlight_accuracy_map.gii')
ng.write(slscores_tex,slscores_path)


### Plot the results with nilearn surface tools ###
# to be written...


