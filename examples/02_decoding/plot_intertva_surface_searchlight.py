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
import os.path as op
import pandas
import tarfile
import glob

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import nibabel.gifti as ng
import nibabel as nb

import nilearn.surface as ns
import nilearn.decoding


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


### Define the analysis mask
fullbrain = False
if fullbrain:
    # compute a "brain" mask (i.e the set of vertices where there is data)
    surfmask = X.sum(1).astype(np.float32)
    surfmask[surfmask!=0] = 1.
    # save this brain mask (this is a dirty way to do it, to be changed...)
    surfmask_tex = nb.load(tex_path)
    surfmask_tex.darrays[0].data = surfmask
    #print(np.flatnonzero(surfmask).size)
    #surfmask_path = op.join(subdir, 'fsaverage5.lh.brainmask.gii')
    #ng.write(surfmask_tex,surfmask_path)
else:
    # reading ROI surface mask
    surfmask_path = op.join(subdir,'fsaverage5.lh.stg_sts.gii')
    surfmask_tex = nb.load(surfmask_path)
    # with random mask of 1000 vertices
    #inds = np.random.permutation(X.shape[0])[np.arange(1000)]
    #surfmask_data = np.zeros(X.shape[0])
    #surfmask_data[inds] = 1
    #darray = ng.GiftiDataArray(data=surfmask_data)
    #surfmask_tex = ng.GiftiImage(darrays=[darray])


### Setting up the surfacic searchlight ###

# Make processing parallel
n_jobs = 2

# Define the cross-validation scheme used for validation.
sss = StratifiedShuffleSplit(n_splits=20, test_size=0.15)

# Define the classifier
logreg = LogisticRegression()

# Set the radius of the searchlight sphere that will scan the mesh
radius = 8

# Define the searchlight "estimator"
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

### Plot the results with nilearn surface tools ###
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()
from nilearn import plotting
plotting.plot_surf_stat_map(fsaverage.infl_left, searchlight.scores_, hemi='left',
                            title='Accuracy map, left hemisphere', colorbar=True,
                            threshold=0.4, bg_map=fsaverage.sulc_right)


