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


from sklearn.model_selection import StratifiedShuffleSplit

import nilearn.surface as ns
import nilearn.decoding

###
### Temporary copied the fetcher
###
from nilearn.datasets.utils import (_get_dataset_dir, _fetch_files, _get_dataset_descr)
from sklearn.datasets.base import Bunch
def fetch_surf_tva_localizer(data_dir=None, verbose=1, resume=True):
    """Download the data from one subject of the InterTVA dataset,
        for the event-related voice localizer run.
        The data provided here is a set of beta maps each estimated
        for one of the 144 audio stimuli presented to the subject,
        and projected onto the cortical surface in the fsaverage5
        template space, for the left hemisphere.
        It is used for the example of the cortex-based searchlight decoding.

    Parameters
    ----------
    data_dir: str, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    verbose: int, optional (default 1)
        Defines the level of verbosity of the output.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
         - 'func_left': Paths to Gifti files containing resting state
                        time series left hemisphere
         - 'phenotypic': array containing tuple with subject ID, age,
                         dominant hand and sex for each subject.
         - 'description': data description of the release and references.

    References
    ----------
    :Download: https://openneuro.org/datasets/ds001771/

    """

    # Preliminary checks and declarations
    dataset_name = 'intertva_localizer_surface'
    #data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
    #                            verbose=verbose)
    intertva_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)
    dataset_dir = _get_dataset_dir('InterTVA_sub-41_surfdata', data_dir=intertva_dir)
    url = 'https://cloud.int.univ-amu.fr/index.php/s/29fJPH5Gm2H9Pxj/download'

    # Dataset description
    fdescr = _get_dataset_descr(dataset_name)

    # First, get the metadata, i.e the labels to be used for the searchlight decoding
    label_path = 'labels_voicelocalizer_voice_vs_nonvoice.tsv'
    #label_file = (label_path, url + 'InterTVA_sub-41_surfdata/{}'.format(label_path),
    #              {'move': label_path})
    label_file = (label_path, url , {'uncompress': True})

    label_localpath = _fetch_files(dataset_dir, [label_file], resume=resume,verbose=verbose)[0]
    labels = np.genfromtxt(label_localpath, dtype='str', skip_header=1)[:,0]

    # Secondly, get the mask of a region of interest for this task
    mask_path = 'fsaverage5.lh.stg_sts.gii'
    mask_file = (mask_path, url, {'uncompress': True})
    mask = _fetch_files(dataset_dir, [mask_file], resume=resume, verbose=verbose)[0]

    # Finally, get the data itself, i.e the 144 single-trial beta maps projected on the surface
    gifti_file_list = []
    for current_trial in range(144):
        gifti_path = 'fsaverage5.lh.beta_{:04d}.gii'.format(current_trial+1)
        gifti_file = (gifti_path, url, {'uncompress': True})
        gifti_file_list.append(gifti_file)

    giftis = _fetch_files(dataset_dir, gifti_file_list, resume=resume, verbose=verbose)
    #giftis = []

    return Bunch(func_left=giftis,
                 mask=mask,
                 phenotypic=labels,
                 description=fdescr)


###
### End: Temporary copied the fetcher
###

### Getting the data ###

'''# probably temporary until a better way to get the data is set up
# get the archive
import urllib
intertva_url = 'https://cloud.int.univ-amu.fr/index.php/s/j8Z84GRXRGBMqec/download'
# future url: https://osf.io/eaq9j/download
tar_fname = 'surf_data.tar.gz'
urllib.request.urlretrieve(intertva_url,tar_fname)

# unpack the data
tar = tarfile.open(tar_fname, "r:gz")
tar.extractall()
tar.close()

# Define the labels to be decoded
subdir = 'InterTVA_sub-41_surfdata/'
labels_df = pandas.read_csv(op.join(subdir,'labels_voicelocalizer_voice_vs_nonvoice.tsv'), sep='\t')
y = np.array(labels_df['label'])
'''

# Load the mesh which will serve to define the searchlight spheres
# Here, we use the freesurfer sphere, sampled at a coarse resolution
# to limit the computational cost (fsaverage5)
sphere_mesh = nilearn.datasets.struct.fetch_surf_fsaverage('fsaverage5_sphere')
mesh_path = sphere_mesh.sphere_left
mesh_data = ns.load_surf_mesh(mesh_path)

# Prepare the matrix containing the functional data
data = fetch_surf_tva_localizer()
y = data.phenotypic
# build list of beta maps
beta_flist = data.func_left

# read the surfacic beta maps
# each beta was computed to estimate the amplitude of the response for a single trial
alldata_singletrials = []
for tex_path in beta_flist:
    tex_data = ns.load_surf_data(tex_path)
    alldata_singletrials.append(tex_data)
X = np.array(alldata_singletrials).T
print(X.shape)


# Define the analysis mask
fullbrain = True
if fullbrain:
    # compute a "brain" mask (i.e the set of vertices where there is data)
    surfmask_tex_data = X.sum(1).astype(np.float32)
    surfmask_tex_data[surfmask_tex_data!=0] = 1.
else:
    # reading ROI surface mask
    surfmask_path = data.mask
    surfmask_tex_data = ns.load_surf_data(surfmask_path)

# Setting up the surfacic searchlight

# Make processing parallel
n_jobs = 2

# Set the radius of the searchlight sphere that will scan the mesh
radius = 6

# Define cros-validation
sss = StratifiedShuffleSplit(n_splits=50, test_size=0.15)

# Define the searchlight "estimator"
searchlight = nilearn.decoding.SurfSearchLight(mesh_data,
                                               surfmask_tex_data,
                                               process_surfmask_tex=surfmask_tex_data,
                                               radius=radius,
                                               verbose=1,
                                               n_jobs=n_jobs,
                                               cv=sss)

# Run the searchlight decoding
# this can take time, depending mostly on the size of the mesh, the number of cross-validation splits and the radius
searchlight.fit(X, y)

# Plot the results with nilearn surface tools
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()
from nilearn import plotting
plotting.plot_surf_stat_map(fsaverage.infl_left, searchlight.scores_, hemi='left',
                            title='Accuracy map, left hemisphere', colorbar=True,
                            threshold=0.4, bg_map=fsaverage.sulc_right)
