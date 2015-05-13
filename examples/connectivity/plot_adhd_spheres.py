"""
Multi-subject connectome for ADHD
=================================

This example estimates a connectome from spheres in the brain. The coordinates
of the spheres have been taken from a meta-analysis on neurosynth website.

"""
import matplotlib.pyplot as plt

from nilearn import plotting


n_subjects = 4  # subjects to consider for group-sparse covariance (max: 40)

# Fetching datasets ###########################################################
print("-- Fetching datasets ...")
from nilearn import datasets
msdl_atlas_dataset = datasets.fetch_msdl_atlas()
adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data


# Extracting region signals ###################################################
from nilearn import image
from nilearn import input_data

from sklearn.externals.joblib import Memory
mem = Memory('nilearn_cache')

adhd_coords = [(-36, 22, -4), (42, 22, -4), (14, 10, 2), (-10, 16, -4)]

masker = input_data.NiftiSpheresMasker(
    adhd_coords, radius=8,
    detrend=True, standardize=True,
    low_pass=None, high_pass=0.01, t_r=2.5,
    memory=mem, memory_level=1, verbose=2)
masker.fit()

subject_time_series = []
func_filenames = adhd_dataset.func
confound_filenames = adhd_dataset.confounds
for func_filename, confound_filename in zip(func_filenames,
                                            confound_filenames):
    print("Processing file %s" % func_filename)

    # Computing some confounds
    hv_confounds = mem.cache(image.high_variance_confounds)(
        func_filename)

    region_ts = masker.transform(func_filename,
                                 confounds=[hv_confounds, confound_filename])
    subject_time_series.append(region_ts)

# Computing group-sparse precision matrices ###################################
print("-- Computing group-sparse precision matrices ...")
from nilearn.group_sparse_covariance import GroupSparseCovarianceCV
gsc = GroupSparseCovarianceCV(verbose=2)
gsc.fit(subject_time_series)

# Displaying results ##########################################################
atlas_imgs = image.iter_img(msdl_atlas_dataset.maps)
atlas_region_coords = [plotting.find_xyz_cut_coords(img) for img in atlas_imgs]

title = "GroupSparseCovariance"
plotting.plot_connectome(-gsc.precisions_[..., 0],
                         adhd_coords, edge_threshold='90%',
                         title=title)
plt.show()
