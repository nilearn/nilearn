from os.path import join
import os
import pickle
import datetime
from nilearn import datasets
from nilearn.decomposition import SparsePCA
# For linear assignment (should be moved in non user space...)

output = '/home/arthur/work/output/adhd'
output = join(output, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(join(output, 'intermediary'))
except:
    pass

# dataset = datasets.fetch_hcp_rest(data_dir='/volatile3', n_subjects=1)
# mask = dataset.mask if hasattr(dataset, 'mask') else None
dataset = datasets.fetch_adhd(n_subjects=40)
smith = datasets.fetch_atlas_smith_2009()
dict_init = smith.rsn20
n_components = 20

func_filenames = dataset.func  # list of 4D nifti files for each subject

# from sandbox.utils import output
# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      dataset.func[0])  # 4D data


sparse_pca = SparsePCA(n_components=n_components,
                       l1_ratio=11,
                       smoothing_fwhm=4.,
                       reduction_ratio=1.,
                       batch_size=10,
                       n_epochs=1,
                       alpha=0.1,
                       dict_init=smith.rsn20,
                       memory="nilearn_cache",
                       memory_level=3,
                       shuffle=True,
                       verbose=2, random_state=0)
estimator = sparse_pca


print('[Example] Learning maps using %s model'
      % type(estimator).__name__)
estimator.fit(func_filenames)
print('[Example] Dumping results')
components_img = estimator.masker_.inverse_transform(estimator.components_)
components_img.to_filename(join(output, 'dict_learning_resting_state.nii.gz'))
with open(join(output, 'parameters'), mode='w+') as f:
    pickle.dump(sparse_pca.get_params(), f)
with open(join(output, 'dataset'), mode='w+') as f:
    pickle.dump(dataset, f)


import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas, find_xyz_cut_coords
from nilearn.plotting.pdf_plotting import plot_to_pdf
from nilearn.image import index_img

print('[Example] Preparing pdf')
plot_to_pdf(components_img, path=join(output, 'components.pdf'))

print('[Example] Displaying')

fig = plt.figure()
cut_coords = find_xyz_cut_coords(index_img(components_img, 6))
plot_prob_atlas(components_img, title="%s" % estimator.__class__.__name__,
                figure=fig, view_type='continuous',
                cut_coords=cut_coords, colorbar=False)

plt.show()
