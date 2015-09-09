import glob
from os.path import join
import os
import time
import pickle
import datetime
from joblib import delayed, Parallel
import numpy as np
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn import datasets
from nilearn.decomposition import SparsePCA, DictLearning
from nilearn.decomposition.base import DecompositionEstimator
from nilearn_sandbox.plotting.pdf_plotting import plot_to_pdf
from nilearn_sandbox._utils.map_alignment import align_list_with_last_nii

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn.plotting import plot_prob_atlas, plot_stat_map


def compare(x, y):
    if len(x) < len(y):
        return -1
    elif len(x) == len(y):
        return cmp(x, y)
    else:
        return 1


def dump_debug_to_pdf(estimator, output):
    print('[Example] Dropping debug information')
    a4_size = (8.27, 11.69)
    size = estimator.debug_info_[0].shape[0]
    fig, axes = plt.subplots(3, 1, figsize=a4_size, sharex=True)
    titles = ['Residuals', 'Sparsity', 'Voxels trajectories']
    ylabels = ['Value', 'Value', 'Voxel value']
    for i, (ax, data) in enumerate(zip(axes, estimator.debug_info_)):
        ax.plot(data)
        ax.set_xlim(0, size)
        ax.set_title(titles[i])
        if i == 2:
            ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabels[i])
    plt.savefig(join(output, 'debug.pdf'))
    plt.close(fig)
    with open(join(output, 'debug.txt'), 'w+') as f:
        if hasattr(estimator, 'score_'):
            f.write('Component score :')
            f.write(str(estimator.score_))
            f.write('\n')
        f.write('Timings :')
        f.write("Math %f - IO %f" % (estimator.time_[0], estimator.time_[1]))
        f.write('\n')
    evolution = sorted(glob.glob(join(output, 'debug', 'components_*.nii.gz')),
                       compare)
    with PdfPages(join(output, 'evolution.pdf')) as pdf:
        for i in range(0, len(evolution), 5):
            fig, axes = plt.subplots(5, 1, figsize=a4_size, squeeze=False)
            axes = axes.reshape(-1)
            for j, ax in enumerate(axes):
                if i + j < len(evolution):
                    plot_prob_atlas(evolution[i + j], axes=ax)
                else:
                    ax.axis('off')
            pdf.savefig(fig)
            plt.close()

    with PdfPages(join(output, 'evolution_single.pdf')) as pdf:
        for i in range(0, len(evolution), 5):
            fig, axes = plt.subplots(5, 1, figsize=a4_size, squeeze=False)
            axes = axes.reshape(-1)
            for j, ax in enumerate(axes):
                if i + j < len(evolution):
                    plot_stat_map(index_img(evolution[i + j], 0), axes=ax)
                else:
                    ax.axis('off')
            pdf.savefig(fig)
            plt.close()


def fit_and_dump(index, estimator, func_filenames, output):
    exp_output = join(output, "experiment_%i" % index)
    os.mkdir(exp_output)
    if type(estimator).__name__:
        debug_folder = join(exp_output, 'debug')
        os.mkdir(debug_folder)
    estimator.debug_folder = debug_folder
    print('[Example] Learning maps using %s model' % type(estimator).__name__)
    t0 = time.time()
    estimator.fit(func_filenames)
    full_time = time.time() - t0
    print('[Example] Dumping results')
    dump_debug_to_pdf(estimator, exp_output)
    components_img = estimator.masker_.inverse_transform(estimator.components_)
    components_filename = join(exp_output, 'components.nii.gz')
    components_img.to_filename(components_filename)
    print('[Example] Preparing pdf')
    plot_to_pdf(components_img, path=join(exp_output, 'components.pdf'))
    timing = np.zeros(3)
    timing[0:2] = estimator.time_
    timing[2] = full_time
    return components_filename, timing


def dump_nii_and_pdf(i, components, dump_dir):
    print("[Example] Dropping aligned components % i" % i)
    filename = join(dump_dir, "experiment_%i.nii.gz" % i)
    components.to_filename(filename)
    print('[Example] Preparing pdf %i' % i)
    plot_to_pdf(components, path=join(dump_dir,
                                    "experiment_%i.pdf" % i))
    return filename


def run_experiment(n_jobs=6, parallel_exp=True, dataset='adhd',
                   init='rsn70', n_subjects=40):
    output = os.path.expanduser('~/work/output/compare')
    temp_dir = os.path.expanduser('~/temp')
    cache_dir = os.path.expanduser('~/nilearn_cache')
    data_dir = os.path.expanduser('~/data')
    output = join(output, datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                           '-%M-%S'))
    try:
        os.makedirs(join(output))
    except:
        pass

    if dataset == 'adhd':
        dataset = datasets.fetch_adhd(n_subjects=min(40, n_subjects))
        mask = None
    elif dataset == 'hcp':
        dataset = datasets.fetch_hcp_rest(n_subjects=n_subjects,
                                          data_dir=data_dir)
        mask = os.path.expanduser('~/data/HCP_mask/mask_img.nii.gz')
    smith = datasets.fetch_atlas_smith_2009()
    if init == 'rsn70':
        dict_init = smith.rsn70
        n_components = 70
    elif init == 'rsn20':
        dict_init = smith.rsn20
        n_components = 20
    else:
        raise ValueError('Unsupported init')
    data_filenames = dataset.func

    print('First functional nifti image (4D) is at: %s' %
          dataset.func[0])

    # This is hacky and should be integrated in the nilearn API in a smooth way
    # Warming up cache with masked images
    print("[Example] Warming up cache")
    decomposition_estimator = DecompositionEstimator(smoothing_fwhm=4.,
                                                     memory=cache_dir,
                                                     mask=mask,
                                                     memory_level=2,
                                                     verbose=10,
                                                     n_jobs=n_jobs)
    decomposition_estimator.fit(data_filenames, preload=True,
                                temp_dir=temp_dir)
    masker = decomposition_estimator.masker_

    estimator_n_jobs = n_jobs if not parallel_exp else 1

    alphas = [3]
    estimators = []
    for alpha in alphas:
        sparse_pca = DictLearning(n_components=n_components, mask=masker,
                                  memory="nilearn_cache", dict_init=dict_init,
                                  reduction_ratio=1,
                                  memory_level=2,
                                  alpha=alpha,
                                  batch_size=20,
                                  verbose=1,
                                  random_state=0,
                                  max_nbytes=None,
                                  n_jobs=estimator_n_jobs,
                                  n_epochs=1)
        estimators.append(sparse_pca)

    with open(join(output, 'estimators'), mode='w+') as f:
        pickle.dump(estimators, f)

    exp_n_jobs = n_jobs if parallel_exp else 1

    res = Parallel(n_jobs=exp_n_jobs, verbose=10)\
        (delayed(fit_and_dump)(index,
                               estimator,
                               data_filenames,
                               output)
         for index, estimator
         in enumerate(estimators))

    components_filename, timings_list = zip(*res)
    timings = np.zeros((len(estimators), 3))
    for i, timing in enumerate(timings_list):
        timings[i] = np.array(timing)
    np.save(join(output, 'timings'), timings)

    if len(estimators) > 1:
        print("Performing alignment")
        map_masker = MultiNiftiMasker(mask_img=masker.mask_img_,).fit()
        components_list = align_list_with_last_nii(map_masker,
                                                   components_filename)
        comparison_dir = join(output, "comparison")
        os.mkdir(comparison_dir)
        Parallel(n_jobs=n_jobs)(
            delayed(dump_nii_and_pdf)(i, components, comparison_dir)
            for i, components in enumerate(components_list))

if __name__ == '__main__':
    t0 = time.time()
    run_experiment(n_jobs=1, dataset='adhd')
    time = time.time() - t0
    print('Total_time : %f s' % time)
