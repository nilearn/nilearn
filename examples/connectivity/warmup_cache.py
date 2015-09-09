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
from nilearn.decomposition import SparsePCA
from nilearn.decomposition.base import DecompositionEstimator
from nilearn_sandbox.plotting.pdf_plotting import plot_to_pdf
from nilearn_sandbox._utils.map_alignment import align_list_with_last_nii

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn.plotting import plot_prob_atlas, plot_stat_map

def warmup(n_jobs=6, dataset='adhd',n_subjects=40):
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

if __name__ == '__main__':
    t0 = time.time()
    warmup(n_jobs=20, dataset='hcp', n_subjects=500)
    time = time.time() - t0
    print('Total_time : %f s' % time)
