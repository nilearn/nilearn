"""
This file contains a bunch of functions run via __main__().
The functions represent feature comprehensive examples
to visualize, inspect, and test the functionality
of nistats.reporting.make_glm_reports().

Disable any of the function calls in the __main__()
to run a specific script and save time.
"""
import os

import numpy as np
import pandas as pd

import nilearn
from nilearn._utils.glm import get_design_from_fslmat
from nilearn.glm.first_level import FirstLevelModel, first_level_from_bids
from nilearn.glm.first_level.design_matrix import \
    make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import resample_to_img
from nilearn.input_data import NiftiSpheresMasker
from nilearn.reporting import make_glm_report

REPORTS_DIR = 'generated_glm_reports'
try:
    os.mkdir(REPORTS_DIR)
except OSError:
    pass


def report_flm_adhd_dmn():  # pragma: no cover
    t_r = 2.
    slice_time_ref = 0.
    n_scans = 176
    pcc_coords = (0, -53, 26)
    adhd_dataset = nilearn.datasets.fetch_adhd(n_subjects=1)
    seed_masker = NiftiSpheresMasker([pcc_coords], radius=10, detrend=True,
                                     standardize=True, low_pass=0.1,
                                     high_pass=0.01, t_r=2.,
                                     memory='nilearn_cache',
                                     memory_level=1, verbose=0)
    seed_time_series = seed_masker.fit_transform(adhd_dataset.func[0])
    frametimes = np.linspace(0, (n_scans - 1) * t_r, n_scans)
    design_matrix = make_first_level_design_matrix(frametimes, hrf_model='spm',
                                                   add_regs=seed_time_series,
                                                   add_reg_names=["pcc_seed"])
    dmn_contrast = np.array([1] + [0] * (design_matrix.shape[1] - 1))
    contrasts = {'seed_based_glm': dmn_contrast}

    first_level_model = FirstLevelModel(t_r=t_r, slice_time_ref=slice_time_ref)
    first_level_model = first_level_model.fit(run_imgs=adhd_dataset.func[0],
                                              design_matrices=design_matrix)

    report = make_glm_report(first_level_model,
                             contrasts=contrasts,
                             title='ADHD DMN Report',
                             cluster_threshold=15,
                             height_control='bonferroni',
                             min_distance=8.,
                             plot_type='glass',
                             report_dims=(1200, 'a'),
                             )
    output_filename = 'generated_report_flm_adhd_dmn.html'
    output_filepath = os.path.join(REPORTS_DIR, output_filename)
    report.save_as_html(output_filepath)
    report.get_iframe()


###########################################################################

def _fetch_bids_data():  # pragma: no cover
    _, urls = datasets.func.fetch_openneuro_dataset_index()

    exclusion_patterns = ['*group*', '*phenotype*', '*mriqc*',
                          '*parameter_plots*', '*physio_plots*',
                          '*space-fsaverage*', '*space-T1w*',
                          '*dwi*', '*beh*', '*task-bart*',
                          '*task-rest*', '*task-scap*', '*task-task*']
    urls = datasets.func.select_from_index(
        urls, exclusion_filters=exclusion_patterns, n_subjects=1)

    data_dir, _ = datasets.func.fetch_openneuro_dataset(urls=urls)
    return data_dir


def _make_flm(data_dir):  # pragma: no cover
    task_label = 'stopsignal'
    space_label = 'MNI152NLin2009cAsym'
    derivatives_folder = 'derivatives/fmriprep'
    models, models_run_imgs, models_events, models_confounds = \
        first_level_from_bids(
            data_dir, task_label, space_label, smoothing_fwhm=5.0,
            derivatives_folder=derivatives_folder)

    model, imgs, _, _ = (
        models[0], models_run_imgs[0], models_events[0],
        models_confounds[0])
    subject = 'sub-' + model.subject_label
    design_matrix = _make_design_matrix_for_bids_feature(data_dir, subject)
    model.fit(imgs, design_matrices=[design_matrix])
    return model, subject


def _make_design_matrix_for_bids_feature(data_dir,
                                         subject):  # pragma: no cover
    fsl_design_matrix_path = os.path.join(
        data_dir, 'derivatives', 'task', subject, 'stopsignal.feat',
        'design.mat')
    design_matrix = get_design_from_fslmat(
        fsl_design_matrix_path, column_names=None)

    design_columns = ['cond_%02d' % i for i in
                      range(len(design_matrix.columns))]
    design_columns[0] = 'Go'
    design_columns[4] = 'StopSuccess'
    design_matrix.columns = design_columns
    return design_matrix


def report_flm_bids_features():  # pragma: no cover
    data_dir = _fetch_bids_data()
    model, subject = _make_flm(data_dir)
    title = 'FLM Bids Features Stat maps'
    report = make_glm_report(model=model,
                             contrasts='StopSuccess - Go',
                             title=title,
                             cluster_threshold=3,
                             )
    output_filename = 'generated_report_flm_bids_features.html'
    output_filepath = os.path.join(REPORTS_DIR, output_filename)
    report.save_as_html(output_filepath)
    report.get_iframe()


###########################################################################

def _pad_vector(contrast_, n_columns):  # pragma: no cover
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))


def report_flm_fiac():  # pragma: no cover
    data = datasets.func.fetch_fiac_first_level()
    fmri_img = [data['func1'], data['func2']]

    from nilearn.image import mean_img
    mean_img_ = mean_img(fmri_img[0])

    design_files = [data['design_matrix1'], data['design_matrix2']]
    design_matrices = [pd.DataFrame(np.load(df)['X']) for df in design_files]

    fmri_glm = FirstLevelModel(mask_img=data['mask'], minimize_memory=True)
    fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrices)

    n_columns = design_matrices[0].shape[1]

    contrasts = {
        'SStSSp_minus_DStDSp': _pad_vector([1, 0, 0, -1], n_columns),
        'DStDSp_minus_SStSSp': _pad_vector([-1, 0, 0, 1], n_columns),
        'DSt_minus_SSt': _pad_vector([-1, -1, 1, 1], n_columns),
        'DSp_minus_SSp': _pad_vector([-1, 1, -1, 1], n_columns),
        'DSt_minus_SSt_for_DSp': _pad_vector([0, -1, 0, 1], n_columns),
        'DSp_minus_SSp_for_DSt': _pad_vector([0, 0, -1, 1], n_columns),
        'Deactivation': _pad_vector([-1, -1, -1, -1, 4], n_columns),
        'Effects_of_interest': np.eye(n_columns)[:5]
    }
    report = make_glm_report(fmri_glm, contrasts,
                             bg_img=mean_img_,
                             height_control='fdr',
                             )
    output_filename = 'generated_report_flm_fiac.html'
    output_filepath = os.path.join(REPORTS_DIR, output_filename)
    report.save_as_html(output_filepath)
    report.get_iframe()


###########################################################################

def _make_design_matrix_slm_oasis(oasis_dataset,
                                  n_subjects):  # pragma: no cover
    age = oasis_dataset.ext_vars['age'].astype(float)
    sex = oasis_dataset.ext_vars['mf'] == b'F'
    intercept = np.ones(n_subjects)
    design_matrix = pd.DataFrame(np.vstack((age, sex, intercept)).T,
                                 columns=['age', 'sex', 'intercept'])
    design_matrix = pd.DataFrame(design_matrix, columns=['age', 'sex',
                                                         'intercept']
                                 )
    return design_matrix


def report_slm_oasis():  # pragma: no cover
    n_subjects = 5  # more subjects requires more memory
    oasis_dataset = nilearn.datasets.fetch_oasis_vbm(n_subjects=n_subjects)
    # Resample the images, since this mask has a different resolution
    mask_img = resample_to_img(nilearn.datasets.fetch_icbm152_brain_gm_mask(),
                               oasis_dataset.gray_matter_maps[0],
                               interpolation='nearest',
                               )
    design_matrix = _make_design_matrix_slm_oasis(oasis_dataset, n_subjects)
    second_level_model = SecondLevelModel(smoothing_fwhm=2.0, mask=mask_img)
    second_level_model.fit(oasis_dataset.gray_matter_maps,
                           design_matrix=design_matrix)

    contrast = [[1, 0, 0], [0, 1, 0]]
    report = make_glm_report(
        model=second_level_model,
        contrasts=contrast,
        bg_img=nilearn.datasets.fetch_icbm152_2009()['t1'],
        height_control=None,
    )
    output_filename = 'generated_report_slm_oasis.html'
    output_filepath = os.path.join(REPORTS_DIR, output_filename)
    report.save_as_html(output_filepath)
    report.get_iframe()


def prefer_parallel_execution(functions_to_be_called):  # pragma: no cover
    try:
        import joblib
        import multiprocessing
    except ImportError:
        print('Joblib not installed, switching to serial execution')
        [run_function(fn) for fn in functions_to_be_called]
    else:
        try:
            import tqdm
        except ImportError:
            inputs = functions_to_be_called
        else:
            inputs = tqdm.tqdm(functions_to_be_called)
        n_jobs = multiprocessing.cpu_count()
        print('Parallelizing execution using Joblib')
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(run_function)(fn) for fn in inputs)


def run_function(fn):  # pragma: no cover
    fn()


def prefer_serial_execution(functions_to_be_called):  # pragma: no cover
    for fn in functions_to_be_called:
        fn()


if __name__ == '__main__':  # pragma: no cover
    functions_to_be_called = [
        report_flm_adhd_dmn,
        report_flm_bids_features,
        report_flm_fiac,
        report_slm_oasis,
    ]
    prefer_parallel_execution(functions_to_be_called)
    # prefer_serial_execution(functions_to_be_called)
