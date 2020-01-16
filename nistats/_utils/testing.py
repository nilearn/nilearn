import json
import os
import string

import numpy as np
import pandas as pd
from nibabel import Nifti1Image


def _write_fake_fmri_data(shapes, rk=3, affine=np.eye(4)):
    mask_file, fmri_files, design_files = 'mask.nii', [], []
    for i, shape in enumerate(shapes):
        fmri_files.append('fmri_run%d.nii' % i)
        data = np.random.randn(*shape)
        data[1:-1, 1:-1, 1:-1] += 100
        Nifti1Image(data, affine).to_filename(fmri_files[-1])
        design_files.append('dmtx_%d.csv' % i)
        pd.DataFrame(np.random.randn(shape[3], rk),
                     columns=['', '', '']).to_csv(design_files[-1])
    Nifti1Image((np.random.rand(*shape[:3]) > .5).astype(np.int8),
                affine).to_filename(mask_file)
    return mask_file, fmri_files, design_files


def _generate_fake_fmri_data(shapes, rk=3, affine=np.eye(4)):
    fmri_data = []
    design_matrices = []
    for i, shape in enumerate(shapes):
        data = np.random.randn(*shape)
        data[1:-1, 1:-1, 1:-1] += 100
        fmri_data.append(Nifti1Image(data, affine))
        columns = np.random.choice(list(string.ascii_lowercase), size=rk)
        design_matrices.append(pd.DataFrame(np.random.randn(shape[3], rk),
                                            columns=columns))
    mask = Nifti1Image((np.random.rand(*shape[:3]) > .5).astype(np.int8),
                       affine)
    return mask, fmri_data, design_matrices


def _write_fake_bold_img(file_path, shape, rk=3, affine=np.eye(4)):
    data = np.random.randn(*shape)
    data[1:-1, 1:-1, 1:-1] += 100
    Nifti1Image(data, affine).to_filename(file_path)
    return file_path


def _basic_paradigm():
    conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c2', 'c2', 'c2']
    onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
    events = pd.DataFrame({'trial_type': conditions,
                             'onset': onsets})
    return events


def _basic_confounds(length):
    columns = ['RotX', 'RotY', 'RotZ', 'X', 'Y', 'Z']
    data = np.random.rand(length, 6)
    confounds = pd.DataFrame(data, columns=columns)
    return confounds


def _create_fake_bids_dataset(base_dir='', n_sub=10, n_ses=2,
                             tasks=['localizer', 'main'],
                             n_runs=[1, 3], with_derivatives=True,
                             with_confounds=True, no_session=False):
    """Creates a fake bids dataset directory with dummy files.
    Returns fake dataset directory name.

    Parameters
    ----------
    base_dir: string (Absolute path), optional
        Absolute directory path in which to create the fake BIDS dataset dir.
        Default: Current directory.

    n_sub: int, optional
        Number of subject to be simulated in the dataset.
        Default: 10

    n_ses: int, optional
        Number of sessions to be simulated in the dataset.
        Ignored if no_session=True.
        Default: 2

    n_runs: List[int], optional
        Default: [1, 3]

    with_derivatives: bool, optional
        In the case derivatives are included, they come with two spaces and
        descriptions. Spaces are 'MNI' and 'T1w'. Descriptions are 'preproc'
        and 'fmriprep'. Only space 'T1w' include both descriptions.
        Default: True

    with_confounds: bool, optional
        Default: True

    no_session: bool, optional
        Specifying no_sessions will only produce runs and files without the
        optional session field. In this case n_ses will be ignored.
        Default: False

    Returns
    -------
    dataset directory name: string
        'bids_dataset'

    Creates
    -------
        Directory with dummy files
    """
    bids_path = os.path.join(base_dir, 'bids_dataset')
    os.makedirs(bids_path)
    # Create surface bids dataset
    open(os.path.join(bids_path, 'README.txt'), 'w')
    vox = 4
    created_sessions = ['ses-%02d' % label for label in range(1, n_ses + 1)]
    if no_session:
        created_sessions = ['']
    for subject in ['sub-%02d' % label for label in range(1, n_sub + 1)]:
        for session in created_sessions:
            subses_dir = os.path.join(bids_path, subject, session)
            if session == 'ses-01' or session == '':
                anat_path = os.path.join(subses_dir, 'anat')
                os.makedirs(anat_path)
                anat_file = os.path.join(anat_path, subject + '_T1w.nii.gz')
                open(anat_file, 'w')
            func_path = os.path.join(subses_dir, 'func')
            os.makedirs(func_path)
            for task, n_run in zip(tasks, n_runs):
                for run in ['run-%02d' % label for label in range(1, n_run + 1)]:
                    fields = [subject, session, 'task-' + task]
                    if '' in fields:
                        fields.remove('')
                    file_id = '_'.join(fields)
                    if n_run > 1:
                        file_id += '_' + run
                    bold_path = os.path.join(func_path, file_id + '_bold.nii.gz')
                    _write_fake_bold_img(bold_path, [vox, vox, vox, 100])
                    events_path = os.path.join(func_path, file_id +
                                               '_events.tsv')
                    _basic_paradigm().to_csv(events_path, sep='\t', index=None)
                    param_path = os.path.join(func_path, file_id +
                                              '_bold.json')
                    with open(param_path, 'w') as param_file:
                        json.dump({'RepetitionTime': 1.5}, param_file)

    # Create derivatives files
    if with_derivatives:
        bids_path = os.path.join(base_dir, 'bids_dataset', 'derivatives')
        os.makedirs(bids_path)
        for subject in ['sub-%02d' % label for label in range(1, 11)]:
            for session in created_sessions:
                subses_dir = os.path.join(bids_path, subject, session)
                func_path = os.path.join(subses_dir, 'func')
                os.makedirs(func_path)
                for task, n_run in zip(tasks, n_runs):
                    for run in ['run-%02d' % label for label in range(1, n_run + 1)]:
                        fields = [subject, session, 'task-' + task]
                        if '' in fields:
                            fields.remove('')
                        file_id = '_'.join(fields)
                        if n_run > 1:
                            file_id += '_' + run
                        preproc = file_id + '_space-MNI_desc-preproc_bold.nii.gz'
                        preproc_path = os.path.join(func_path, preproc)
                        _write_fake_bold_img(preproc_path, [vox, vox, vox, 100])
                        preproc = file_id + '_space-T1w_desc-preproc_bold.nii.gz'
                        preproc_path = os.path.join(func_path, preproc)
                        _write_fake_bold_img(preproc_path, [vox, vox, vox, 100])
                        preproc = file_id + '_space-T1w_desc-fmriprep_bold.nii.gz'
                        preproc_path = os.path.join(func_path, preproc)
                        _write_fake_bold_img(preproc_path, [vox, vox, vox, 100])
                        if with_confounds:
                            confounds_path = os.path.join(func_path, file_id +
                                                          '_desc-confounds_regressors.tsv')
                            _basic_confounds(100).to_csv(confounds_path,
                                                         sep='\t', index=None)
    return 'bids_dataset'
