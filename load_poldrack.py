import numpy as np
import nibabel
from scipy import ndimage
from nilearn.datasets import fetch_haxby
from nilearn.input_data import NiftiMasker
from sklearn.externals.joblib import Memory

poldrack_subjects = np.arange(1, 17)


def load_subject_poldrack(subject_id, smooth=0):
    betas_fname = ('data/Jimura_Poldrack_2012_zmaps/gain_realigned/'
                   'sub0%02d_zmaps.nii.gz' % subject_id)
    img = nibabel.load(betas_fname)
    X = img.get_data()
    affine = img.get_affine()
    finite_mask = np.all(np.isfinite(X), axis=-1)
    mask = np.logical_and(np.all(X != 0, axis=-1),
                          finite_mask)
    if smooth:
        for i in range(X.shape[-1]):
            X[..., i] = ndimage.gaussian_filter(X[..., i], smooth)
        X[np.logical_not(finite_mask)] = np.nan
    y = np.array([np.arange(1, 9)] * 6).ravel()

    assert len(y) == 48
    assert len(y) == X.shape[-1]
    return X, y, mask, affine


def load_gain_poldrack(smooth=0):
    X = []
    y = []
    subject = []
    mask = []
    for i in poldrack_subjects:
        X_, y_, this_mask, affine = load_subject_poldrack(i, smooth=smooth)
        X_ -= X_.mean(axis=-1)[..., np.newaxis]
        std = X_.std(axis=-1)
        std[std == 0] = 1
        X_ /= std[..., np.newaxis]
        X.append(X_)
        y.extend(y_)
        subject.extend(len(y_) * [i, ])
        mask.append(this_mask)
    X = np.concatenate(X, axis=-1)
    mask = np.sum(mask, axis=0) > .5 * len(poldrack_subjects)
    mask = np.logical_and(mask, np.all(np.isfinite(X), axis=-1))

    return X[mask, :].T, np.array(y), np.array(subject), mask, affine


def load_haxby(cond1="face", cond2="house", memory="cache",
               full_brain=False):
    data_files = fetch_haxby(n_subjects=1)
    epi = nibabel.load(data_files.func[0])

    # only keep back of the brain
    if not full_brain:
        data = epi.get_data()
        data[:, data.shape[1] / 2:, :] = 0
        epi = nibabel.Nifti1Image(data, epi.get_affine())

    anat = nibabel.load(data_files['anat'][0])
    labels = np.recfromcsv(data_files.session_target[0], delimiter=" ")
    stimuli = labels['labels']
    condition_mask = np.logical_or(stimuli == cond1, stimuli == cond2)
    masker = NiftiMasker(standardize=True, memory=memory,
                         mask_strategy='epi')
    masked_timecourses = masker.fit_transform(epi)[condition_mask]
    stimuli = stimuli[condition_mask]
    y = np.unique(stimuli, return_inverse=True)[1] * 2 - 1
    mask = masker.mask_img_.get_data().astype(np.bool)
    session_labels = labels['chunks'][condition_mask]

    return masked_timecourses, y, mask, masker, session_labels, anat


def load_poldrack(full_brain=True, memory=Memory("None")):
    X, y, _, mask, affine = memory.cache(load_gain_poldrack)(smooth=0)
    ymin, ymax = 32, 35

    if not full_brain:  # take a smaller dataset for debug
        nx, ny, nz = mask.shape
        data = np.zeros((len(X), nx, ny, nz), dtype=np.float32)
        data[:, mask] = X
        mask[:, :ymin] = False
        mask[:, ymax:] = False
        X = data[:, mask]
        del data

    return X, y, mask, affine
