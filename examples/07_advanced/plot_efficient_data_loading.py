# ruff: noqa: D100

import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from memory_profiler import profile
from nibabel import load

from nilearn.datasets import fetch_atlas_difumo, fetch_development_fmri
from nilearn.image import (
    concat_imgs,
    index_img,
    iter_img,
    load_img,
    resample_to_img,
)
from nilearn.maskers import NiftiMasker


@profile
def get_fmri_path():
    fmri_data = fetch_development_fmri(n_subjects=1)
    concat = concat_imgs(fmri_data.func)
    fmri_path = Path("temp", "fmri.nii.gz")
    concat.to_filename(fmri_path)
    return fmri_path


@profile
def get_atlas_path():
    atlas = fetch_atlas_difumo(dimension=64)
    atlas_path = atlas.maps
    return atlas_path


@profile
def atlas_to_masks(atlas_path, fmri_path):
    masks = load_img(atlas_path)
    # only keep the first 6 regions
    masks = index_img(masks, slice(0, 6))
    mask_paths = []
    Path("temp").mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(iter_img(masks)):
        resampled_mask = resample_to_img(
            mask,
            fmri_path,
            interpolation="nearest",
            copy_header=True,
            force_resample=True,
        )
        path = Path("temp", f"mask_{i}.nii.gz")
        data = resampled_mask.get_fdata()
        data[data != 0] = 1
        resampled_mask = resampled_mask.__class__(
            data, resampled_mask.affine, resampled_mask.header
        )
        resampled_mask.to_filename(path)
        mask_paths.append(path)
    return mask_paths


@profile
def mask_fmri_single(fmri_path, mask_path):
    masker = NiftiMasker(mask_img=mask_path, standardize=True)
    return masker.fit_transform(fmri_path)


@profile
def mask_fmri_single_efficient(fmri_path, mask_path):
    return np.asarray(load(fmri_path).dataobj)[
        np.asarray(load(mask_path).dataobj).astype(bool)
    ]


@profile
def mask_fmri_parallel(fmri_path, mask_paths):
    n_jobs = 6
    fmri_ts = Parallel(n_jobs=n_jobs)(
        delayed(mask_fmri_single)(fmri_path, mask) for mask in mask_paths
    )
    return fmri_ts


@profile
def mask_fmri_efficient_parallel(fmri_path, mask_paths):
    n_jobs = 6
    fmri_ts = Parallel(n_jobs=n_jobs)(
        delayed(mask_fmri_single_efficient)(fmri_path, mask)
        for mask in mask_paths
    )
    return fmri_ts


@profile
def main():
    fmri_path = get_fmri_path()
    atlas_path = get_atlas_path()
    mask_paths = atlas_to_masks(atlas_path, fmri_path)

    print("waiting")
    time.sleep(30)
    print("start nifti masker")

    ts_nifti_masker = mask_fmri_parallel(fmri_path, mask_paths)
    del ts_nifti_masker

    print("waiting")
    time.sleep(30)
    print("start efficient masker")

    ts_efficient = mask_fmri_efficient_parallel(fmri_path, mask_paths)
    del ts_efficient


if __name__ == "__main__":
    main()
