"""
Experiment to compare image loading methods
===========================================
"""

# %%
# Define utility functions
# ------------------------

import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from memory_profiler import memory_usage

from nilearn.datasets import fetch_adhd, fetch_atlas_basc_multiscale_2015
from nilearn.image import (
    concat_imgs,
    load_img,
    new_img_like,
    resample_to_img,
)
from nilearn.maskers import NiftiMasker


def get_fmri_path(n_subjects=1):
    fmri_data = fetch_adhd(n_subjects=n_subjects)
    concat = concat_imgs(fmri_data.func)
    output_dir = Path.cwd() / "results" / "plot_compare_img_loading"
    output_dir.mkdir(parents=True, exist_ok=True)
    fmri_path = Path(output_dir, "fmri.nii.gz")
    concat.to_filename(fmri_path)
    return concat, fmri_path


def get_atlas_path():
    atlas_path = fetch_atlas_basc_multiscale_2015(resolution=64).maps
    return atlas_path


def atlas_to_masks(atlas_path, fmri_path, n_regions=6):
    atlas_img = load_img(atlas_path)
    resampled_atlas = resample_to_img(
        atlas_img,
        fmri_path,
        interpolation="nearest",
        copy_header=True,
        force_resample=True,
    )
    output_dir = Path.cwd() / "results" / "plot_compare_img_loading"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_imgs = []
    mask_paths = []
    for idx in range(1, n_regions + 1):
        mask = resampled_atlas.get_fdata() == idx
        mask = new_img_like(
            ref_niimg=fmri_path,
            data=mask,
            affine=resampled_atlas.affine,
            copy_header=True,
        )
        mask_imgs.append(mask)
        path = output_dir / f"mask_{idx}.nii.gz"
        mask.to_filename(path)
        mask_paths.append(path)

    return mask_imgs, mask_paths


def nifti_masker_single(fmri_path, mask_path):
    return NiftiMasker(mask_img=mask_path).fit_transform(fmri_path)


def numpy_masker_single_path(fmri_path, mask_path):
    return np.asarray(nib.load(fmri_path).dataobj)[
        np.asarray(nib.load(mask_path).dataobj).astype(bool)
    ]


def numpy_masker_single_inmemory(fmri_img, mask_img):
    return np.asarray(fmri_img.dataobj)[
        np.asarray(mask_img.dataobj).astype(bool)
    ]


def numpy_masker_shared_single(img, mask):
    return img[np.asarray(mask.dataobj).astype(bool)]


def nifti_masker_parallel(fmri_path, mask_paths, n_regions=6):
    return Parallel(n_jobs=n_regions)(
        delayed(nifti_masker_single)(fmri_path, mask) for mask in mask_paths
    )


def numpy_masker_parallel_path(fmri_path, mask_paths, n_regions=6):
    return Parallel(n_jobs=n_regions)(
        delayed(numpy_masker_single_path)(fmri_path, mask)
        for mask in mask_paths
    )


def numpy_masker_parallel_inmemory(fmri_img, mask_imgs, n_regions=6):
    return Parallel(n_jobs=n_regions)(
        delayed(numpy_masker_single_inmemory)(fmri_img, mask)
        for mask in mask_imgs
    )


def numpy_masker_shared_parallel(img, masks, n_regions=6):
    return Parallel(n_jobs=n_regions)(
        delayed(numpy_masker_shared_single)(img, mask) for mask in masks
    )


def plot_memory_usage(
    fig,
    ax,
    method,
    usage,
    peak_usage,
):
    # get zero time
    zero_time = usage[0][1]
    # subtract zero time from all timestamps and convert to dict
    usage = {time - zero_time: mem for mem, time in usage}

    # plot memory usage over time
    (line,) = ax.plot(usage.keys(), usage.values(), label=method)
    line_color = line.get_color()

    order = {
        "load_img": 1,
        "concat_imgs": 2,
        "nibabel_load": 3,
    }

    # use order of max usage and time to calculate offset for annotations
    xoffset = np.array(list(usage.keys())).max() * 0.001
    yoffset = np.array(list(usage.values())).max() * 0.01 * order[method]

    # add annotations on each peak
    for peak in peak_usage:
        # get only the memory usage
        mems = np.array([mem for mem, _ in peak_usage[peak]])
        peak_time = peak_usage[peak][mems.argmax()][1] - zero_time
        peak_mem = mems.max()
        ax.annotate(
            f"{peak_mem:.2f} MiB",
            xy=(peak_time, peak_mem),
            xytext=(
                peak_time - xoffset,
                peak_mem + yoffset,
            ),
            color=line_color,
        )
        # only annotate with the masking method for the last peak
        if order[method] == 3:
            yoffset *= 1.5
            ax.annotate(
                peak,
                xy=(peak_time, peak_mem),
                xytext=(peak_time - xoffset, peak_mem + yoffset),
            )

    # increase the y-axis limit by 20% to make the plot more readable
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.2)

    return fig, ax


def plot_scatter_memvcomputation_time(
    fig,
    ax,
    method,
    usage,
    peak_usage,
    color,
):
    # get zero time
    zero_time = usage[0][1]

    peak_mems = []
    computation_times = []
    names = []

    for peak in peak_usage:
        # get max memory usage
        peak_mem = np.array([mem for mem, _ in peak_usage[peak]]).max()
        # get total computation time
        start_time = peak_usage[peak][0][1] - zero_time
        end_time = peak_usage[peak][-1][1] - zero_time
        computation_time = end_time - start_time
        computation_times.append(computation_time)
        peak_mems.append(peak_mem)
        names.append(peak)

    plt.scatter(
        computation_times,
        peak_mems,
        label=method,
        c=color,
    )

    ann_colors = {
        "nifti_masker": "tab:pink",
        "numpy_masker": "tab:olive",
        "numpy_masker_shared": "tab:cyan",
    }

    # add annotations indicating the method
    for peak_mem, computation_time, name in zip(
        peak_mems, computation_times, names
    ):
        ax.annotate(
            name,
            xy=(computation_time, peak_mem),
            xytext=(computation_time + 0.01, peak_mem),
            color=ann_colors[name],
        )

    return fig, ax


# %%
# Main function that runs each method one by one
# ----------------------------------------------
def main(n_images=1, n_regions=6, wait_time=30, load="concat_imgs"):
    """
    Compare the performance of NiftiMasker vs. numpy masking vs.
    numpy masking + shared memory both with single and
    `n_regions` parallel processes.

    The first two methods can be used with either file paths
    or in-memory images. So we also compare their memory usage.

    We add `wait_time` between each method to see the memory usage
    of each method separately in the plot.

    Steps:

    1. fetch `n_images` subjects from development fMRI dataset and
    `n_regions` regions from the Difumo atlas.
    2. convert these regions to binary masks and resample them to the
    fMRI data.
    3. run the following methods in sequence:
        - NiftiMasker with parallel nifti file paths
        - NiftiMasker with parallel in-memory nifti images
        - numpy masking with parallel nifti file paths
        - numpy masking with parallel in-memory nifti images
        - numpy masking with nifti image in-memory shared by parallel
        processes


    Parameters
    ----------
    n_images : int, default=1
        Number of subjects to fetch from the development fMRI dataset. These
        subject images would be concatenated to form a single nifti file.
        Can be increased to simulate larger data.

    n_regions : int, default=6
        Number of regions to fetch from the Difumo atlas. These regions would
        be converted to binary masks and used to mask the fMRI data. This is
        also the number of jobs to run in parallel.

    wait_time : int, default=30
        Time to wait between each method to see the memory usage of each
        method separately in the plot.
    """
    fmri_img, fmri_path = get_fmri_path(n_subjects=n_images)
    if load == "load_img":
        del fmri_img
        fmri_img = load_img(fmri_path)
    elif load == "nibabel_load":
        del fmri_img
        fmri_img = nib.load(fmri_path)
    atlas_path = get_atlas_path()
    mask_imgs, mask_paths = atlas_to_masks(
        atlas_path, fmri_path, n_regions=n_regions
    )

    peak_usage = {
        "nifti_masker": [],
        "numpy_masker": [],
        "numpy_masker_shared": [],
    }

    time.sleep(wait_time)

    print(f"\nLoading via {load}\n")

    peak_usage["nifti_masker"] = memory_usage(
        (nifti_masker_parallel, (fmri_img, mask_imgs, n_regions)),
        timestamps=True,
        include_children=True,
        multiprocess=True,
    )
    print(f"{peak_usage['nifti_masker']=}\n")

    time.sleep(wait_time)

    peak_usage["numpy_masker"] = memory_usage(
        (
            numpy_masker_parallel_inmemory,
            (fmri_img, mask_imgs, n_regions),
        ),
        timestamps=True,
        include_children=True,
        multiprocess=True,
    )
    print(f"{peak_usage['numpy_masker']=}\n")

    time.sleep(wait_time)

    fmri_data = np.asarray(fmri_img.dataobj)
    shm = SharedMemory(create=True, size=fmri_data.nbytes)
    shared_data = np.ndarray(
        fmri_data.shape, dtype=fmri_data.dtype, buffer=shm.buf
    )
    np.copyto(shared_data, fmri_data)
    del fmri_data

    time.sleep(wait_time)

    peak_usage["numpy_masker_shared"] = memory_usage(
        (
            numpy_masker_shared_parallel,
            (shared_data, mask_imgs, n_regions),
        ),
        timestamps=True,
        include_children=True,
        multiprocess=True,
    )
    print(f"{peak_usage['numpy_masker_shared']=}\n")

    shm.close()
    shm.unlink()

    return peak_usage


# %%
# Run the main function and plot memory usage
# -------------------------------------------
if __name__ == "__main__":
    N_SUBJECTS = 6
    N_REGIONS = 6
    WAIT_TIME = 30

    usages = []
    peak_usages = []

    for loading_method in ["load_img", "concat_imgs", "nibabel_load"]:
        usage, peak_usage = memory_usage(
            (
                main,
                (N_SUBJECTS, N_REGIONS, WAIT_TIME, loading_method),
            ),
            include_children=True,
            multiprocess=True,
            timestamps=True,
            retval=True,
        )
        usages.append(usage)
        peak_usages.append(peak_usage)

    plot_path = Path.cwd() / "results" / "plot_compare_img_loading"
    plot_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # plot memory usage vs. computation time
    for usage, peak_usage, loading_method, color in zip(
        usages,
        peak_usages,
        [
            "load_img",
            "concat_imgs",
            "nibabel_load",
        ],
        [
            "tab:blue",
            "tab:orange",
            "tab:green",
        ],
    ):
        fig, ax = plot_scatter_memvcomputation_time(
            fig,
            ax,
            loading_method,
            usage,
            peak_usage,
            color,
        )

    ax.set_xlabel("Computation time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title(
        f"Memory usage vs. computation time with N_SUBJECTS={N_SUBJECTS},"
        f" N_REGIONS={N_REGIONS}"
    )
    ax.legend()
    plt.savefig(
        plot_path / f"memvcomputation_time_n{N_SUBJECTS}_j{N_REGIONS}.png"
    )
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    # plot memory usage vs. computation time
    for usage, peak_usage, loading_method in zip(
        usages,
        peak_usages,
        [
            "load_img",
            "concat_imgs",
            "nibabel_load",
        ],
    ):
        # plot memory usage over time
        fig, ax = plot_memory_usage(
            fig,
            ax,
            loading_method,
            usage,
            peak_usage,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title(
        f"Memory usage over time with N_SUBJECTS={N_SUBJECTS},"
        f" N_REGIONS={N_REGIONS}"
    )
    ax.legend()
    plt.savefig(plot_path / f"memory_usage_n{N_SUBJECTS}_j{N_REGIONS}.png")
    plt.show()
