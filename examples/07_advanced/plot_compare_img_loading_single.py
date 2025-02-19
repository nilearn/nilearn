"""
Experiment to compare image loading methods
===========================================
"""

# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import memory_usage
from nibabel import load

from nilearn.datasets import fetch_adhd, fetch_atlas_basc_multiscale_2015
from nilearn.image import (
    concat_imgs,
    get_data,
    load_img,
    new_img_like,
    resample_to_img,
)
from nilearn.maskers import NiftiMasker


def get_fmri_path(n_subjects=1):
    output_dir = Path.cwd() / "results" / "plot_compare_img_loading_single"
    output_dir.mkdir(parents=True, exist_ok=True)
    fmri_path = Path(output_dir, "fmri.nii.gz")
    fmri_data = fetch_adhd(n_subjects=n_subjects)
    concat = concat_imgs(fmri_data.func)
    concat.to_filename(fmri_path)
    return fmri_path


def get_mask_path(fmri_path):
    output_dir = Path.cwd() / "results" / "plot_compare_img_loading_single"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "mask.nii.gz"
    atlas_path = fetch_atlas_basc_multiscale_2015(resolution=7).maps
    atlas_img = load_img(atlas_path)
    resampled_atlas = resample_to_img(
        atlas_img,
        fmri_path,
        interpolation="nearest",
        copy_header=True,
        force_resample=True,
    )
    mask = resampled_atlas.get_fdata() == 1
    mask = new_img_like(
        ref_niimg=fmri_path,
        data=mask,
        affine=resampled_atlas.affine,
        copy_header=True,
    )
    mask.to_filename(mask_path)
    return mask_path


def image_loader(img_path, method="proxy_image"):
    if method == "proxy_image":
        return load_img(img_path)
    elif method == "array_image":
        return new_img_like(img_path, get_data(img_path), copy_header=True)


def nifti_masker_single(
    img_path,
    mask_path,
    method="proxy_image",
):
    t_load = time.time()
    img = image_loader(img_path, method=method)
    mask = image_loader(mask_path, method=method)
    t0_mask = time.time()
    NiftiMasker(mask_img=mask).fit_transform(img)
    return t_load, t0_mask, time.time()


def numpy_masker_single(img_path, mask_path):
    t_load = time.time()
    img = np.asarray(image_loader(img_path, method=method).dataobj)
    mask = np.asarray(image_loader(mask_path, method=method).dataobj).astype(
        bool
    )
    t0_mask = time.time()
    img[mask]
    return t_load, t0_mask, time.time()


def plot_memory_usage(fig, ax, usages, call_times):
    for usage in usages:
        if not usages[usage]:
            continue
        # get zero time
        zero_time = usages[usage][0][1]

        # get only the memory usage
        mems = np.array([mem for mem, _ in usages[usage]])
        times = np.array([time for _, time in usages[usage]]) - zero_time
        peak_time = times[mems.argmax()]
        peak_mem = mems.max()

        # plot memory usage over time
        (line,) = ax.plot(times, mems, label=usage)
        line_color = line.get_color()

        # use order of max usage and time to calculate offset for annotations
        xoffset = peak_time * 0.001
        yoffset = peak_mem * 0.01

        ax.annotate(
            f"{peak_mem:.2f} MiB",
            xy=(peak_time, peak_mem),
            xytext=(
                peak_time - xoffset,
                peak_mem + yoffset,
            ),
            color=line_color,
        )

        # plot call times
        for call_idx, call_time in enumerate(call_times[usage]):
            ax.axvline(call_time - zero_time, color=line_color, linestyle="--")
            if call_idx == 0:
                annotation = "load"
            if call_idx == 1:
                annotation = "start masking"
            if call_idx == 2:
                annotation = "end masking"
            ax.annotate(
                annotation,
                xy=(call_time - zero_time, 0),
                xytext=(call_time - zero_time, peak_mem + (peak_mem * 0.1)),
                color=line_color,
            )

    # increase the y-axis limit by 20% to make the plot more readable
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.2)

    return fig, ax


# %%
if __name__ == "__main__":
    N_SUBJECTS = 20
    img_path = get_fmri_path(N_SUBJECTS)
    mask_path = get_mask_path(img_path)
    plot_path = Path.cwd() / "results" / "plot_compare_img_loading_single"
    plot_path.mkdir(parents=True, exist_ok=True)

    n_timepoints = load(img_path).shape[-1]

    usages = {}
    call_times = {}

    loading_methods = ["proxy_image", "array_image"]

    for method in loading_methods:
        print(f"Running {nifti_masker_single.__name__}, {method=}")
        usage, call_time = memory_usage(
            (nifti_masker_single, (img_path, mask_path)),
            timestamps=True,
            retval=True,
        )
        usages[f"nifti_{method}"] = usage
        call_times[f"nifti_{method}"] = call_time

    fig, ax = plt.subplots(figsize=(15, 6))
    fig, ax = plot_memory_usage(fig, ax, usages, call_times)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title(f"Memory usage for an image with {n_timepoints=}")
    ax.legend()
    plt.savefig(
        plot_path / f"nifti_memory_usage_t{n_timepoints}.png",
        bbox_inches="tight",
    )
    plt.show()

    usages = {}
    call_times = {}

    for method in loading_methods:
        print(f"Running {numpy_masker_single.__name__}, {method=}")
        usage, call_time = memory_usage(
            (numpy_masker_single, (img_path, mask_path)),
            timestamps=True,
            retval=True,
        )
        usages[f"numpy_{method}"] = usage
        call_times[f"numpy_{method}"] = call_time

    fig, ax = plt.subplots(figsize=(15, 6))
    fig, ax = plot_memory_usage(fig, ax, usages, call_times)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title(f"Memory usage for an image with {n_timepoints=}")
    ax.legend()
    plt.savefig(
        plot_path / f"numpy_memory_usage_t{n_timepoints}.png",
        bbox_inches="tight",
    )
    plt.show()
