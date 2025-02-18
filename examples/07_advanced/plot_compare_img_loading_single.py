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


def nifti_masker_concat_img(img, mask_img):
    t_load = time.time()
    img = concat_imgs(img)
    mask_img = concat_imgs(mask_img)
    t0_mask = time.time()
    NiftiMasker(mask_img=mask_img).fit_transform(img)
    tend_mask = time.time()
    return t_load, t0_mask, tend_mask


def nifti_masker_load_img(img, mask_img):
    t_load = time.time()
    img = load_img(img)
    mask_img = load_img(mask_img)
    t0_mask = time.time()
    NiftiMasker(mask_img=mask_img).fit_transform(img)
    tend_mask = time.time()
    return t_load, t0_mask, tend_mask


def nifti_masker_nibabel_load(img, mask_img):
    t_load = time.time()
    img = load(img)
    mask_img = load(mask_img)
    t0_mask = time.time()
    NiftiMasker(mask_img=mask_img).fit_transform(img)
    tend_mask = time.time()
    return t_load, t0_mask, tend_mask


def numpy_masker_concat_img(fmri_path, mask_path):
    t_load = time.time()
    img = np.asarray(concat_imgs(fmri_path).dataobj)
    mask_img = np.squeeze(
        np.asarray(concat_imgs(mask_path).dataobj).astype(bool)
    )
    t0_mask = time.time()
    img[mask_img]
    tend_mask = time.time()
    return t_load, t0_mask, tend_mask


def numpy_masker_load_img(fmri_path, mask_path):
    t_load = time.time()
    img = np.asarray(load_img(fmri_path).dataobj)
    mask_img = np.asarray(load_img(mask_path).dataobj).astype(bool)
    t0_mask = time.time()
    img[mask_img]
    tend_mask = time.time()
    return t_load, t0_mask, tend_mask


def numpy_masker_nibabel_load(fmri_path, mask_path):
    t_load = time.time()
    img = np.asarray(load(fmri_path).dataobj)
    mask_img = np.asarray(load(mask_path).dataobj).astype(bool)
    t0_mask = time.time()
    img[mask_img]
    tend_mask = time.time()
    return t_load, t0_mask, tend_mask


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

    usages = {}
    call_times = {}

    funcs = [
        nifti_masker_concat_img,
        nifti_masker_load_img,
        nifti_masker_nibabel_load,
        numpy_masker_concat_img,
        numpy_masker_load_img,
        numpy_masker_nibabel_load,
    ]

    # run all the nifti_masker functions first
    for func in funcs[:3]:
        print(f"Running {func.__name__}")
        mem_usage, call_time = memory_usage(
            (func, (img_path, mask_path)), timestamps=True, retval=True
        )
        usages[func.__name__] = mem_usage
        call_times[func.__name__] = call_time

    plot_path = Path.cwd() / "results" / "plot_compare_img_loading_single"
    plot_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 6))
    fig, ax = plot_memory_usage(fig, ax, usages, call_times)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title(f"Memory usage over time with N_SUBJECTS={N_SUBJECTS}")
    ax.legend()
    plt.savefig(
        plot_path / f"nifti_memory_usage_n{N_SUBJECTS}.png",
        bbox_inches="tight",
    )
    plt.show()

    usages = {}
    call_times = {}

    # run all the numpy masker functions
    for func in funcs[3:]:
        print(f"Running {func.__name__}")
        mem_usage, call_time = memory_usage(
            (func, (img_path, mask_path)), timestamps=True, retval=True
        )
        usages[func.__name__] = mem_usage
        call_times[func.__name__] = call_time
    fig, ax = plt.subplots(figsize=(15, 6))
    fig, ax = plot_memory_usage(fig, ax, usages, call_times)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title(f"Memory usage over time with N_SUBJECTS={N_SUBJECTS}")
    ax.legend()
    plt.savefig(
        plot_path / f"numpy_memory_usage_n{N_SUBJECTS}.png",
        bbox_inches="tight",
    )
    plt.show()
