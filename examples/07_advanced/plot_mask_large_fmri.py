"""
Working with long time series fMRI images
=========================================

In this example, we will demonstrate how one can work with large fMRI images
more efficiently. Note that fMRI images can be large on-disk due to several
different factors, including a long acquisition or high-resolution sampling.
Currently, this example focuses on memory-efficient interactions with long
time series fMRI data. In this case, loading the whole time series into memory
may represent a significant computational cost. We will therefore explore
strategies to minimize the amount of data that is loaded into memory,
and we will compare these strategies against a naive usage of
:class:`~nilearn.maskers.NiftiMasker`.

To make this more concrete, we will create a large fMRI image with over
800 time points by concatenating individual subjects in the
:func:`~nilearn.datasets.fetch_development_fmri` dataset.
Our goal is to extract data from several regions of interest (ROIs)
defined by a number of binary masks, all in parallel.

When using :class:`~nilearn.maskers.NiftiMasker` to extract data from each
ROI in parallel, each parallel process will load the entire fMRI image into
memory. This can lead to a significant increase in memory usage and may be
infeasible in some computational environments.

We will compare three different methods to mask the data from the fMRI image:

1. A naive, unoptimized usage of :class:`~nilearn.maskers.NiftiMasker`
2. Using array proxies defined with :mod:`nibabel`
3. Using :class:`multiprocessing.shared_memory.SharedMemory`

"""

# %%
# Create a large fMRI image
# -------------------------
# Here we will create a "large" fMRI image by fetch 5 subjects'
# fMRI images via the :func:`~nilearn.datasets.fetch_adhd`
# function, concatenating them and then saving to a file.

from pathlib import Path

from nilearn.datasets import fetch_adhd
from nilearn.image import concat_imgs

N_SUBJECTS = 6
N_REGIONS = 6

fmri_data = fetch_adhd(n_subjects=N_SUBJECTS)
fmri_img = concat_imgs(fmri_data.func)

output_dir = Path.cwd() / "results" / "plot_mask_large_fmri"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Large fmri file will be saved to:\n{output_dir}")

fmri_path = output_dir / "large_fmri.nii.gz"
fmri_img.to_filename(fmri_path)

# %%
# Create a set of binary masks
# ----------------------------
# We will now create 6 binary masks from a brain atlas. Here we will use the
# multiscale functional brain parcellations via the
# :func:`~nilearn.datasets.fetch_atlas_basc_multiscale_2015` function.
# We will fetch a 64-region version of this atlas and then create separate
# binary masks for the first 6 regions.

from nilearn.datasets import fetch_atlas_basc_multiscale_2015
from nilearn.image import load_img, new_img_like, resample_to_img

atlas_path = fetch_atlas_basc_multiscale_2015(resolution=64).maps

atlas_img = load_img(atlas_path)

# %%
# In this case, the atlas and the fMRI image have different resolutions.
# So we will resample the atlas to the fMRI image. It is important to do that
# because only :class:`~nilearn.maskers.NiftiMasker` can handle the resampling
# of the mask to the fMRI image but other methods considered here will not.

resampled_atlas = resample_to_img(
    atlas_img,
    fmri_img,
    interpolation="nearest",
    copy_header=True,
    force_resample=True,
)

mask_imgs = []
mask_paths = []
for idx in range(N_REGIONS):
    mask = resampled_atlas.get_fdata() == idx
    mask = new_img_like(
        ref_niimg=fmri_img,
        data=mask,
        affine=resampled_atlas.affine,
        copy_header=True,
    )

    mask_imgs.append(mask)
    path = output_dir / f"mask_{idx}.nii.gz"
    mask.to_filename(path)
    mask_paths.append(path)

# %%
# Mask the fMRI image using NiftiMasker
# -------------------------------------
# Let's first see how we would typically mask the fMRI image using
# :class:`~nilearn.maskers.NiftiMasker`. This is the most common way to extract
# data from an fMRI image as it makes it easy to standardize, smooth, detrend,
# etc. the data.
#
# We will define a function that will do that so that it's easier to use it
# with the ``memory_profiler`` package to measure the memory usage.

from nilearn.maskers import NiftiMasker


def nifti_masker_single(fmri_path, mask_path):
    return NiftiMasker(mask_img=mask_path).fit_transform(fmri_path)


# %%
# Furthermore, we can input the fmri image and the masks in two different ways:
#
# 1. Using the file paths
# 2. Using the in-memory objects
#
# So we will measure the memory usage for both cases.
#
# Let's first create a dictionary to store the memory usage for each method

from memory_profiler import memory_usage

nifti_masker = {"single": {"path": [], "in_memory": []}}

nifti_masker["single"]["path"] = memory_usage(
    (nifti_masker_single, (fmri_path, mask_paths[0])),
    max_usage=True,
)
print(
    "Peak memory usage: with NiftiMasker, single mask, with paths:\n"
    f"{nifti_masker['single']['path']} MiB"
)

nifti_masker["single"]["in_memory"] = memory_usage(
    (nifti_masker_single, (fmri_img, mask_imgs[0])),
    max_usage=True,
)
print(
    "Peak memory usage: with NiftiMasker, single mask, with in-memory image:\n"
    f"{nifti_masker['single']['in_memory']} MiB"
)

# %%
# Masking using NiftiMasker in parallel
# -------------------------------------
# Now let's see how we would mask the fMRI image using multiple masks in
# parallel using the :mod:`joblib` package.
#
# Let's add another key to the previous dictionary to store the memory usage
# for this case.

from joblib import Parallel, delayed


def nifti_masker_parallel(fmri_path, mask_paths):
    return Parallel(n_jobs=N_REGIONS)(
        delayed(nifti_masker_single)(fmri_path, mask) for mask in mask_paths
    )


nifti_masker["parallel"] = {"path": [], "in_memory": []}

nifti_masker["parallel"]["path"] = memory_usage(
    (nifti_masker_parallel, (fmri_path, mask_paths)),
    max_usage=True,
    include_children=True,
    multiprocess=True,
)
print(
    f"Peak memory usage: with NiftiMasker, {N_REGIONS} jobs in parallel, "
    "with paths:\n"
    f"{nifti_masker['parallel']['path']} MiB"
)

nifti_masker["parallel"]["in_memory"] = memory_usage(
    (nifti_masker_parallel, (fmri_img, mask_imgs)),
    max_usage=True,
    include_children=True,
    multiprocess=True,
)
print(
    f"Peak memory usage: with NiftiMasker, {N_REGIONS} jobs in parallel, "
    "with in-memory images:\n"
    f"{nifti_masker['parallel']['in_memory']} MiB"
)


# %%
# Masking using numpy indexing
# ----------------------------
# Now let's see how we can mask the fMRI image using numpy indexing. This
# could be more efficient than the NiftiMasker when we simply need to mask
# the fMRI image with binary masks and don't need to standardize, smooth, etc.
# the image.
#
# In addition, we will use :func:`nibabel.loadsave.load` function to load the
# fMRI image as a proxy object. This will allow us to load the data directly
# from the file as a numpy array without loading the entire image into memory.
# You can find more information about this in the :mod:`nibabel` documentation,
# here: https://nipy.org/nibabel/images_and_memory.html
#
# As before we will do this by loading the data from the file paths and from
# the in-memory objects. This time we will define two functions to do that.

import nibabel as nib
import numpy as np


def numpy_masker_single_path(fmri_path, mask_path):
    return np.asarray(nib.load(fmri_path).dataobj)[
        np.asarray(nib.load(mask_path).dataobj).astype(bool)
    ]


numpy_masker = {"single": {"path": [], "in_memory": []}}

numpy_masker["single"]["path"] = memory_usage(
    (numpy_masker_single_path, (fmri_path, mask_paths[0])),
    max_usage=True,
)
print(
    "Peak memory usage: with numpy indexing, single mask, with path:\n"
    f"{numpy_masker['single']['path']} MiB"
)


def numpy_masker_single_inmemory(fmri_img, mask_img):
    return np.asarray(fmri_img.dataobj)[
        np.asarray(mask_img.dataobj).astype(bool)
    ]


numpy_masker["single"]["in_memory"] = memory_usage(
    (numpy_masker_single_inmemory, (fmri_img, mask_imgs[0])),
    max_usage=True,
)
print(
    "Peak memory usage: with numpy indexing, single mask, "
    "with in-memory image:\n"
    f"{numpy_masker['single']['in_memory']} MiB"
)

# %%
# Masking using numpy indexing in parallel
# ----------------------------------------
# Now let's see how we can mask the fMRI image using multiple masks in parallel
# using numpy indexing.


def numpy_masker_parallel_path(fmri_path, mask_paths):
    return Parallel(n_jobs=N_REGIONS)(
        delayed(numpy_masker_single_path)(fmri_path, mask)
        for mask in mask_paths
    )


numpy_masker["parallel"] = {"path": [], "in_memory": []}

numpy_masker["parallel"]["path"] = memory_usage(
    (numpy_masker_parallel_path, (fmri_path, mask_paths)),
    max_usage=True,
    include_children=True,
    multiprocess=True,
)
print(
    f"Peak memory usage: with numpy indexing, {N_REGIONS} jobs in parallel, "
    "with path:\n"
    f"{numpy_masker['parallel']['path']} MiB"
)


def numpy_masker_parallel_inmemory(fmri_img, mask_imgs):
    return Parallel(n_jobs=N_REGIONS)(
        delayed(numpy_masker_single_inmemory)(fmri_img, mask)
        for mask in mask_imgs
    )


numpy_masker["parallel"]["in_memory"] = memory_usage(
    (numpy_masker_parallel_inmemory, (fmri_img, mask_imgs)),
    max_usage=True,
    include_children=True,
    multiprocess=True,
)
print(
    f"Peak memory usage: with numpy indexing, {N_REGIONS} jobs in parallel, "
    "with in-memory images:\n"
    f"{numpy_masker['parallel']['in_memory']} MiB"
)

# %%
# Masking using numpy indexing in parallel with shared memory
# -----------------------------------------------------------
# Finally, let's see how we can mask the fMRI image using multiple masks in
# parallel using numpy indexing and shared memory from the
# :mod:`multiprocessing` module.
#
# For this method, we would have to load the fMRI image into shared memory
# that can be accessed by multiple processes. This way, each process can
# access the data directly from the shared memory without loading the entire
# image into memory again.

from multiprocessing.shared_memory import SharedMemory

fmri_array = np.asarray(fmri_img.dataobj)
shm = SharedMemory(create=True, size=fmri_array.nbytes)
shared_array = np.ndarray(
    fmri_array.shape, dtype=fmri_array.dtype, buffer=shm.buf
)
np.copyto(shared_array, fmri_array)
del fmri_array

# %%
# Here, the image is already in-memory, so there is no need to examine the
# the two cases as we did before.


def numpy_masker_shared_single(img, mask):
    return img[np.asarray(mask.dataobj).astype(bool)]


def numpy_masker_shared_parallel(img, masks):
    return Parallel(n_jobs=N_REGIONS)(
        delayed(numpy_masker_shared_single)(img, mask) for mask in masks
    )


numpy_masker_shared = {"single": [], "parallel": []}

numpy_masker_shared["single"] = memory_usage(
    (numpy_masker_shared_single, (shared_array, mask_imgs[0])), max_usage=True
)
print(
    f"Peak memory usage: with numpy indexing, shared memory, "
    "and single mask:\n"
    f"{numpy_masker_shared['single']} MiB"
)


numpy_masker_shared["parallel"] = memory_usage(
    (numpy_masker_shared_parallel, (shared_array, mask_imgs)),
    max_usage=True,
    include_children=True,
    multiprocess=True,
)
print(
    f"Peak memory usage: with numpy indexing, shared memory, "
    f"{N_REGIONS} jobs in parallel:\n"
    f"{numpy_masker_shared['parallel']} MiB"
)

# cleanup
shm.close()
shm.unlink()

# %%
# Let's plot the memory usage for each method to compare them.
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(
    [
        "NiftiMasker,\nwith path",
        "NiftiMasker,\nwith in-memory\nimage",
        "Numpy indexing,\nwith path",
        "Numpy indexing,\nwith in-memory\nimage",
        "Numpy indexing,\nwith shared\nmemory",
    ],
    [
        nifti_masker["parallel"]["path"],
        nifti_masker["parallel"]["in_memory"],
        numpy_masker["parallel"]["path"],
        numpy_masker["parallel"]["in_memory"],
        numpy_masker_shared["parallel"],
    ],
    color=[
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
        (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
        (1.0, 0.4980392156862745, 0.054901960784313725),
        (1.0, 0.7333333333333333, 0.47058823529411764),
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    ],
)
plt.ylabel("Peak memory usage (MiB)")
plt.title(f"Memory usage comparison with {N_REGIONS} jobs in parallel")
plt.show()

# %%
# Conclusion
# ----------
# So using numpy indexing with shared memory is the most efficient way to mask
# large fMRI images in parallel. However, it is important to note that this
# method is only useful when we only need to mask the fMRI image with binary
# masks and don't need to standardize, smooth, etc. the image. Otherwise,
# using :class:`~nilearn.maskers.NiftiMasker` is still the most appropriate
# way to extract data from an fMRI image.
#
# Furthermore, the differences in memory usage between the methods can be more
# significant when working with much larger images and/or more jobs/regions
# in parallel. You can try increasing the ``N_SUBJECTS`` and ``N_REGIONS``
# variables at the beginning of this script to see how the memory usage changes
# for each method.
