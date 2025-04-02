"""
Factors affecting performance of neuroimaging workflows
=======================================================

In this example we will discuss the factors that can affect the performance of
neuroimaging workflows and the steps that can be taken to optimize them.

The performance of a workflow can be measured in terms of the time it takes to
complete and the amount of memory it uses. Both these aspects are heavily
dependent on the size of the data being processed and how that data is loaded
into memory.

Another important aspect is the operations being performed on the data. For
example, there can be situations where we either need all the data in
memory at once, or where we can process the data in chunks.

So here we will compare both  the time and memory usage of (1) different methods of
loading and then (2) operations where we need all the data in memory at once versus
where we can process the data in chunks.

Proxy images vs. array images
=============================

A nifti image can be loaded as a proxy image or an array image. This page on
Nibabel documentation does a good job of explaining the difference between the
two: https://nipy.org/nibabel/images_and_memory.html

But TLDR; a proxy image is an object that only points to the actual numpy
array data on disk. This means that the data is not loaded into memory until
it is accessed. On the other hand, an array image is an object that loads the
data into memory as soon as it is created.

Proxy images
============

If you are reading an image from the disk, you can do so via nibabel's
:func:`nibabel.loadsave.load` and nilearn's :func:`nilearn.image.load_img`
function. Both of these functions return a proxy image. The difference is
that with :func:`nibabel.loadsave.load` you only get the proxy image and you
have to call the ``.get_fdata()`` method to load the data into memory.
On the other hand, with :func:`~nilearn.image.load_img` you get a proxy image
that loads the data into memory as soon as it is created.

"""

# %%
# Create a large fMRI image
# -------------------------
# Here we will create a "large" fMRI image by fetching 10 subjects'
# fMRI images via the :func:`~nilearn.datasets.fetch_adhd`
# function, concatenating them and then saving to a file.

from pathlib import Path

from nilearn.datasets import fetch_adhd
from nilearn.image import concat_imgs

from nilearn.maskers import NiftiMasker

# increase N_SUBJECTS to increase the size of the image
N_SUBJECTS = 10

def get_fmri_path(n_subjects=1, output_dir=None):
    output_dir = Path.cwd() / "results" / "plot_performance_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    fmri_path = Path(output_dir, "fmri.nii.gz")
    if fmri_path.exists():
        return fmri_path
    fmri_data = fetch_adhd(n_subjects=n_subjects)
    concat = concat_imgs(fmri_data.func)
    concat.to_filename(fmri_path)
    return fmri_path

output_dir = Path.cwd() / "results" / "plot_performance_comparison"
example_fmri_path = get_fmri_path(n_subjects=N_SUBJECTS)

print(f"Large example fmri file will be saved to:\n{output_dir}")
# %%
# Time taken to load an image
# ---------------------------
# So we expect that when simply loading the image,
# :func:`nibabel.loadsave.load` would be faster and lower on memory usage
# (because it doesn't load the data into memory) than
# :func:`~nilearn.image.load_img`.

import nibabel as nib
from nilearn.image import load_img

# load image via nibabel.load
%time nib.load(example_fmri_path)
# CPU times: user 2.09 ms, sys: 1.01 ms, total: 3.1 ms
# Wall time: 2.93 ms


# load image via nilearn.image.load_img
%time load_img(example_fmri_path)
# CPU times: user 3.93 s, sys: 1.24 s, total: 5.17 s
# Wall time: 5.17 s

# %%
# Memory usage while loading an image
# -----------------------------------
# We can also measure the memory usage of each of these methods using the
# ``memory_profiler`` package. Once we have installed the package (via
# ``pip install memory_profiler``), we can use ``%memit`` magic command to
# measure the memory usage of a single line of code.

%load_ext memory_profiler

# load image via nibabel.load
%memit nib.load(example_fmri_path)
# peak memory: 2179.85 MiB, increment: 0.00 MiB

# load image via nilearn.image.load_img
%memit load_img(example_fmri_path)
# peak memory: 6113.84 MiB, increment: 3933.99 MiB

# %%
# Some use cases
# ==============
# Once we have loaded the image, we can perform various operations on it.
# We will consider two cases here:
# 1. Taking the mean over the time axis, which requires all the data to be
#    loaded into memory at once.
# 2. Extracting a 3D volume at a given time point from the 4D image, which
#    only requires a chunk of data to be loaded into memory.

# Mean over the time axis
# -----------------------
# To take the mean over the time axis, we can use
# :func:`nilearn.image.mean_img`. This function requires all the data to be
# loaded into memory at once.
#
# So when we load the image with :func:`~nilearn.image.load_img` and then pass
# it to :func:`~nilearn.image.mean_img` function, the data is readily
# available in memory and the function can operate quickly.

from nilearn.image import mean_img

img_nilearn = load_img(example_fmri_path)

# mean over image loaded via nilearn.image.load_img
%time mean_img(img_nilearn, copy_header=True)
# CPU times: user 142 ms, sys: 12.8 ms, total: 155 ms
# Wall time: 176 ms

# %%
# But when compared to loading the image with :func:`nibabel.loadsave.load`:

img_nibabel = nib.load(example_fmri_path)
# mean over image loaded via nibabel.load
%time mean_img(img_nibabel, copy_header=True)
# CPU times: user 4.11 s, sys: 1.22 s, total: 5.34 s
# Wall time: 5.34 s

# %%
# This takes more time because :func:`~nilearn.image.mean_img` will have to
# load the data before it can take the mean.
#
# But it is important to note that the overall time taken to first load the
# image and take the mean is similar for both the methods.
# This is simply because the data has to be loaded at some point either before
# or within :func:`~nilearn.image.mean_img`.
#
# We can verify that by timing the loading and mean calculation together:

%%time
img_nilearn = load_img(example_fmri_path)
mean_img(img_nilearn, copy_header=True)
# CPU times: user 4.1 s, sys: 1.28 s, total: 5.38 s
# Wall time: 5.38 s

# %%
# The memory usage of the two would also be similar for the same reason.

%%memit
img_nilearn = load_img(example_fmri_path)
mean_img(img_nilearn, copy_header=True)
# peak memory: 10059.32 MiB, increment: 3936.28 MiB

# %%
%%memit
img_nibabel = nib.load(example_fmri_path)
mean_img(img_nibabel, copy_header=True)
# peak memory: 8091.86 MiB, increment: 1967.71 MiB

#%%
# Extracting a 3D volume
# ----------------------
#
# Now let's say we want to extract a 3D volume at some time point from the
# 4D image. Here we only need that 3D volume to be loaded into memory.
#
# Proxy images come with an attribute called ``.dataobj`` that allows us to
# directly access the chunk of data we need.
#
# So with :func:`~nilearn.image.load_img`:

%%time
img_nilearn = load_img(example_fmri_path)
img_nilearn.dataobj[..., 3]
# CPU times: user 4.04 s, sys: 1.53 s, total: 5.57 s
# Wall time: 5.57 s

# %%
# And with :func:`nibabel.loadsave.load`:

%%time
img_nibabel = nib.load(example_fmri_path)
img_nibabel.dataobj[..., 3]
# CPU times: user 11.8 ms, sys: 9.19 ms, total: 21 ms
# Wall time: 20.2 ms

# %%
# What happens here with :func:`~nilearn.image.load_img` is that we load the
# entire image into memory even though we only need a chunk of it. This is why
# it takes more time than :func:`nibabel.loadsave.load` which only loads the
# chunk of data we need.
#
# We will see that with the memory usage as well:

%%memit
img_nilearn = load_img(example_fmri_path)
img_nilearn.dataobj[..., 3]
# peak memory: 8093.21 MiB, increment: 3936.11 MiB

# %%
%%memit
img_nibabel = nib.load(example_fmri_path)
img_nibabel.dataobj[..., 3]
# peak memory: 4158.06 MiB, increment: 0.00 MiB

# %%
# Array images
# ============
#
# In practice, you would initially only use proxy images when you load an image
# from the disk. But once you perform an operation that modifies the image,
# you would get an array image.
#
# For example, if you smooth an image using :func:`nilearn.image.smooth_img`
# function, it will return an array image. We can check this using nibabel's
# :func:`nibabel.arrayproxy.is_proxy` function on the image.

from nilearn.image import smooth_img

img_nilearn = load_img(example_fmri_path)
img_smoothed = smooth_img(img_nilearn, fwhm=6)
nib.is_proxy(img_smoothed.dataobj)
# False

# %%
# But :func:`nibabel.arrayproxy.is_proxy` would return ``True`` for
# ``img_nilearn.dataobj``:

nib.is_proxy(img_nilearn.dataobj)

# %%
# So if you are performing subsequent operations that only require a chunk of
# data in the memory, it could be beneficial to first save the image to disk
# and then loading it again via :func:`nibabel.loadsave.load` function to get a
# proxy image.

# However, if you anyway need all the data in memory, you can directly use
# the array image in subsequent operations.

# This applies to most of the operations under nilearn's :mod:`nilearn.image`
# module as they all return array images.

# Finally, another possible use case could be when you want to perform several
# operations on the same image in parallel.

# We examine such a case in detail in this example:
# :ref:`sphx_glr_auto_examples_07_advanced_plot_mask_large_fmri.py`.
