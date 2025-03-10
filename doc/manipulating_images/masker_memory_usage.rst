.. _masker_memory_usage:

================================================
Optimizing performance of neuroimaging workflows
================================================

In this chapter we will discuss the factors that can affect the performance of
neuroimaging workflows and the steps that can be taken to optimize them.

The performance of a workflow can be measured in terms of the time it takes to
complete and the amount of memory it uses. Both these aspects are heavily
dependent on the size of the data being processed and how that data is loaded
into memory.

Another important aspect is the operations being performed on the data. For
example, there can be situations where we either need all the data in
memory at once, or where we can process the data in chunks.

So here we will compare both the time and memory usage of different methods of
loading and then operations where we need all the data in memory at once and
where we can process the data in chunks.

Proxy images and array images
=============================

A nifti image can be loaded as a proxy image or an array image. This page on
Nibabel documentation does a good job of explaining the difference between the
two: https://nipy.org/nibabel/images_and_memory.html

But in short, a proxy image is an object that only points to the actual numpy
array data on disk. This means that the data is not loaded into memory until
it is accessed. On the other hand, an array image is an object that loads the
data into memory as soon as it is created.

If you are reading an image from the disk, you can do so via nibabel's
``load`` and nilearn's ``load_img`` function. Both of these functions return
a proxy image. The difference is that with nibabel's ``load`` you
only get the proxy image and you have to call the ``get_fdata`` method to load
the data into memory. On the other hand, with nilearn's ``load_img`` you get a
proxy image that loads the data into memory as soon as it is created.

Time taken to load an image
---------------------------

So we expect that when simply loading the image nibabel's ``load`` would be
faster and lower on memory usage (because it doesn't load the data into memory)
than nilearn's ``load_img``.

..code-block:: python

    import nibabel as nib
    from nilearn.image import load_img

    # load image via nibabel.load
    %time nib.load(example_fmri_path)

    # CPU times: user 1.78 ms, sys: 4.07 ms, total: 5.85 ms
    # Wall time: 5.09 ms

    # load image via nilearn.image.load_img
    %time load_img(example_fmri_path)

    # CPU times: user 4.48 s, sys: 1.88 s, total: 6.36 s
    # Wall time: 7.44 s


Memory usage while loading an image
--------------------------------------

We can also measure the memory usage of each of these methods using the
``memory_profiler`` package. Once we have installed the package, we can use
``%memit`` magic command to measure the memory usage of a single line of code.

.. code-block:: python

    %load_ext memory_profiler

    # load image via nibabel.load
    %memit nib.load(example_fmri_path)

    # peak memory: 570.31 MiB, increment: 0.05 MiB

    # load image via nilearn.image.load_img
    %memit load_img(example_fmri_path)

    # peak memory: 2789.92 MiB, increment: 2298.12 MiB


More use cases
==============

Once we have loaded the image, we can perform various operations on it.
We will consider two cases here:

    1. Taking the mean over the time axis, which requires all the data to be
       loaded into memory at once.
    2. Extracting a 3D volume at a given time point from the 4D image, which
       only requires a chunk of data to be loaded into memory.

Mean over the time axis
-----------------------

To take the mean over the time axis, we can use the ``mean_img`` function from
nilearn. This function requires all the data to be loaded into memory at once.

..code-block:: python

    from nilearn.image import mean_img

    img_nilearn = load_img(example_fmri_path)
    img_nibabel = nib.load(example_fmri_path)

    # mean over image loaded via nilearn.image.load_img
    %time mean_img(img_nilearn, copy_header=True)
    # CPU times: user 225 ms, sys: 324 ms, total: 549 ms
    # Wall time: 555 ms

    %memit mean_img(img_nilearn, copy_header=True)
    # peak memory: 3669.36 MiB, increment: 3487.14 MiB


    # mean over image loaded via nibabel.load
    %time mean_img(img_nibabel, copy_header=True)
    # CPU times: user 4.84 s, sys: 2.29 s, total: 7.13 s
    # Wall time: 8.79 s

    %memit mean_img(img_nibabel, copy_header=True)
    # peak memory: 3668.64 MiB, increment: 3483.02 MiB
