.. _performance_comparison:

=======================================================
Factors affecting performance of neuroimaging workflows
=======================================================

In this chapter we will discuss the factors that can affect the performance of
neuroimaging workflows and the steps that can be taken to optimize them.

The performance of a workflow can be measured in:

1. the time it takes to complete
2. the amount of memory it uses

Both these aspects are heavily dependent on the size of the data being
processed and how that data is loaded into memory.

Another important aspect is the operations being performed on the data. For
example, there can be situations where we either need all the data in
memory at once, or where we can process the data in chunks.

So here we will compare both the time and memory usage of different methods of
loading and then operations where we need all the data in memory at once and
where we can process the data in chunks.

Proxy images vs. array images
=============================

A nifti image can be loaded as a proxy image or an array image. This page on
:doc:`Nibabel documentation <nibabel:images_and_memory>` does a good job
of explaining the difference between the two.

But TL;DR: a proxy image (on the left in the figure below) is an object that
only points to the actual numpy array data on disk. This means that the data
is not loaded into memory until it is accessed, for example, by explicitly
calling the ``.get_fdata()`` method or while performing an operation like
:func:`nilearn.image.mean_img`. On the other hand, an array image is an object
that loads the data into memory as soon as it is created.

.. mermaid::

    flowchart TD
        subgraph "Disk"
            style Disk fill:#e2eeff,stroke:#000000
            NiftiFile["NIfTI File on Disk"]
        end

        ProxyObj["Proxy Object"]
        ProxyRef["Reference to data on disk"]
        OpFunction["Operation (e.g., mean_img)"]

        ArrayObj["Array Object"]

        subgraph "Memory"
            style Memory fill:#fff2e2,stroke:#000000
            MemData["Data already loaded"]
            ArrayOp[".get_fdata()"]
            LoadedData["Data loaded into memory"]
        end

        NiftiFile --> ProxyObj
        ProxyObj --> ProxyRef
        ProxyRef --> OpFunction
        OpFunction --> LoadedData

        NiftiFile --> ArrayObj
        ArrayObj --> MemData
        MemData --> ArrayOp

        style ProxyObj fill:#d4f1f9,stroke:#0077b6
        style ArrayObj fill:#ffdfd3,stroke:#e07a5f
        style OpFunction fill:#d8f3dc,stroke:#2d6a4f
        style ArrayOp fill:#d8f3dc,stroke:#2d6a4f
        style LoadedData fill:#ffdfd3,stroke:#e07a5f
        style MemData fill:#ffdfd3,stroke:#e07a5f

Proxy images
============

If you are reading an image from the disk, you can do so via nibabel's
:func:`nibabel.loadsave.load` and nilearn's :func:`nilearn.image.load_img`
function. Both of these functions return a proxy image. The difference is
that with :func:`nibabel.loadsave.load` you only get the proxy image and you
have to call the ``.get_fdata()`` method to load the data into memory.
On the other hand, with :func:`~nilearn.image.load_img` you get a proxy image
that loads the data into memory as soon as it is created.

Time taken to load an image
---------------------------

So we expect that when simply loading the image, :func:`nibabel.loadsave.load`
would be faster and lower on memory usage (because it doesn't load the data
into memory) compared to :func:`~nilearn.image.load_img`.

.. code-block:: python

    import nibabel as nib
    from nilearn.image import load_img

    # load image via nibabel.load
    %time nib.load(example_fmri_path)
    # CPU times: user 2.77 ms, sys: 3.76 ms, total: 6.53 ms
    # Wall time: 5.72 ms

    # load image via nilearn.image.load_img
    %time load_img(example_fmri_path)
    # CPU times: user 6.19 s, sys: 2.89 s, total: 9.08 s
    # Wall time: 9.07 s

Memory usage while loading an image
-----------------------------------

We can also measure the memory usage of each of these methods using the
``memory_profiler`` package. Once we have installed the package (via
``pip install memory_profiler``), we can use ``%memit`` magic command to
measure the memory usage of a single line of code.

.. code-block:: python

    %load_ext memory_profiler

    # load image via nibabel.load
    %memit nib.load(example_fmri_path)
    # peak memory: 2180.11 MiB, increment: 0.25 MiB

    # load image via nilearn.image.load_img
    %memit load_img(example_fmri_path)
    # peak memory: 6116.31 MiB, increment: 3936.18 MiB

Some use cases
==============

Once we have loaded the image, we can perform various operations on it.
We will consider two cases here:

1. Taking the mean over the time axis, which requires all the data to be
   loaded into memory at once.
2. Extracting a 3D volume at a given time point from the 4D image, which
   only requires a chunk of data to be loaded into memory.

Mean over the time axis
-----------------------

To take the mean over the time axis, we can use :func:`nilearn.image.mean_img`.
This function requires all the data to be loaded into memory at once.

So when we load the image with :func:`~nilearn.image.load_img` and then pass it
to :func:`~nilearn.image.mean_img` function, the data is readily available in
memory and the function can operate quickly.

.. code-block:: python

    from nilearn.image import mean_img

    img_nilearn = load_img(example_fmri_path)
    # mean over image loaded via nilearn.image.load_img
    %time mean_img(img_nilearn, copy_header=True)
    # CPU times: user 734 ms, sys: 309 ms, total: 1.04 s
    # Wall time: 1.04 s

But when compared to loading the image with :func:`nibabel.loadsave.load`:

.. code-block:: python

    img_nibabel = nib.load(example_fmri_path)
    # mean over image loaded via nibabel.load
    %time mean_img(img_nibabel, copy_header=True)
    # CPU times: user 7.35 s, sys: 5.74 s, total: 13.1 s
    # Wall time: 13.1 s

This takes more time because :func:`~nilearn.image.mean_img` will have to load
the data before it can take the mean.

But it is important to note that the overall time taken to first load the
image and take the mean is similar for both the methods.
This is simply because the data has to be loaded at some point either before
(i.e., with :func:`~nilearn.image.load_img`) or within
:func:`~nilearn.image.mean_img`.

We can verify that by adding the timing of the loading and
:func:`~nilearn.image.mean_img` calculation together. Let's define functions
that load the image and then take the mean one for each of the two loading
methods.

.. code-block:: python

    def mean_nilearn(fmri):
        img_nilearn = load_img(fmri)
        mean_img(img_nilearn, copy_header=True)

    def mean_nibabel(fmri):
        img_nibabel = nib.load(fmri)
        mean_img(img_nibabel, copy_header=True)

.. code-block:: python

    %time mean_nilearn(example_fmri_path)
    # CPU times: user 7.14 s, sys: 3.45 s, total: 10.6 s
    # Wall time: 10.6 s

The memory usage of the two would also be similar for the same reason.

.. code-block:: python

    %memit mean_nilearn(example_fmri_path)
    # peak memory: 10060.05 MiB, increment: 3935.48 MiB

    %memit mean_nibabel(example_fmri_path)
    # peak memory: 10060.05 MiB, increment: 3935.48 MiB

Extracting a 3D volume
----------------------

Now let's say we want to extract a 3D volume at some time point from the
4D image. Here we only need that 3D volume to be loaded into memory.

Proxy images come with an attribute called ``.dataobj`` that allows us to
directly access the chunk of data we need.

So with :func:`~nilearn.image.load_img`:

.. code-block:: python

    def slice_nilearn(fmri):
        img_nilearn = load_img(fmri)
        img_nilearn.dataobj[..., 3]

    def slice_nibabel(fmri):
        img_nibabel = nib.load(fmri)
        img_nibabel.dataobj[..., 3]

.. code-block:: python

    %time slice_nilearn(example_fmri_path)
    # CPU times: user 7.39 s, sys: 5.64 s, total: 13 s
    # Wall time: 13 s

And with :func:`nibabel.loadsave.load`:

.. code-block:: python

    %time slice_nibabel(example_fmri_path)
    # CPU times: user 24.5 ms, sys: 4.24 ms, total: 28.7 ms
    # Wall time: 27 ms

What happens here with :func:`~nilearn.image.load_img` is that we load the
entire image into memory even though we only need a chunk of it. This is why it
takes more time than :func:`nibabel.loadsave.load` which only loads the chunk
of data we need.

We will see that with the memory usage as well:

.. code-block:: python

    %memit slice_nilearn(example_fmri_path)
    # peak memory: 10060.75 MiB, increment: 3935.48 MiB

.. code-block:: python

    %memit slice_nibabel(example_fmri_path)
    # peak memory: 6120.99 MiB, increment: 0.00 MiB

Array images
============

In practice, you would initially only use proxy images when you load an image
from the disk. But once you perform an operation that modifies the image,
you would get an array image; i.e., one that is loaded to disk as a numpy
array.

For example, if you smooth an image using :func:`nilearn.image.smooth_img`
function, it will return an array image. We can check this using nibabel's
:func:`nibabel.arrayproxy.is_proxy` function on the image's ``dataobj``
property.

.. code-block:: python

    from nilearn.image import smooth_img

    img_nilearn = load_img(example_fmri_path)
    img_smoothed = smooth_img(img_nilearn, fwhm=6)
    nib.is_proxy(img_smoothed.dataobj)
    # False

But :func:`nibabel.arrayproxy.is_proxy` would return ``True`` for
``img_nilearn.dataobj``:

.. code-block:: python

    nib.is_proxy(img_nilearn.dataobj)
    # True

So if you are performing subsequent operations that only require a chunk of
data in the memory, it could be beneficial to first save the image to disk and
then loading it again via :func:`nibabel.loadsave.load` function to get a
proxy image.

However, if you will need all the data in memory at once (i.e., as we saw with
:func:`~nilearn.image.mean_img`), you can directly use the array image in
subsequent operations.

This applies to most of the operations under nilearn's :mod:`nilearn.image`
module as they all return array images.

Finally, another possible use case could be when you want to perform several
operations on the same image in parallel.

We examine such a case in detail in this example:
:ref:`sphx_glr_auto_examples_07_advanced_plot_mask_large_fmri.py`.
