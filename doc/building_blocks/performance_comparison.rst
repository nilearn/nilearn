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

A nifti image can be loaded as a proxy image or an array image.
:doc:`This page <nibabel:images_and_memory>` on Nibabel's documentation does a
good job of explaining the difference between the two.

But in summary: a proxy image (on right in the schematic below) is an object
that only points to the actual numpy array data on disk. This means that the
data is not loaded into memory until it is asked for, for example, by
explicitly calling the ``.get_fdata()`` method or while performing an operation
like :func:`nilearn.image.mean_img`. On the other hand, an array image
(on the left) is an object that loads the entire data into memory as soon
as it is created.

.. mermaid::

    flowchart TD
        subgraph "Disk"
            style Disk fill:#e2eeff,stroke:#000000
            NiftiFile["NIfTI File on Disk"]
        end

        ProxyObj["Proxy Object"]
        ProxyRef["Reference to data on disk"]
        OpFunction[".get_fdata() or image operation (e.g., mean_img())"]

        ArrayObj["Array Object"]

        subgraph "Memory"
            style Memory fill:#fff2e2,stroke:#000000
            MemData["Data already loaded"]
            ArrayOp[".get_fdata() or image operation (e.g., mean_img())"]
            LoadedData["Data loaded into memory"]
        end

        NiftiFile --> ProxyObj
        ProxyObj -. ".dataobj" .-> ProxyRef
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


.. note::

    The "reference to actual data on disk" in proxy images is stored under
    the ``dataobj`` property. According to Nibabel's documentation:

        Proxy images are images that have a ``dataobj`` property that is not a
        numpy array, but an *array proxy* that can fetch the array data from
        disk.

.. _proxy_images:

Proxy images
------------

If you are reading an image from the disk, you can do so via nibabel's
:func:`nibabel.loadsave.load` and nilearn's :func:`nilearn.image.load_img`
function. Both of these functions return a proxy image. The difference is
that with :func:`nibabel.loadsave.load` you only get the proxy image and you
have to call the ``.get_fdata()`` method to load the data into memory.
With :func:`~nilearn.image.load_img` you still get a proxy image but data
loading is already done for you. You can think of this as if
:func:`~nilearn.image.load_img` is calling ``.get_fdata()`` internally before
returning the proxy image.

You can check if a proxy image is loaded into memory or not using the
``in_memory`` attribute:

.. code-block:: python

    import nibabel as nib
    from nilearn.image import load_img

    img_nibabel = nib.load(example_fmri_path)
    img_nilearn = load_img(example_fmri_path)

    img_nibabel.in_memory
    # False

    img_nilearn.in_memory
    # True

You can also remove the data from memory by calling the ``.uncache()`` method
on the proxy image:

.. code-block:: python

    img_nilearn.uncache()
    img_nilearn.in_memory
    # False

Array images
------------

In practice, you would initially only use proxy images when you load an image
from the disk. But once you perform an operation that modifies the image,
you would get an array image that exists in memory completely.

All the functions under nilearn's :mod:`nilearn.image` module return array
images. We can check this by using nibabel's
:func:`~nibabel.arrayproxy.is_proxy` function on the output image's
``dataobj`` property. This function returns ``True`` if the image is a proxy
image and ``False`` if it is an array image.

For example, let's smooth an image using :func:`nilearn.image.smooth_img`:

.. code-block:: python

    from nilearn.image import smooth_img

    # loaded a proxy image
    img_nilearn = load_img(example_fmri_path)
    nib.is_proxy(img_nilearn.dataobj)
    # True

    # smooth the image
    img_smoothed = smooth_img(img_nilearn, fwhm=6)
    # now we have an array image
    nib.is_proxy(img_smoothed.dataobj)
    # False


:func:`nibabel.loadsave.load` vs. :func:`~nilearn.image.load_img`
=================================================================

Now let's compare the two methods of loading an image in terms of time
and memory usage. For this we will use ``%time`` and ``%memit`` ipython
magic commands.

The ``%time`` command measures the time taken to run given code and is
available by default in ``ipython``. The ``%memit`` command is part of the
``memory_profiler`` package and measures the memory usage. To use it, you
need to install the package first:

.. code-block:: bash

    pip install memory_profiler

Then you can load the package in your ``ipython`` session using:

.. code-block:: python

    %load_ext memory_profiler

First let's simply load an image using both methods. This will give us a
tangible understanding of what we explained in the :ref:`proxy_images` section
above.

Time taken to load an image
---------------------------

Since :func:`nibabel.loadsave.load` does not actually load the data into
memory, it should be faster than :func:`~nilearn.image.load_img`:

.. code-block:: python

    # load image via nibabel.load
    %time nib.load(example_fmri_path)
    # CPU times: user 1.54 ms, sys: 0 ns, total: 1.54 ms
    # Wall time: 1.22 ms

    # load image via nilearn.image.load_img
    %time load_img(example_fmri_path)
    # CPU times: user 4.12 s, sys: 2.44 s, total: 6.56 s
    # Wall time: 6.56 s

Memory usage while loading an image
-----------------------------------

Similarly, it should also use less memory.
The ``%memit`` command will give us the peak memory usage and the increment
in memory usage after running the command.

The peak memory usage is the maximum amount of memory used during the execution
of the command and the increment is the amount of memory used by the
command itself given by the difference between the peak memory usage before and
after running the command.

So to avoid confusion, we will only look at the increment in memory usage
as the peak memory usage can be affected by other variables defined in the
``ipython`` session.

.. note::

    In case you are running these commands yourself sequentially, you may
    want to run them in a new ``ipython`` session to avoid any
    interference from other variables and get reliable readings.

.. code-block:: python

    # load image via nibabel.load
    %memit nib.load(example_fmri_path)
    # peak memory: 207.63 MiB, increment: 0.25 MiB

    # load image via nilearn.image.load_img
    %memit load_img(example_fmri_path)
    # peak memory: 4143.26 MiB, increment: 3936.60 MiB

Some practical use cases
========================

Now let's look at some use cases where these two ways of loading an image could
affect the performance of a workflow.

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
    %time mean_img(img_nilearn, copy_header=True)
    # CPU times: user 154 ms, sys: 3.09 ms, total: 157 ms
    # Wall time: 167 ms

But when compared to loading the image with :func:`nibabel.loadsave.load`:

.. code-block:: python

    img_nibabel = nib.load(example_fmri_path)
    %time mean_img(img_nibabel, copy_header=True)
    # CPU times: user 4.14 s, sys: 1.36 s, total: 5.51 s
    # Wall time: 5.5 s

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
    # CPU times: user 4.09 s, sys: 1.42 s, total: 5.5 s
    # Wall time: 5.5 s

The memory usage of the two would also be similar for the same reason.

.. code-block:: python

    %memit mean_nilearn(example_fmri_path)
    # peak memory: 4144.13 MiB, increment: 3936.58 MiB

    %memit mean_nibabel(example_fmri_path)
    # peak memory: 4145.63 MiB, increment: 3936.86 MiB

Extracting a 3D volume
----------------------

Now let's say we want to extract a 3D volume at some time point from the
4D image. Here we only need that 3D volume to be loaded into memory.

We can do this by simply using the ``dataobj`` property of the proxy image.
Let's define two functions that load the image and then extract a 3D volume at
4th time point: one using :func:`~nilearn.image.load_img` and the other using
:func:`nibabel.loadsave.load`:

.. code-block:: python

    def slice_nilearn(fmri):
        img_nilearn = load_img(fmri)
        img_nilearn.dataobj[..., 3]

    def slice_nibabel(fmri):
        img_nibabel = nib.load(fmri)
        img_nibabel.dataobj[..., 3]

Now timing the two functions we see that ``slice_nilearn`` takes much more
time than ``slice_nibabel``:

.. code-block:: python

    %time slice_nilearn(example_fmri_path)
    # CPU times: user 3.93 s, sys: 1.26 s, total: 5.19 s
    # Wall time: 5.19 s

.. code-block:: python

    %time slice_nibabel(example_fmri_path)
    # CPU times: user 10.2 ms, sys: 1.87 ms, total: 12.1 ms
    # Wall time: 11.7 ms

What happens here with :func:`~nilearn.image.load_img` is that we load the
entire image into memory even though we only need a chunk of it. This is why it
takes more time than :func:`nibabel.loadsave.load` which only loads the chunk
of data we need.

We will see that with the memory usage as well:

.. code-block:: python

    %memit slice_nilearn(example_fmri_path)
    # peak memory: 4143.89 MiB, increment: 3936.76 MiB

.. code-block:: python

    %memit slice_nibabel(example_fmri_path)
    # peak memory: 209.62 MiB, increment: 2.12 MiB

So, overall, if you are performing certain operations that only
require a chunk of data in the memory, it would be beneficial to make sure
you're working with a proxy image loaded via :func:`nibabel.loadsave.load`.

However, if you will need all the data in memory at once (i.e., as we saw with
:func:`~nilearn.image.mean_img`), you can use any of the two methods to load
the image.

Finally, another possible use case could be when you want to perform several
operations on the same image in parallel.

We examine such a case in detail in this example:
:ref:`sphx_glr_auto_examples_07_advanced_plot_mask_large_fmri.py`.
