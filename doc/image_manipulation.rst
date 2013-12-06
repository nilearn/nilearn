.. _getting_started:

==================================
Introduction to image manipulation
==================================

In neuroimaging, image manipulation is necessary to improve the quality of the
images and get information out of them. In fMRI, it is typical to use such
methods on brain maps or brain masks.

This tutorial is targeted specifically for this kind of task. In this
practical examples, we show how to extract meaningful features from a brain and
build a mask upon it. We will be using nilearn primitives along with simple yet
powerful image processing functions provided by Numpy and Scipy python packages.

As a last resort, if you are still not satisfied of your mask, you can use
specific software to edit Nifti images like FSLview.
For advanced users, scikits-image provides more complicated image processing
algorithms that may be used on brain maps and masks.

Preliminary steps
=================

Data Loading
------------

This tutorial shows how to run a typical preliminary study. We will show how to
select features and build our own mask using simple functions and nilearn
primitives. 

As usual, nibabel is used to load the data.

.. literalinclude:: ../../todo.py
    :start-after: # Fetch dataset and restrict labels to face and houses
    :end-before: # Create a function to display an axial slice

.. Perhaps this is the occasion to go deeper in explaining the affine?
 
The affine matrix is a transformation matrix used to retrieve voxel coordinates
in their original space. Diagonal values in the affine matrix are the spatial
resolutions along each axis. Different values means that the data array is
anisotropic, which is important for the smooting step.

Visualization
-------------

In this example, we will focus on a single slice of the brain. For convenience,
we define a function to display a slice easily.

.. literalinclude:: ../../todo.py
    :start-after: # Create a function to display an axial slice
    :end-before: # Smoothing

.. Put the mean raw map here

Smoothing
---------

Functional MRI data has a low signal-to-noise ratio. When using simple methods
that are not robust to noise, it is necessary to smooth the data. Smoothing is
usually applied using a gaussian function with 4mm to 8mm full-width at
half-maximum. Even if scipy provides functions, like `gaussian_filter1d`,
to perform gaussian smoothing, it does not take into account the potential
anisotropy of data express in the affine. The function
`nilearn.masking._smooth_array` has been made to 


.. literalinclude:: ../../todo.py
    :start-after: # Smoothing
    :end-before: # Run a T-test for face and houses

.. Put the mean smoothed raw map here

Selecting features
------------------

.. literalinclude:: ../../todo.py
    :start-after: # Run a T-test for face and houses
    :end-before: # Thresholding

Thresholding
------------

ICA maps are continuous. Interesting voxels are usually the voxels with higher
value. In order to extract them, we simply put voxels with lower value to 0.


.. literalinclude:: ../../todo.py
    :start-after: # Thresholding
    :end-before: # Binarization and intersection with VT mask

.. Put the thresholded maps here

There is no absolute rule to chose the right theshold. A rule of thumb could be
to keep the highest 10% voxels.

Mask intersection
-----------------

.. literalinclude:: ../../todo.py
    :start-after: # Binarization and intersection with VT mask
    :end-before: # Dilation

Dilation
--------

.. literalinclude:: ../../todo.py
    :start-after: # Dilation
    :end-before: # Identification of connected components

Extracting connected components
-------------------------------

.. literalinclude:: ../../todo.py
    :start-after: # Identification of connected components
