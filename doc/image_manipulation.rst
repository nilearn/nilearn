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

.. literalinclude:: ../plot_image_manipulation.py
    :start-after: # Fetch dataset
    :end-before: # Create a function to display an axial slice

Visualization
-------------

In this example, we will focus on a single slice of the brain. For convenience,
we define a function to display a slice easily.

.. literalinclude:: ../plot_image_manipulation.py
    :start-after: # Create a function to display an axial slice
    :end-before: # Smoothing

Smoothing
---------

Functional MRI data has a low signal-to-noise ratio. When using simple methods
that are not robust to noise, it is necessary to smooth the data. Smoothing is
usually applied using a gaussian function with 4mm to 8mm full-width at
half-maximum. Even if scipy provides functions, like `gaussian_filter1d`,
to perform gaussian smoothing, it does not take into account the potential
anisotropy of data expressed in the affine. The function
`nilearn.image.smooth` takes care of that for you and can even take the path to
the file as a parameter.


.. literalinclude:: ../plot_image_manipulation.py
    :start-after: # Smoothing
    :end-before: # Run a T-test for face and houses

.. figure:: auto_examples/images/plot_image_manipulation_1.png
    :target: auto_examples/plot_image_manipulation.html
    :align: center
    :scale: 50%

Selecting features
------------------

Functional MRI data are high dimensional comparend to the number of samples
(usually 50000 voxels for 1000 samples). In this setting, machine learning
algorithm can perform poorly. However, a simple statistical test can help
reducing the number of voxels.

The Student's t-test performs a simple statistical test that determines if two
distributions are statistically different. It can be used to compare voxel
timeseries in two different conditions (when houses or faces are shown in our
case). If the timeserie distribution is similar in the two conditions, then the
voxel is not very interesting to discriminate the condition.

This test returns p-values that represents probabilities that the two
timeseries are drawn from the same distribution. The lower is the p-value, the
more discriminative is the voxel.

.. literalinclude:: ../plot_image_manipulation.py
    :start-after: # Run a T-test for face and houses
    :end-before: # Thresholding

.. figure:: auto_examples/images/plot_image_manipulation_2.png
    :target: auto_examples/plot_image_manipulation.html
    :align: center
    :scale: 50%

This feature selection method is available in the scikit-learn where it has been
extended to several classes (TODO: put ref to f_classif).

Thresholding
------------

Higher p-values are kept as voxels of interest. Applying a threshold to an array
is easy thanks to numpy indexing a la Matlab.

.. literalinclude:: ../plot_image_manipulation.py
    :start-after: # Thresholding
    :end-before: # Binarization and intersection with VT mask

.. figure:: auto_examples/images/plot_image_manipulation_3.png
    :target: auto_examples/plot_image_manipulation.html
    :align: center
    :scale: 50%

Mask intersection
-----------------

We now want to restrict our study to the ventral temporal area. The
corresponding mask is provided in `haxby.mask_vt`. We want to compute the
intersection of this mask with our mask. The first step is to load it with
nibabel. We then use a logical and to keep only voxels that are selected in both
masks.

.. literalinclude:: ../plot_image_manipulation.py
    :start-after: # Binarization and intersection with VT mask
    :end-before: # Dilation

.. figure:: auto_examples/images/plot_image_manipulation_4.png
    :target: auto_examples/plot_image_manipulation.html
    :align: center
    :scale: 50%

Dilation
--------

We observe that our voxels are a bit scattered across the brain. To obtain more
compact shape, we use a morphological dilation. This is a common step to be sure
not to forget voxels located on the edge of a ROI.

.. literalinclude:: ../plot_image_manipulation.py
    :start-after: # Dilation
    :end-before: # Identification of connected components

.. figure:: auto_examples/images/plot_image_manipulation_5.png
    :target: auto_examples/plot_image_manipulation.html
    :align: center
    :scale: 50%

Extracting connected components
-------------------------------

Scipy function `ndimage.label` identify connected components in our final mask.

.. literalinclude:: ../plot_image_manipulation.py
    :start-after: # Identification of connected components
    :end-before: # Save the result

.. figure:: auto_examples/images/plot_image_manipulation_6.png
    :target: auto_examples/plot_image_manipulation.html
    :align: center
    :scale: 50%

Save the result
---------------

The final result is saved using nibabel for further consultation with a software
like FSLview for example.

.. literalinclude:: ../plot_image_manipulation.py
    :start-after: # Save the result
