.. _data_manipulation:

=====================================================================
Manipulating images: resampling, smoothing, masking, ROIs...
=====================================================================

This chapter discusses how nilearn can be used to do simple operations on
brain images.


.. contents:: **Chapters contents**
    :local:
    :depth: 1

.. _preprocessing_functions:

Functions for data preparation steps
=====================================

.. currentmodule:: nilearn.input_data

The :class:`NiftiMasker` can automatically perform important data preparation
steps. These steps are also available as independent functions if you want to
set up your own data preparation procedure:

.. currentmodule:: nilearn

* Resampling: :func:`nilearn.image.resample_img` and :func:`nilearn.image.resample_to_img`.
  See the following examples:

  * affine transforms on data and bounding boxes: :ref:`sphx_glr_auto_examples_04_manipulating_images_plot_affine_transformation.py`,
  * resample an image to a template reference image: :ref:`sphx_glr_auto_examples_04_manipulating_images_plot_resample_to_template.py`.
 
* Computing the mean of images (along the time/4th dimension):
  :func:`nilearn.image.mean_img`
* Applying numpy functions on an image or a list of images:
  :func:`nilearn.image.math_img`
* Swapping voxels of both hemisphere (e.g., useful to homogenize masks
  inter-hemispherically):
  :func:`nilearn.image.swap_img_hemispheres`
* Smoothing: :func:`nilearn.image.smooth_img`
* Masking:

  * compute from EPI images: :func:`nilearn.masking.compute_epi_mask`
  * compute from images with a flat background:
    :func:`nilearn.masking.compute_background_mask`
  * compute for multiple sessions/subjects:
    :func:`nilearn.masking.compute_multi_epi_mask`
    :func:`nilearn.masking.compute_multi_background_mask`
  * apply: :func:`nilearn.masking.apply_mask`
  * intersect several masks (useful for multi sessions/subjects): :func:`nilearn.masking.intersect_masks`
  * unmasking: :func:`nilearn.masking.unmask`

* Cleaning signals (e.g., linear detrending, standardization,
  confound removal, low/high pass filtering): :func:`nilearn.signal.clean`

.. _resampling:

Resampling images
=================

Resampling one image to match another
-------------------------------------

:func:`nilearn.image.resample_to_img` resamples an image to a reference
image.

.. topic:: **Example**

    * :ref:`sphx_glr_auto_examples_04_manipulating_images_plot_resample_to_template.py`

.. image:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_resample_to_template_001.png
    :target: ../auto_examples/04_manipulating_images/plot_resample_to_template.html
    :scale: 55%

Resampling specific target affine, shape, or resolution
--------------------------------------------------------

The resampling procedure takes as input the
`target_affine` to resample (resize, rotate...) images in order to match
the spatial configuration defined by the new affine (i.e., matrix
transforming from voxel space into world space).

Additionally, a `target_shape` can be used to resize images
(i.e., cropping or padding with zeros) to match an expected data
image dimensions (shape composed of x, y, and z).

As a common use case, resampling can be a viable means to
downsample image quality on purpose to increase processing speed
and lower memory consumption of an analysis pipeline.
In fact, certain image viewers (e.g., FSLView) also require images to be
resampled to display overlays.

On an advanced note, automatic computation of offset and bounding box
can be performed by specifying a 3x3 matrix instead of the 4x4 affine.
In this case, nilearn computes automatically the translation part
of the transformation matrix (i.e., affine).

.. image:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_affine_transformation_002.png
    :target: ../auto_examples/04_manipulating_images/plot_affine_transformation.html
    :scale: 33%
.. image:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_affine_transformation_004.png
    :target: ../auto_examples/04_manipulating_images/plot_affine_transformation.html
    :scale: 33%
.. image:: ../auto_examples/04_manipulating_images/images/sphx_glr_plot_affine_transformation_003.png
    :target: ../auto_examples/04_manipulating_images/plot_affine_transformation.html
    :scale: 33%


.. topic:: **Special case: resampling to a given voxel size**

   Specifying a 3x3 matrix that is diagonal as a target_affine fixes the
   voxel size. For instance to resample to 3x3x3 mm voxels::

    >>> import numpy as np
    >>> target_affine = np.diag((3, 3, 3))

Masking data manually
=====================

Extracting a brain mask
------------------------

If we do not have a spatial mask of the target regions, a brain mask
can be easily extracted from the fMRI data by the
:func:`nilearn.masking.compute_epi_mask` function:

.. figure:: ../auto_examples/01_plotting/images/sphx_glr_plot_visualization_002.png
    :target: ../auto_examples/01_plotting/plot_visualization.html
    :align: right
    :scale: 50%

.. literalinclude:: ../../examples/01_plotting/plot_visualization.py
     :start-after: # Extracting a brain mask
     :end-before: # Applying the mask to extract the corresponding time series


.. _mask_4d_2_3d:

From 4D Nifti images to 2D data arrays
--------------------------------------

fMRI data is usually represented as a 4D block of data: 3 spatial
dimensions and one time dimension. In practice, we are usually
interested in working on the voxel time-series in the
brain. It is thus convenient to apply a brain mask in order to convert the
4D brain images representation into a restructured 2D data representation,
`voxel` **x** `time`, as depicted below:

.. image:: ../images/masking.jpg
    :align: center
    :width: 100%


.. literalinclude:: ../../examples/01_plotting/plot_visualization.py
     :start-after: # Applying the mask to extract the corresponding time series
     :end-before: # Find voxels of interest

.. figure:: ../auto_examples/01_plotting/images/sphx_glr_plot_visualization_003.png
    :target: ../auto_examples/01_plotting/plot_visualization.html
    :align: center
    :scale: 50


Image operations: creating a ROI mask manually
===============================================

Computing Regions of Interest (ROI) mask by ourselves requires a chain of image
operations to do from the input data to a mask over specific targets of
interest. The complete operations are listed below:

 * Fetching datasets: We use Haxby datasets and its experiments. The whole datasets
   can be fetched using a function :func:`nilearn.datasets.fetch_haxby`.
   See the documentation for more details about the Haxby datasets and its experiments.

 * Smoothing: Before building a statistical test, we do simple pre-processing step called
   image smoothing on functional images using function :func:`nilearn.image.smooth_img`
   with parameter given as fwhm=6.

 * Selecting features: Given the smoothed functional data, we select two features of
   interest with face and house experimental conditions. The method we use is a simple
   Student's t-test with scipy function :func:`scipy.stats.ttest_ind`.

 * Thresholding: Now, we threshold the statistical map to have better representation of
   voxels of interest.

 * Mask intersection and dilation: Post-processing the results with simple
   morphological operations, mask intersection and dilation. Here, we use the
   mask from the experiments and our results to select only those voxels which
   are common in both masks.

 * On the other hand, we again do `morphological dilation
   # <http://en.wikipedia.org/wiki/Dilation_(morphology)>`_ called Mask Dilation
   to more compact blobs. The function is used from
   :func:`scipy.ndimage.binary_dilation`.

 * Extracting connected components: We end with splitting the connected ROIs into two
   separate regions (ROIs), one in each hemisphere. The function **scipy.ndimage.label**
   from the scipy Python library is used in this setting.

 * Saving the result: The final voxel mask is saved using `nibabel.save` for further
   inspection with a software such as FSLView.

.. _nibabel: http://nipy.sourceforge.net/nibabel/

.. topic:: **Code**

    A complete script of above steps with full description can be found :ref:`here
    <sphx_glr_auto_examples_04_manipulating_images_plot_roi_extraction.py>`.

.. seealso::

     * :ref:`Automatic region extraction on 4D atlas images
       <sphx_glr_auto_examples_04_manipulating_images_plot_extract_rois_smith_atlas.py>`.
