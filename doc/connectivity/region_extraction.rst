.. _region_extraction:

===================================================================
Region Extraction for better brain parcellations
===================================================================

.. topic:: **Page summary**

   This section shows how to use Region Extractor to extract brain connected
   regions/components into a separate brain activation region and also
   shows how to learn functional connectivity interactions between each
   separate region.

.. contents:: **Contents**
    :local:
    :depth: 1


.. topic:: **References**

   * `Abraham et al. "Region segmentation for sparse decompositions: better
     brain parcellations from rest fMRI", Sparsity Techniques in Medical Imaging,
     Sep 2014 <https://hal.inria.fr/hal-01093944>`_

.. currentmodule:: nilearn.datasets

Fetching resting state functional datasets
==========================================

We use ADHD resting state functional connectivity datasets of 20 subjects,
which is already preprocessed and publicly available at
`<http://fcon_1000.projects.nitrc.org/indi/adhd200/>`_. We use utilities
:func:`fetch_adhd` implemented in nilearn for automatic fetching of these
datasets.


.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # We use nilearn's datasets downloading utilities
    :end-before: ################################################################################

.. currentmodule:: nilearn.decomposition

Brain maps using Dictionary Learning
====================================

Here, we use object :class:`DictLearning`, a multi subject model to decompose multi
subjects fMRI datasets into functionally defined maps. We do this by setting
the parameters and calling the object fit on the filenames of datasets without
necessarily converting each file to Nifti1Image object.


.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # object and fit the model to the functional datasets
    :end-before: # Visualization of resting state networks

.. currentmodule:: nilearn.plotting

Visualization of Dictionary Learning maps
=========================================

Showing maps stored in components_img using nilearn plotting utilities.
Here, we use :func:`plot_prob_atlas` for easy visualization of 4D atlas maps
onto the anatomical standard template. Each map is displayed in different
color and colors are random and automatically picked.

.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # Show networks using plotting utilities
    :end-before: ################################################################################

.. image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_extract_regions_dictlearning_maps_001.png
    :target: ../auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
    :scale: 60

.. currentmodule:: nilearn.regions

Region Extraction with Dictionary Learning maps
===============================================

We use object :class:`RegionExtractor` for extracting brain connected regions
from dictionary maps into separated brain activation regions with automatic
thresholding strategy selected as thresholding_strategy='ratio_n_voxels'. We use
thresholding strategy to first get foreground information present in the maps and
then followed by robust region extraction on foreground information using
Random Walker algorithm selected as extractor='local_regions'.

Here, we control foreground extraction using parameter threshold=.5, which
represents the expected proportion of voxels included in the regions
(i.e. with a non-zero value in one of the maps). If you need to keep more
proportion of voxels then threshold should be tweaked according to the maps data.

The parameter min_region_size=1350 mm^3 is to keep the minimum number of extracted
regions. We control the small spurious regions size by thresholding in voxel units
to adapt well to the resolution of the image. Please see the documentation of
nilearn.regions.connected_regions for more details.

.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # maps, less the threshold means that more intense non-voxels will be survived.
    :end-before: # Visualization of region extraction results

.. currentmodule:: nilearn.plotting

Visualization of Region Extraction results
==========================================

Showing region extraction results. The same :func:`plot_prob_atlas` is used
for visualizing extracted regions on a standard template. Each extracted brain
region is assigned a color and as you can see that visual cortex area is extracted
quite nicely into each hemisphere.

.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # Visualization of region extraction results
    :end-before: ################################################################################

.. image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_extract_regions_dictlearning_maps_002.png
    :target: ../auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
    :scale: 60

.. currentmodule:: nilearn.connectome

Computing functional connectivity matrices
==========================================

Here, we use the object called :class:`ConnectivityMeasure` to compute
functional connectivity measured between each extracted brain regions. Many different
kinds of measures exists in nilearn such as "correlation", "partial correlation", "tangent",
"covariance", "precision". But, here we show how to compute only correlations by
selecting parameter as kind='correlation' as initialized in the object.

The first step to do is to extract subject specific time series signals using
functional data stored in func_filenames and the second step is to call fit_tranform()
on the time series signals. Here, for each subject we have time series signals of
shape=(176, 23) where 176 is the length of time series and 23 is the number of
extracted regions. Likewise, we have a total of 20 subject specific time series signals.
The third step, we compute the mean correlation across all subjects.

.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # To estimate correlation matrices we import connectome utilities from nilearn
    :end-before: #################################################################

.. currentmodule:: nilearn.plotting

Visualization of functional connectivity matrices
=================================================

Showing mean of correlation matrices computed between each extracted brain regions.
At this point, we make use of nilearn image and plotting utilities to find
automatically the coordinates required, for plotting connectome relations.
Left image is the correlations in a matrix form and right image is the
connectivity relations to brain regions plotted using :func:`plot_connectome`

.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # Plot resulting connectomes
    :end-before: ################################################################################

.. |matrix| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_extract_regions_dictlearning_maps_003.png
   :target: ../auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
   :scale: 60

.. |connectome| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_extract_regions_dictlearning_maps_004.png
   :target: ../auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
   :scale: 60

.. centered:: |matrix| |connectome|

Validating results
==================

Showing only one specific network regions before and after region extraction.

Left image displays the regions of one specific resting network without region extraction
and right image displays the regions split apart after region extraction. Here, we can
validate that regions are nicely separated identified by each extracted region in different
color.

.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # First, we plot a network of index=4 without region extraction

.. |dmn| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_extract_regions_dictlearning_maps_005.png
   :target: ../auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
   :scale: 50
    
.. |dmn_reg| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_extract_regions_dictlearning_maps_006.png
   :target: ../auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
   :scale: 50

.. centered:: |dmn| |dmn_reg|

.. seealso::

   The full code can be found as an example:
   :ref:`sphx_glr_auto_examples_03_connectivity_plot_extract_regions_dictlearning_maps.py`
