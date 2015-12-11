.. _region_extraction:

===================================================================
Region Extraction for better brain parcellations
===================================================================

.. topic:: **Page summary**

   This section shows how to use Region Extractor to extract each connected
   brain regions/components into a separate brain activation regions and also
   shows how to learn functional connectivity interactions between each
   separate regions.

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


.. literalinclude:: ../../examples/connectivity/plot_extract_regions_canica_maps.py
    :start-after: # utilities
    :end-before: ################################################################################

.. currentmodule:: nilearn.decomposition

Data decomposition using Canonical ICA
======================================

Here, we use :class:`CanICA`, a multi subject model to decompose previously
fetched multi subjects datasets. We do this by setting the parameters in the
object and calling fit on the functional filenames without necessarily
converting each filename to Nifti1Image object.


.. literalinclude:: ../../examples/connectivity/plot_extract_regions_canica_maps.py
    :start-after: # decomposition module
    :end-before: # Visualization

.. currentmodule:: nilearn.plotting

Visualization of Canonical ICA maps
===================================

Showing ICA maps stored in components_img using nilearn plotting utilities.
Here, we use :func:`plot_prob_atlas` for easy visualization of 4D atlas maps
onto the anatomical standard template. Each ICA map is displayed in different
color and colors are random and automatically picked.

.. literalinclude:: ../../examples/connectivity/plot_extract_regions_canica_maps.py
    :start-after: # Show ICA maps by using plotting utilities
    :end-before: ################################################################################

.. image:: ../auto_examples/connectivity/images/sphx_glr_plot_extract_regions_canica_maps_001.png
    :target: ../auto_examples/connectivity/plot_extract_regions_canica_maps.html
    :scale: 60

.. currentmodule:: nilearn.regions

Region Extraction with CanICA maps
==================================

We use object :class:`RegionExtractor` for extracting brain connected regions
from ICA maps into separated brain activation regions with automatic
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

.. literalinclude:: ../../examples/connectivity/plot_extract_regions_canica_maps.py
    :start-after: # regions, both can be done by importing Region Extractor from regions module
    :end-before: # Visualization

.. currentmodule:: nilearn.plotting

Visualization of Region Extraction results
==========================================

Showing region extraction results. The same :func:`plot_prob_atlas` is used
for visualizing extracted regions on a standard template. Each extracted brain
region is assigned a color and as you can see that visual cortex area is extracted
quite nicely into each hemisphere.

.. literalinclude:: ../../examples/connectivity/plot_extract_regions_canica_maps.py
    :start-after: # Show region extraction results
    :end-before: ################################################################################

.. image:: ../auto_examples/connectivity/images/sphx_glr_plot_extract_regions_canica_maps_002.png
    :target: ../auto_examples/connectivity/plot_extract_regions_canica_maps.html
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

.. literalinclude:: ../../examples/connectivity/plot_extract_regions_canica_maps.py
    :start-after: # To estimate correlation matrices we import connectome utilities from nilearn
    :end-before: # Visualization

.. currentmodule:: nilearn.plotting

Visualization of functional connectivity matrices
=================================================

Showing mean of correlation matrices computed between each extracted brain regions.
At this point, we make use of nilearn image and plotting utilities to find
automatically the coordinates required, for plotting connectome relations.
Left image is the correlations in a matrix form and right image is the
connectivity relations to brain regions plotted using :func:`plot_connectome`

.. literalinclude:: ../../examples/connectivity/plot_extract_regions_canica_maps.py
    :start-after: # Import image utilities in utilising to operate on 4th dimension
    :end-before: ################################################################################

.. |matrix| image:: ../auto_examples/connectivity/images/sphx_glr_plot_extract_regions_canica_maps_003.png
   :target: ../auto_examples/connectivity/plot_extract_regions_canica_maps.html
   :scale: 60

.. |connectome| image:: ../auto_examples/connectivity/images/sphx_glr_plot_extract_regions_canica_maps_004.png
   :target: ../auto_examples/connectivity/plot_extract_regions_canica_maps.html
   :scale: 60

.. centered:: |matrix| |connectome|

Validating results
==================

Showing only Default Mode Network (DMN) regions before and after region
extraction by manually identifying the index of DMN in ICA decomposed maps.

Left image displays the DMN regions without region extraction and right image
displays the DMN regions after region extraction. Here, we can validate that
the DMN regions are nicely separated displaying each extracted region in different color.

.. literalinclude:: ../../examples/connectivity/plot_extract_regions_canica_maps.py
    :start-after: # First we plot DMN without region extraction, interested in only index=[3]

.. |dmn| image:: ../auto_examples/connectivity/images/sphx_glr_plot_extract_regions_canica_maps_005.png
   :target: ../auto_examples/connectivity/plot_extract_regions_canica_maps.html
   :scale: 50
    
.. |dmn_reg| image:: ../auto_examples/connectivity/images/sphx_glr_plot_extract_regions_canica_maps_006.png
   :target: ../auto_examples/connectivity/plot_extract_regions_canica_maps.html
   :scale: 50

.. centered:: |dmn| |dmn_reg|

.. seealso::

   The full code can be found as an example:
   :ref:`sphx_glr_auto_examples_connectivity_plot_extract_regions_canica_maps.py`
