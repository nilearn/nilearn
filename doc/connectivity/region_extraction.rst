.. _region_extraction:

================================================
Region Extraction for better brain parcellations
================================================

.. topic:: **Page summary**

   This section shows how to use :class:`~nilearn.regions.RegionExtractor`
   to extract connected regions/components into a separate brain
   region and also shows how to learn functional connectivity
   interactions between each separate region.

.. topic:: **References**

   * `Abraham et al. "Region segmentation for sparse decompositions: better
     brain parcellations from rest fMRI", Sparsity Techniques in Medical Imaging,
     Sep 2014 <https://hal.inria.fr/hal-01093944>`_

.. currentmodule:: nilearn.datasets

Fetching movie-watching based functional datasets
=================================================

We use a naturalistic stimuli based movie-watching functional connectivity dataset
of 20 subjects, which is already preprocessed, downsampled to 4mm isotropic resolution, and publicly available at
`<https://osf.io/5hju4/files/>`_. We use utilities
:func:`fetch_development_fmri` implemented in nilearn for automatic fetching of this
dataset.


.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # We use nilearn's datasets downloading utilities
    :end-before: ##############################################################################

.. currentmodule:: nilearn.decomposition

Brain maps using :term:`Dictionary learning`
============================================

Here, we use object :class:`DictLearning`, a multi subject model to decompose multi
subjects :term:`fMRI` datasets into functionally defined maps. We do this by setting
the parameters and calling :meth:`DictLearning.fit` on the filenames of datasets without
necessarily converting each file to :class:`~nibabel.nifti1.Nifti1Image` object.


.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # functional datasets
    :end-before: # Visualization of functional networks

.. currentmodule:: nilearn.plotting

Visualization of :term:`Dictionary learning` maps
=================================================

Showing maps stored in ``components_img`` using nilearn plotting utilities.
Here, we use :func:`plot_prob_atlas` for easy visualization of 4D atlas maps
onto the anatomical standard template. Each map is displayed in different
color and colors are random and automatically picked.

.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # Show networks using plotting utilities
    :end-before: ##############################################################################

.. |dict-maps| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_extract_regions_dictlearning_maps_001.png
    :target: ../auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
    :scale: 80

.. centered:: |dict-maps|

.. currentmodule:: nilearn.regions

Region Extraction with :term:`Dictionary learning` maps
=======================================================

We use object :class:`RegionExtractor` for extracting brain connected regions
from dictionary maps into separated brain activation regions with automatic
thresholding strategy selected as ``thresholding_strategy='ratio_n_voxels'``.
We use thresholding strategy to first get foreground information present in the
maps and then followed by robust region extraction on foreground information using
Random Walker algorithm selected as ``extractor='local_regions'``.

Here, we control foreground extraction using parameter ``threshold=.5``, which
represents the expected proportion of :term:`voxels<voxel>` included in the regions
(i.e. with a non-zero value in one of the maps). If you need to keep more
proportion of :term:`voxels<voxel>` then threshold should be tweaked according to
the maps data.

The parameter ``min_region_size=1350 mm^3`` is to keep the minimum number of extracted
regions. We control the small spurious regions size by thresholding in :term:`voxel`
units to adapt well to the resolution of the image. Please see the documentation of
:func:`~nilearn.regions.connected_regions` for more details.

.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # more intense non-voxels will be survived.
    :end-before: # Visualization of region extraction results

.. currentmodule:: nilearn.plotting

Visualization of Region Extraction results
==========================================

Showing region extraction results. The same function :func:`plot_prob_atlas` is used
for visualizing extracted regions on a standard template. Each extracted brain
region is assigned a color and as you can see that visual cortex area is extracted
quite nicely into each hemisphere.

.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # Visualization of region extraction results
    :end-before: ##############################################################################

.. |dict| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_extract_regions_dictlearning_maps_002.png
    :target: ../auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
    :scale: 80

.. centered:: |dict|

.. currentmodule:: nilearn.connectome

Computing functional connectivity matrices
==========================================

Here, we use the object called :class:`ConnectivityMeasure` to compute
functional connectivity measured between each extracted brain regions. Many different
kinds of measures exists in nilearn such as "correlation", "partial correlation", "tangent",
"covariance", "precision". But, here we show how to compute only correlations by
selecting parameter as ``kind='correlation'`` as initialized in the object.

The first step to do is to extract subject specific time series signals using
functional data stored in ``func_filenames`` and the second step is to call
:meth:`ConnectivityMeasure.fit_transform` on the time series signals.
Here, for each subject we have time series signals of ``shape=(168, n_regions_extracted)``
where 168 is the length of time series and ``n_regions_extracted`` is the number of
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
    :start-after: # connectome relations.
    :end-before: ##############################################################################

.. |matrix| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_extract_regions_dictlearning_maps_003.png
   :target: ../auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
   :scale: 60

.. |connectome| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_extract_regions_dictlearning_maps_004.png
   :target: ../auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
   :scale: 60

.. centered:: |matrix| |connectome|

Validating results
==================

Showing only one specific network regions before and after region extraction. The first image displays the regions of one specific functional network without region extraction.

.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # without region extraction (left plot).
    :end-before: ##############################################################################

.. |dmn| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_extract_regions_dictlearning_maps_005.png
   :target: ../auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
   :scale: 80

.. centered:: |dmn|

The second image displays the regions split apart after region extraction. Here, we can
validate that regions are nicely separated identified by each extracted region in different
color.

.. literalinclude:: ../../examples/03_connectivity/plot_extract_regions_dictlearning_maps.py
    :start-after: # related to original network given as 4.

.. |dmn_reg| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_extract_regions_dictlearning_maps_006.png
   :target: ../auto_examples/03_connectivity/plot_extract_regions_dictlearning_maps.html
   :scale: 80

.. centered:: |dmn_reg|

.. seealso::

   The full code can be found as an example:
   :ref:`sphx_glr_auto_examples_03_connectivity_plot_extract_regions_dictlearning_maps.py`
