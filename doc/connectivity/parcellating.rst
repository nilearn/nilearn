.. _parcellating_brain:

==============================================
Clustering to parcellate the brain in regions
==============================================

This page discusses how clustering can be used to parcellate the brain
into homogeneous regions from functional imaging data.

|

.. topic:: **Reference**

   A big-picture reference on the use of clustering for brain
   parcellations.

    Thirion, et al. `"Which fMRI clustering gives good brain
    parcellations?."
    <http://journal.frontiersin.org/article/10.3389/fnins.2014.00167/full>`_
    Frontiers in neuroscience 8.167 (2014): 13.

Data loading and massaging
===========================

Resting-state data
-------------------

.. currentmodule:: nilearn.datasets

Clustering is commonly applied to resting-state data, but any brain
functional data will give rise of a functional parcellation, capturing
intrinsic brain architecture in the case of resting-state data.
In the examples, we use rest data downloaded with the function 
:func:`fetch_adhd` (see :ref:`loading_data`).

Loading and masking and with NiftiMasker
-----------------------------------------

Before clustering, the brain volumes need to be turned to a data matrix,
for instance of time-series. The :class:`nilearn.input_data.NiftiMasker`
extract these on a mask. If no mask is given with the data, the masker
can compute one.

The masker can perform important :ref:`preprocessing operations
<masker_preprocessing_steps>`, such as detrending signals, standardizing
them, removing confounds, or smoothing the images.

.. topic:: **Example code**

   All the steps discussed in this section can be seen implemented in
   :ref:`a full code example
   <sphx_glr_auto_examples_03_connectivity_plot_rest_clustering.py>`.

Applying clustering
==========================

.. topic:: **Which clustering to use**

    The question of which clustering method to use is in itself subject
    to debate. There are many clustering methods; their computational
    cost will vary, as well as their results. A `well-cited empirical
    comparison
    <http://journal.frontiersin.org/article/10.3389/fnins.2014.00167/full>`_
    suggests that:

    * For a large number of clusters, it is preferable to use Ward
      agglomerative clustering with spatial constraints

    * For a small number of clusters, it is preferable to use Kmeans
      clustering after spatially-smoothing the data.

    Both clustering algorithms (as well as many others) are provided by
    `scikit-learn
    <http://scikit-learn.org/stable/modules/clustering.html>`_. Ward
    clustering is the easiest to use, as it can be done with the Feature
    agglomeration object. It is also very fast. We detail it bellow.

|

**Compute a connectivity matrix**
Before applying Ward's method, we compute a spatial neighborhood matrix,
aka connectivity matrix. This is useful to constrain clusters to form
contiguous parcels (see `the scikit-learn documentation
<http://scikit-learn.org/stable/modules/clustering.html#adding-connectivity-constraints>`_)

This is done from the mask computed by the masker: a niimg from which we
extract a numpy array and then the connectivity matrix.


**Ward clustering principle**
Ward's algorithm is a hierarchical clustering algorithm: it
recursively merges voxels, then clusters that have similar signal
(parameters, measurements or time courses).

**Caching** In practice the implementation of Ward clustering first
computes a tree of possible merges, and then, given a requested number of
clusters, breaks apart the tree at the right level.

As the tree is independent of the number of clusters, we can rely on caching to speed things up when varying the
number of clusters. In Wards clustering,
the *memory* parameter is used to cache the computed component tree. You
can give it either a *joblib.Memory* instance or the name of a directory
used for caching.


.. note::

    The Ward clustering computing 1000 parcels runs typically in about 10
    seconds. Admitedly, this is very fast.

.. seealso::

   * A function :func:`nilearn.regions.connected_label_regions` which can be useful to
     break down connected components into regions. For instance, clusters defined using
     KMeans whereas it is not necessary for Ward clustering due to its
     spatial connectivity.


Using and visualizing the resulting parcellation
==================================================

.. currentmodule:: nilearn.input_data

Visualizing the parcellation
-----------------------------

For every scikit-learn clustering object, the labels of the parcellation
are found its `labels_` after fitting it to the data. To turn them into a
brain image, we need to unmask them with the :class:`NiftiMasker`
*inverse_transform* method.

Note that by default, clusters are labeled from 0 to
(n_clusters - 1), and the label 0 may be confused with a background.

To visualize the clusters, we assign random colors to each cluster
for the labels visualization.

.. figure:: ../auto_examples/03_connectivity/images/sphx_glr_plot_rest_clustering_001.png
   :target: ../auto_examples/03_connectivity/plot_rest_clustering.html
   :align: center
   :scale: 80

Compressed representation
--------------------------

The clustering can be used to transform the data into a smaller
representation, taking the average on each parcel:

- call *ward.transform* to obtain the mean value of each cluster (for each
  scan)
- call *ward.inverse_transform* on the previous result to turn it back into
  the masked picture shape

.. |left_img| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_rest_clustering_002.png
   :target: ../auto_examples/03_connectivity/plot_rest_clustering.html
   :width: 49%

.. |right_img| image:: ../auto_examples/03_connectivity/images/sphx_glr_plot_rest_clustering_003.png
   :target: ../auto_examples/03_connectivity/plot_rest_clustering.html
   :width: 49%

|left_img| |right_img|

We can see that using only 2000 parcels, the original image is well
approximated.

|

.. topic:: **Example code**

   All the steps discussed in this section can be seen implemented in
   :ref:`a full code example
   <sphx_glr_auto_examples_03_connectivity_plot_rest_clustering.py>`.


