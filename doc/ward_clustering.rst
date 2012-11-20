.. _nyu_rest:

===============
fMRI clustering
===============

.. currentmodule:: nisl.datasets

Resting-state dataset
========================

Here, we use a `resting-state <http://www.nitrc.org/projects/nyu_trt/>`_ 
dataset from test-retest study performed at NYU. Details on the data 
can be found in the documentation for the downloading function 
:func:`fetch_nyu_rest`.

Preprocessing: loading and masking
==================================

As seen in :ref:`previous sections <downloading_data>`, we
fetch the data
from internet and load it with a provided function:

.. literalinclude:: ../plot_rest_clustering.py
    :start-after: ### Load nyu_rest dataset #####################################################
    :end-before: ### Ward ######################################################################

No mask is given with the data so we let the masker compute one.
The result is a niimg from which we extract a numpy array that is
used to mask our original *X*.

Applying Ward clustering
==========================

Compute connectivity map
------------------------

Before computing the ward itself, we compute a connectivity map. This is
useful to constrain clusters to form contiguous parcels (see `the
scikit-learn documentation
<http://www.scikit-learn.org/stable//modules/clustering.html#adding-connectivity-constraints>`_)

.. literalinclude:: ../plot_rest_clustering.py
    :start-after: # Compute connectivity matrix: which voxel is connected to which
    :end-before: # Computing the ward for the first time, this is long...

Principle
---------

The Ward algorithm is a hierarchical clustering algorithm: it
successfully merges together voxels that have similar timecourses.

Caching
-------

Note that in practice the scikit-learn implementation of the Ward
clustering first computes a tree of possible merges, and then, the
requested number of clusters breaks it apart the tree at the right level.

As no matter how many clusters we want, we do not need to compute the
tree again, we can rely on caching to speed things up when varying the
number of cluster. Scikit-learn integrates a transparent caching library
(`joblib <http://packages.python.org/joblib/>`_). In the ward clustering,
the *memory* parameter is used to cache the computed component tree. You
can give it either a *joblib.Memory* instance or the name of directory
used for caching.

Apply the ward
--------------

Here we simply launch the ward to find 1000 clusters and we time it.

.. literalinclude:: ../plot_rest_clustering.py
    :start-after: # Computing the ward for the first time, this is long...
    :end-before: # Compute the ward with more clusters, should be faster

This runs in about 10 seconds (depending on your computer configuration). Now,
we are not satisfied of the result and we want to cluster the picture in 2000
elements.

.. literalinclude:: ../plot_rest_clustering.py
    :start-after: # Compute the ward with more clusters, should be faster
    :end-before: ### Show result ############################################################### 

Now that the component tree has been computed, computation is must faster
thanks to caching. You should have the result in less than 1 second.

Post-Processing and visualization
===================================

Unmasking
---------

After applying the ward, we must unmask the data. This can be done simply :

.. literalinclude:: ../plot_rest_clustering.py
    :start-after: # Unmask data
    :end-before: # Display the labels 

You can see that masked data is filled with -1 values. This is done for the
sake of visualization. In fact, clusters are labeled with going from 0 to
(n_clusters - 1). By putting every other values to -1, we assure that
uninteresting values will not mess with the visualization.

Label visualization
--------------------

We can visualize the clusters. We assign random colors to each cluster
for the labels visualization.

.. literalinclude:: ../plot_rest_clustering.py
    :start-after: # Display the labels 
    :end-before: # Display the original data


.. figure:: auto_examples/images/plot_rest_clustering_1.png
   :target: auto_examples/plot_rest_clustering.html
   :align: center
   :scale: 60

Compressed picture
------------------

By transforming a picture in a new one in which the value of each voxel
is the mean value of the cluster it belongs to, we are creating a
compressed version of the original picture. We can obtain this
representation thanks to a two step procedure :

- call *ward.transform* to obtain the mean value of each cluster (for each
  scan)
- call *ward.inverse_transform* on the previous result to turn it back into
  the masked picture shape

.. literalinclude:: ../plot_rest_clustering.py
    :start-after: # Display the original data

.. |left_img| image:: auto_examples/images/plot_rest_clustering_2.png
   :target: auto_examples/plot_rest_clustering.html
   :width: 49%

.. |right_img| image:: auto_examples/images/plot_rest_clustering_3.png
   :target: auto_examples/plot_rest_clustering.html
   :width: 49%

|left_img| |right_img|

We can see that using only 2000 parcels, we can approximate well the
original image.

