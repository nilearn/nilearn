.. _nyu_rest:

===============
fMRI clustering
===============

.. currentmodule:: nisl.datasets

Resting-state dataset
========================

Here, we use a resting-state dataset from test-retest study performed at
NYU. Details on the data can be found in the documentation for the
downloading function :func:`fetch_nyu_rest`.

Preprocessing
==============

Loading
-------

Thanks to nisl dataset manager, fetching the dataset is rather easy. Do not
forget to set your environment variable *NISL_DATA* is you want your dataset
to be stored in a specific path.

.. literalinclude:: ../plot_rest_clustering.py
        :start-after: ### Load nyu_rest dataset #####################################################
        :end-before: ### Mask ######################################################################

Masking
-------

No mask is given with the data so we have to compute one ourselves. We use a
simple heuristic proposed by T. Nichols based on the picture histogram. The
idea is to threshold values and eliminates voxels present in the "black peak"
(peak in the histogram representing background dark voxels).

.. literalinclude:: ../plot_rest_clustering.py
        :start-after: ### Mask ######################################################################
        :end-before: ### Ward ######################################################################

The result is a numpy array of boolean that is used to mask our original X.

Applying Ward clustering
==========================

Compute connectivity map
------------------------

Before computing the ward itself, it is necessary to compute a connectivity
map. Otherwise, the ward will consider each voxel independantly and this may
lead to a wrong clustering (see
http://www.scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html)

.. literalinclude:: ../plot_rest_clustering.py
        :start-after: # Compute connectivty map 
        :end-before: # Computing the ward for the first time, this is long...

Principle
---------

The Ward algorithm computes a hierarchical tree of the picture components.
Consequently, the root of the tree is the sum of all components (ie the whole
picture) and there are as many leaves in the tree as components in the picture.

Once that the tree is computed, the only step left to obtain the requested
number of components is cutting the tree at the right level. No matter how many
clusters we want, we do not need to compute the tree again.

Caching
-------

Joblib is a library made to put in cache some function calls to avoid
unnecessary computation. If a function is called with joblib, the parameters
and results are cached. If the same function is called using the same
parameters, then joblib bypass the function call and returns the previously
computed result immediately.

Scikit-learn integrates joblib in a user friendly way to cache results of some
function calls when it is possible. For example, in the ward clustering, the
*memory* parameter allows the user to create a joblib cache to store the
computed component tree. Either a *joblib.Memory* instance or a unique string
identifier can be passed as a *memory* parameter.

Apply the ward
--------------

Here we simply launch the ward to find 500 clusters and we time it.

.. literalinclude:: ../plot_rest_clustering.py
        :start-after: # Computing the ward for the first time, this is long...
        :end-before: # Compute the ward with more clusters, should be faster

This runs in about 10 seconds (depending on your computer configuration). Now,
we are not satisfied of the result and we want to cluster the picture in 1000
elements.

.. literalinclude:: ../plot_rest_clustering.py
        :start-after: # Compute the ward with more clusters, should be faster
        :end-before: ### Spectral clustering #######################################################

Now that the component tree has been computed, computation is must faster. You
should have the result in less than 1 second.

Post-Processing
===============

Unmasking
---------

After applying the ward, we must unmask the data. This can be done simply :

.. literalinclude:: ../plot_rest_clustering.py
        :start-after: # Unmask data
        :end-before: # Create a compressed picture

You can see that masked data is filled with -1 values. This is done for the
sake of visualisation. In fact, clusters are labelled with going from 0 to
(n_clusters - 1). By putting every other values to -1, we assure that
uninteresting values will not mess with the visualization.

Compressed picture
------------------

A compressed picture is a picture in which the value of each voxel is the
mean value of the cluster it belongs to. We can obtain this representation
thanks to a two step procedure :

- call *ward.transform* to obtain the mean value of each cluster (for each
  scan)
- call *ward.inverse_transform* on the previous result to turn it back into
  the masked picture shape

.. literalinclude:: ../plot_rest_clustering.py
        :start-after: # Create a compressed picture
        :end-before: ### Show result ###############################################################

Visualisation
=============

Then we can visualize the clusters. One color from the spectrum will be
attributed to each cluster for the labels visualisation and the compressed
picture is show in the classical gray colormap.

.. literalinclude:: ../plot_rest_clustering.py
        :start-after: ### Show result ###############################################################

.. figure:: auto_examples/images/plot_rest_clustering_1.png
   :target: auto_examples/plot_rest_clustering.html
   :align: center
