.. _searchlight:

===========================================================
Searchlight : finding voxels containing maximum information
===========================================================

.. currentmodule:: nisl.searchlight

Searchlight principle
=====================

Searchlight was introduced in `Information-based functional brain mapping
<http://www.pnas.org/content/103/10/3863>`_, Nikolaus Kriegeskorte,
Rainer Goebel and Peter Bandettini (PNAS 2006) and consists in scanning the
images volume with a *searchlight*. It can basically be related to mathematical
morphology in the sense that a structuring element will be used to iterate over
a 3D volume. But, instead of applying a mathematical operator (like minimum for
erosion or maximum for dilatation), we will run a classifier on the *floodlit*
voxels and score them (using cross validation). This gives us an
*information-based functional brain mapping*.

First Step : loading Haxby dataset
==================================

Haxby dataset is a visual task dataset where several objects are show to the
volunteer. We will keep only face and houses in order to speed up computing.
For more details on Haxby, please see haxby_decoding.

Second Step : masking the data
==============================

One of the main element that distinguich Searchlight from other algorithms is
this notion of structuring element that scan the entire volume. If this seems
rather intuitive, it has in fact an impact on the masking procedure.

Masking is a commonly used technique to restrain the computation area. For
example, one may want to consider only some regions of interest or only the
back of the brain, in the case of a visual task.

Most of the time, fMRI data is masked and then given to the algorithm. This is
not possible in the case of Searchlight because, to compute the score of
non-masked voxels, some masked voxels may be needed. This is why two masks will
be used here :

- *mask* is the anatomical mask
- *process_mask* is a subset of mask and contains voxels to be processed.

*process_mask* will then be used to restrain computation to one slice, in the
back of the brain. *mask* will ensure that no value outside of the brain is
taken into account when iterating with the sphere.

.. literalinclude:: ../plot_haxby_searchlight.py
        :start-after:#   up computation)
        :end-before:### Restrict to faces and houses ##############################################

Third Step : Set up the cross validation
========================================

Searchlight will iterate on the volume and give a score to each voxel. This
score is computed by running a classifier on selected voxels. In order to make
this score as accurate as possible (and avoid overfitting), a cross validation
is made.

Classifier
----------

The classifier used by default by Searchlight is LinearSVC with C=1 but this
can be customed easily by passing an estimator parameter to the cross
validation.

See scikit-learn documentation for
`other classifiers <http://scikit-learn.org/supervised_learning.html>`_

Score function
--------------

Let *face* and *house* be our two classes. The classifier will learn to guess
which picture is presented just by looking at the fMRI. It will then be tested
thanks to a testing set. There are several ways to compute a score out of the
results. Here we have choosed the precision that measures proportion of true
positives among all positives results for one class.

Again, many others are available in 
`scikit-learn documentation
<http://scikit-learn.org/supervised_learning.html>`_

Cross validation
----------------

Cross validation is a process through which a set of samples is divided in two
sets, training and testing sets, to score a classifier. Several strategies can
be adopted : here we use the *K*-Fold that randomly divides the sample set in
*k* sets and makes *k* iterations so that a set is used as testing set and the
other ones as training sets.

Fourth Step : Running Searchlight
=================================

Running Searchlight is straightforward now that everything is set. The only
parameter left is the radius of the ball that will run through the data.
Kriegskorte uses a 4mm radius because it yielded the best detection
performance in his simulation.

Visualisation
=============

As the activation map is cropped, we use the mean image of all scans as a
background. We can see here that voxels in the visual cortex contains
information to distinguish pictures showed to the volunteer, which was the
expected result.
