.. _searchlight:

===========================================================
Searchlight : finding voxels containing maximum information
===========================================================

.. currentmodule:: nisl.datasets

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

