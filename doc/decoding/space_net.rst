.. _space_net:

==========================================================
SpaceNet: decoding with spatial structure for better maps
==========================================================

The SpaceNet decoder
=====================

:class:`nilearn.decoding.SpaceNetRegressor` and :class:`nilearn.decoding.SpaceNetClassifier`
implements spatial penalties which improve brain decoding power as well as decoder maps:

* penalty="tvl1": priors inspired from TV (Total Variation, see :footcite:t:`Michel2011`),
  TV-L1 (see :footcite:t:`Baldassarre2012` and :footcite:t:`Gramfort2013`).

* penalty="graph-net": GraphNet prior (see :footcite:t:`Grosenick2013`).

These regularize :term:`classification` and :term:`regression`
problems in brain imaging. The results are brain maps which are both
sparse (i.e regression coefficients are zero everywhere, except at
predictive :term:`voxels<voxel>`) and structured (blobby). The superiority of TV-L1
over methods without structured priors like the Lasso, :term:`SVM`, :term:`ANOVA`,
Ridge, etc. for yielding more interpretable maps and improved
prediction scores is now well established (see :footcite:t:`Baldassarre2012`,
:footcite:t:`Gramfort2013` :footcite:t:`Grosenick2013`).

Note that TV-L1 prior leads to a difficult optimization problem, and so can be slow to run.
Under the hood, a few heuristics are used to make things a bit faster. These include:

- Feature preprocessing, where an F-test is used to eliminate
  non-predictive :term:`voxels<voxel>`, thus reducing the size of the brain
  mask in a principled way.
- Continuation is used along the regularization path, where the
  solution of the optimization problem for a given value of the
  regularization parameter ``alpha`` is used as initialization
  for the next regularization (smaller) value on the regularization
  grid.

**Implementation:** See :footcite:t:`Dohmatob2015` and :footcite:t:`Dohmatob2014`
for technical details regarding the implementation of SpaceNet.

Related example
===============

:ref:`Age prediction on OASIS dataset with SpaceNet <sphx_glr_auto_examples_02_decoding_plot_oasis_vbm_space_net.py>`.

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_oasis_vbm_space_net_002.png
   :align: center

.. note::

    Empirical comparisons using this method have been removed from
    documentation in version 0.7 to keep its computational cost low. You can
    easily try SpaceNet instead of :term:`FREM` in :ref:`mixed gambles study <sphx_glr_auto_examples_02_decoding_plot_mixed_gambles_frem.py>` or :ref:`Haxby study <sphx_glr_auto_examples_02_decoding_plot_haxby_frem.py>`.

.. seealso::

    :ref:`FREM <frem>`, a pipeline ensembling many models that yields very
    good decoding performance at a lower computational cost.

References
==========

.. footbibliography::
