.. for doctests to run, we need to define variables that are define in
   the literal includes
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> fmri_masked  = iris.data
    >>> target = iris.target
    >>> session = np.ones_like(target)
    >>> n_samples = len(target)

.. _space_net:

==========================================================
SpaceNet: decoding with spatial structure for better maps
==========================================================

The SpaceNet decoder
=====================

SpaceNet implements spatial penalties which improve brain decoding power as well as decoder maps:

* penalty="tvl1": priors inspired from TV (Total Variation) `[Michel et
  al. 2011] <https://hal.inria.fr/inria-00563468/document>`_, TV-L1
  `[Baldassarre et al. 2012]
  <http://www0.cs.ucl.ac.uk/staff/M.Pontil/reading/neurosparse_prni.pdf>`_,
  `[Gramfort et al. 2013] <https://hal.inria.fr/hal-00839984>`_ (option: ),

* penalty="graph-net": GraphNet prior `[Grosenick et al. 2013]
  <https://www.ncbi.nlm.nih.gov/pubmed/23298747>`_)

These regularize classification and regression
problems in brain imaging. The results are brain maps which are both
sparse (i.e regression coefficients are zero everywhere, except at
predictive voxels) and structured (blobby). The superiority of TV-L1
over methods without structured priors like the Lasso, SVM, ANOVA,
Ridge, etc. for yielding more interpretable maps and improved
prediction scores is now well established `[Baldassarre et al. 2012]
<http://www0.cs.ucl.ac.uk/staff/M.Pontil/reading/neurosparse_prni.pdf>`_,
`[Gramfort et al. 2013] <https://hal.inria.fr/hal-00839984>`_,
`[Grosenick et al. 2013] <https://www.ncbi.nlm.nih.gov/pubmed/23298747>`_.


Note that TV-L1 prior leads to a difficult optimization problem, and so
can be slow to run. Under the hood, a few heuristics are used to make
things a bit faster. These include:

- Feature preprocessing, where an F-test is used to eliminate
  non-predictive voxels, thus reducing the size of the brain mask in
  a principled way.
- Continuation is used along the regularization path, where the
  solution of the optimization problem for a given value of the
  regularization parameter `alpha` is used as initialization
  for the next regularization (smaller) value on the regularization
  grid.

**Implementation:** See `[Dohmatob et al. 2015 (PRNI)]
<https://hal.inria.fr/hal-01147731>`_ and  `[Dohmatob
et al. 2014 (PRNI)] <https://hal.inria.fr/hal-00991743>`_ for
technical details regarding the implementation of SpaceNet.

Empirical comparisons
=====================


Comparison on mixed gambles study
----------------------------------

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_mixed_gambles_space_net_001.png
   :align: right
   :scale: 60

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_mixed_gambles_space_net_002.png
   :scale: 60

.. topic:: **Code**

    The complete script can be found
    :ref:`here <sphx_glr_auto_examples_02_decoding_plot_mixed_gambles_space_net.py>`.


Comparison on Haxby study
--------------------------

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_space_net_001.png
   :align: right
   :scale: 60

.. figure:: ../auto_examples/02_decoding/images/sphx_glr_plot_haxby_space_net_002.png
   :scale: 60

.. topic:: **Code**

    The complete script can be found
    :ref:`here <sphx_glr_auto_examples_02_decoding_plot_haxby_space_net.py>`.

.. seealso::

     * :ref:`Age prediction on OASIS dataset with SpaceNet <sphx_glr_auto_examples_02_decoding_plot_oasis_vbm_space_net.py>`.

     * The `scikit-learn documentation <http://scikit-learn.org>`_
       has very detailed explanations on a large variety of estimators and
       machine learning techniques. To become better at decoding, you need
       to study it.
