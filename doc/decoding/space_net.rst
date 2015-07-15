.. for doctests to run, we need to define variables that are define in
   the literal includes
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> fmri_masked  = iris.data
    >>> target = iris.target
    >>> session = np.ones_like(target)
    >>> n_samples = len(target)

.. _space_net.rst:

======================================
Multi-variate decoding with SpaceNet
======================================

The SpaceNet decoder
--------------------
SpaceNet implements a suite of multi-variate priors which for improved
brain decoding. It uses priors like TV (Total Variation) `[Michel et
al. 2011] <https://hal.inria.fr/inria-00563468/document>`_, TV-L1
`[Baldassarre et al. 2012]
<http://www0.cs.ucl.ac.uk/staff/M.Pontil/reading/neurosparse_prni.pdf>`_,
`[Gramfort et al. 2013] <https://hal.inria.fr/hal-00839984>`_
(option: penalty="tvl1"), and Smooth-Lasso `[Hebiri et al. 2011]
<https://hal.archives-ouvertes.fr/hal-00462882/document>`_ (known
as GraphNet in neuroimaging `[Grosenick et al. 2013]
<https://hal.inria.fr/hal-00839984>`_) (option:
penalty="smooth-lasso") to regularize classification and regression
problems in brain imaging. The result are brain maps which are both
sparse (i.e regression coefficients are zero everywhere, except at
predictive voxels) and structured (blobby). The superiority of TV-L1
over methods without structured priors like the Lasso, SVM, ANOVA,
Ridge, etc. for yielding more interpretable maps and improved
prediction scores is now well established `[Baldassarre et al. 2012]
<http://www0.cs.ucl.ac.uk/staff/M.Pontil/reading/neurosparse_prni.pdf>`_,
`[Gramfort et al. 2013] <https://hal.inria.fr/hal-00839984>`_,
`[Grosenick et al. 2013] <https://hal.inria.fr/hal-00839984>`_.


The following table summarizes the parameter(s) used to activate a given penalty:

- TV-L1: `penalty="tv-l1"`
- Smooth-Lasso: `penalty="smooth-lasso"` (this is the default prior in SpaceNet)
- TV: `l1_ratio=0`
- Lasso: `l1_ratio=1`

Note that TV-L1 prior leads to a hard optimization problem, and so can
be slow to run. Under the hood, a few heuristics are used to make
things a bit faster. These include:

- Feature preprocessing, where an F-test is used to eliminate
  non-predictive voxels, thus reducting the size of the brain mask in
  a principled way.
- Continuation is used along the regularization path, where the
  solution of the optimization problem for a given value of the
  regularization parameter `alpha` is used as initialization
  of for next the regularization (smaller) value on the regularization
  grid.

**Implementation:** See `[Dohmatob et al. 2015 (PRNI)]
<https://hal.inria.fr/hal-01147731>`_ and  `[Dohmatob
et al. 2015 (PRNI)] <https://hal.inria.fr/hal-00991743>`_ for
technical details regarding the implementation of SpaceNet.

..
   Below are some application examples of SpaceNet.

Mixed gambles
.............

.. figure:: ../auto_examples/decoding/images/plot_poldrack_space_net_001.png
   :align: right
   :scale: 60

.. figure:: ../auto_examples/decoding/images/plot_poldrack_space_net_002.png
   :scale: 60

.. literalinclude:: ../../examples/decoding/plot_poldrack_space_net.py


Haxby
.....

.. figure:: ../auto_examples/decoding/images/plot_haxby_space_net_001.png
   :align: right
   :scale: 60

.. figure:: ../auto_examples/decoding/images/plot_haxby_space_net_002.png
   :scale: 60

See the script here:
:doc:`plot_haxby_space_net.py <../auto_examples/decoding/plot_haxby_space_net>`

Oasis VBM
.........

.. figure:: ../auto_examples/decoding/images/plot_oasis_vbm_space_net_001.png
   :align: right
   :scale: 60

.. figure:: ../auto_examples/decoding/images/plot_oasis_vbm_space_net_002.png
   :scale: 60

See the script here:
:doc:`plot_haxby_space_net.py <../auto_examples/decoding/plot_haxby_space_net>`


..
   .. seealso::

     * The :ref:`supervised_learning` section of the nilearn documentation.

     * The `scikit-learn documentation <http://scikit-learn.org>`_
       has very detailed explanations on a large variety of estimators and
       machine learning techniques. To become better at decoding, you need
       to study it.
