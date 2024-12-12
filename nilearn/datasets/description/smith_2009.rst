Smith 2009 atlas
================

Access
------
See :func:`nilearn.datasets.fetch_atlas_smith_2009`.

Notes
-----
This atlas provides spatial maps of the major brain networks
during task-constrained brain activity
and task-unconstrained (resting) brain activity.

Those were derived from 6 minutes of :term:`resting-state` time series
from 36 subjects as well as from the from the smoothed task activity coordinates
of healthy subjects stored in the BrainMap database.

See :footcite:t:`Smith2009b` and :footcite:t:`Laird2011`.

Content
-------
    :'rsn20': 20 :term:`ICA` maps derived from :term:`resting-state` decomposition
    :'rsn10': 10 :term:`ICA` maps from the above that matched across task and rest
    :'rsn70': 70 :term:`ICA` maps derived from :term:`resting-state` decomposition
    :'bm20': 20 :term:`ICA` maps derived from decomposition BrainMap task data
    :'bm10': 10 :term:`ICA` maps from the above that matched across task and rest
    :'bm70': 70 :term:`ICA` maps derived from decomposition BrainMap task data


References
----------

.. footbibliography::

For more information about this dataset's structure:
https://www.fmrib.ox.ac.uk/datasets/brainmap+rsns/


License
-------
unknown
