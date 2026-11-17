.. _seitzman_2018_atlas:

Seitzman 2018 atlas
===================

Access
------
See :func:`nilearn.datasets.fetch_coords_seitzman_2018`.

Notes
-----
300 regions coordinates in cortical, subcortical and cerebellar regions.

These regions cover cortical, subcortical and cerebellar regions and are
assigned to one of 13 networks:
(Auditory, CinguloOpercular, DefaultMode,
DorsalAttention, FrontoParietal, MedialTemporalLobe, ParietoMedial, Reward, Salience, SomatomotorDorsal, SomatomotorLateral, VentralAttention, Visual)
and have a regional label
(cortexL, cortexR, cerebellum, thalamus, hippocampus, basalGanglia, amygdala, cortexMid).

Here, we apply a winner-take-all partitioning method
to :term:`resting-state` :term:`fMRI` data
and careful consideration of anatomy to generate novel functionally-constrained regions
in the thalamus, basal ganglia, amygdala, hippocampus, and cerebellum.
We validate these regions in three datasets via several anatomical and functional criteria,
including known anatomical divisions and functions,
as well as agreement with existing literature.
Further, we demonstrate that combining these regions with established cortical regions recapitulates
and extends previously described functional network organization.

See :footcite:t:`Seitzman2020`.

For more information see:
https://greenelab.ucsd.edu/data_software

ROI coordinates downloaded from:
https://wustl.app.box.com/s/twpyb1pflj6vrlxgh3rohyqanxbdpelw

Content
-------
.. nilearn_dataset_content:: seitzman_2018_atlas

References
----------
.. footbibliography::

License
-------
.. nilearn_dataset_license:: seitzman_2018_atlas
