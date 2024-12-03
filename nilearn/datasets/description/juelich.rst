.. _juelich_atlas:

Juelich atlas
=============

Access
------
See :func:`nilearn.datasets.fetch_atlas_juelich`.

Notes
-----
The Julich-Brain cytoarchitectonic atlas presents cytoarchitectonic maps in several coordinate spaces,
such as :term:`MNI` colin27, MNI152, and freesurfer.
These maps originate from peer-reviewed probability maps that define
both cortical and subcortical brain regions.
Notably, these probability maps account for the brain's inter-individual variability
by analyzing data from multiple post-mortem samples.
For a whole-brain parcellation, the available probability maps are combined
into a maximum probability map by considering
for each :term:`voxel` the probability of all cytoarchitectonic brain regions,
and determining the most probable assignment.

For more details: https://fsl.fmrib.ox.ac.uk/fsl/docs/#/other/datasets

Content
-------
    :'maps': nifti image containing regions or their probability
    :'labels': list of labels for the regions in the atlas.


References
----------
For the overall scientific concept and methodology of the Julich-Brain cytoarchitectonic atlas,
please cite :footcite:t:`Amunts2020`.

License
-------
See https://fsl.fmrib.ox.ac.uk/fsl/docs/#/license?id=fsl-license
