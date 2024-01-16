Juelich


Notes
-----
The Julich-Brain Cytoarchitectonic Atlas presents cytoarchitectonic maps in several coordinate spaces,
such as MNI colin27, MNI152, and freesurfer.
These maps originate from peer-reviewed probability maps that define
both cortical and subcortical brain regions.
Notably, these probability maps account for the brain's inter-individual variability
by analyzing data from multiple post-mortem samples.
For a whole-brain parcellation, the available probability maps are combined
into a maximum probability map by considering for each voxel the probability of all cytoarchitectonic brain regions,
and determining the most probable assignment.

For more details: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases

Content
-------
    :'maps': nifti image containing regions or their probability
    :'labels': list of labels for the regions in the atlas.


References
----------
For the overall scientific concept and methodology of the Julich-Brain Cytoarchitectonic Atlas, please cite:

Amunts, K., Mohlberg, H., Bludau, S., & Zilles, K. (2020).
Julich-Brain: A 3D probabilistic atlas of the human brain's cytoarchitecture.
Science, 369(6506), 988-992. DOI: 10.1126/science.abb4588


License
-------
See https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Licence
