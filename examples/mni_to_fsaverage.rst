.. admonition:: Projecting volume to the fsaverage space
   :class: attention

   In this example, we project volume data registered to MNI space
   onto surface meshes in the Freesurfer fsaverage space.
   This can lead to inaccuracies (:footcite:t:`Wu2018`)
   and volume data should ideally first be refgistered to fsaverage
   before projection.
   This can be done with the function
   `neuromaps.transforms.mni152_to_fsaverage <https://netneurolab.github.io/neuromaps/generated/neuromaps.transforms.mni152_to_fsaverage.html>`_
   from the `neuromaps package <https://netneurolab.github.io/neuromaps/index.html>`_.
