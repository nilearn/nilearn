SPM multimodal


Notes
-----
The example shows the analysis of an :term:`SPM` dataset studying face perception.
The analysis is performed in native space.
Realignment parameters are provided with the input images,
but those have not been resampled to a common space.

The experimental paradigm is simple, with two conditions:
viewing a face image or a scrambled face image,
supposedly with the same low-level statistical properties,
to find face-specific responses.

Content
-------
    :'func1': Paths to functional images for run 1
    :'func2': Paths to functional images for run 2
    :'trials_ses1': Path to onsets file for run 1
    :'trials_ses2': Path to onsets file for run 2
    :'anat': Path to anat file


References
----------
See :cite:`spm_multiface`.

For details on the data, please see:

Henson, R.N., Goshen-Gottstein, Y., Ganel, T., Otten, L.J., Quayle, A.,
Rugg, M.D. Electrophysiological and haemodynamic correlates of face
perception, recognition and priming. Cereb Cortex. 2003 Jul;13(7):793-805.
https://doi.org/10.1093/cercor/13.7.793

License
-------
