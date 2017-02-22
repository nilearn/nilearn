COBRE


Notes
-----
This work is a derivative from the COBRE sample found in the International
Neuroimaging Data-sharing Initiative.
(http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html), originally
released under Creative Commons - Attribution Non-Commercial.
It includes preprocessed resting - state functional magnetic resonance images
for 72 patients diagnosed with schizophrenia and 74 healthy controls.

Content
-------
    :'phenotypic_data.tsv.gz': A gzipped tabular-separated value file,
    with each column representing a phenotypic variable as well as measures
    of data quality, related to motions. Each row corresponds to one
    participant, except the first row which contains the names of the
    variables.
    :'keys_phenotypic_data.json': a json file describing each variable found
    in 'phenotypic_data.tsv.gz'.
    :'fmri_XXXXXXX.tsv.gz': A gzipped tabular-separated value file, with each
    column representing a confounding variable for the time series of
    participant XXXXXXX, which is the same participant ID found in
    'phenotypic_data.tsv.gz'. Each row corresponds to a time frame, except for
    the first row, which contains the names of the variables.
    :'keys_confounds.json': a json file describing each variable found in the
    files 'fmri_XXXXXXX.tsv.gz'.
    :'fmri_XXXXXXX.nii.gz': a 3D + t nifti volume at 6 mm isotropic resolution.
    Each fMRI data features 150 volumes.


Usage recommendations
---------------------
Individual analyses: You may want to remove some time frames with excessive
motion for each subject, see the confounding variable called 'scrub' in
'fmri_XXXXXXX.tsv.gz'. Also, after removing these time frames there may not be
enough usable data. We recommend a minimum number of 60 time frames. A fairly
large number of confounds have been made available as part of the release: slow
time drifts, motion paramaters, frame displacement, scrubbing, average WM/Vent
signal, COMPCOR, global signal.
We strongly recommend regression of slow time drifts.
Everything else is optional.

Group analyses: There will also be some residuals effect of motion, which you
may want to regress out from connectivity measures at the group level. The
number of acceptable time frames as well as a measure of residual motion, can
be found in the variables 'Frames OK' and 'FD scrubbed' in
'phenotypic_data.tsv.gz'. Finally, the simplest use case with these data is to
predict the overall presence of a diagnosis of schizophrenia (values 'Control'
or 'Patient' in the phenotypic variable 'Subject Type').


Preprocessing
-------------
The datasets were analysed using the NeuroImaging Analysis Kit (NIAK
https://github.com/SIMEXP/niak) version 0.17, under CentOS version 6.3 with
Octave(http://gnu.octave.org) version 4.0.2 and the Minc toolkit
(http://www.bic.mni.mcgill.ca/ServicesSoftware/ServicesSoftwareMincToolKit)
version 0.3.18.

Note that a number of confounding variables were estimated and are made
available as part of the release.

WARNING: no confounds were actually regressed from the data, so it can be done
interactively by the user who will be able to explore different analytical
paths easily.


References
----------
Ad-Dab’bagh, et. al., 2006. The CIVET Image-Processing Environment: A Fully
Automated Comprehensive Pipeline for Anatomical Neuroimaging Research. In:
Corbetta, M. (Ed.), Proceedings of the 12th Annual Meeting of the Human Brain
Mapping Organization. Neuroimage, Florence, Italy.

Bellec, P., et. al., 2010. Multi-level bootstrap analysis of stable clusters in
resting-state fMRI. NeuroImage 51 (3), 1126-1139.

F. Carbonell, P. Bellec, A. Shmuel. Validation of a superposition model of
global and system-specific resting state activity reveals anti-correlated
networks. Brain Connectivity 2011 1(6): 496-510.

Chai, X. J., et. al., 2012. Anticorrelations in resting state networks without
global signal regression. NeuroImage 59 (2), 1420-1428.

Collins, D. L., Evans, A. C., 1997. Animal: validation and applications of
nonlinear registration-based segmentation. International Journal of Pattern
Recognition and Artificial Intelligence 11, 1271-1294.

Fonov, V., et. al., 2011. Unbiased average age-appropriate atlases for
pediatric studies. NeuroImage 54 (1), 313-327.

Giove, F., et. al., 2009. Images-based suppression of unwanted global signals
in resting-state functional connectivity studies. Magnetic resonance imaging
27 (8), 1058-1064.

Power, J. D., et. al., 2012. Spurious but systematic correlations in functional
connectivity MRI networks arise from subject motion. NeuroImage 59 (3),
2142-2154.
