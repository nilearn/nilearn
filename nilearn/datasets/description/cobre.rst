COBRE


Notes
-----
This work is a derivative from the COBRE sample found in the [International Neuroimaging Data-sharing Initiative (INDI)](http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html), originally released under Creative Commons -- Attribution Non-Commercial. It includes preprocessed resting-state functional magnetic resonance images for 72 patients diagnosed with schizophrenia (58 males, age range = 18-65 yrs) and 74 healthy controls (51 males, age range = 18-65 yrs). The fMRI dataset for each subject are single nifti files (.nii.gz), featuring 150 EPI blood-oxygenation level dependent (BOLD) volumes were obtained in 5 mns (TR = 2 s, TE = 29 ms, FA = 75°, 32 slices, voxel size = 3x3x4 mm3, matrix size = 64x64, FOV = mm2).


Content
-------
    :'README.md': a markdown (text) description of the release
    :'phenotypic_data.tsv.gz': A gzipped tabular-separated value file, with each column representing a phenotypic variable as well as measures of data quality (related to motions). Each row corresponds to one participant, except the first row which contains the names of the variables (see file below for a description).
    :'keys_phenotypic_data.json': a json file describing each variable found in 'phenotypic_data.tsv.gz'.
    :'fmri_XXXXXXX.tsv.gz': A gzipped tabular-separated value file, with each column representing a confounding variable for the time series of participant XXXXXXX (which is the same participant ID found in 'phenotypic_data.tsv.gz'). Each row corresponds to a time frame, except for the first row, which contains the names of the variables (see file below for a definition).
    :'keys_confounds.json': a json file describing each variable found in the files 'fmri_XXXXXXX.tsv.gz'.
    :'fmri_XXXXXXX.nii.gz': a 3D+t nifti volume at 6 mm isotropic resolution, stored as short (16 bits) integers, in the MNI non-linear 2009a symmetric space
(http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009). Each fMRI data features 150 volumes. 


Usage recommendations
---------------------
    * Individual analyses: You may want to remove some time frames with excessive motion for each subject, see the confounding variable called 'scrub' in 'fmri_XXXXXXX.tsv.gz'. Also, after removing these time frames there may not be enough usable data. We recommend a minimum number of 60 time frames. A fairly large number of confounds have been made available as part of the release (slow time drifts, motion paramaters, frame displacement, scrubbing, average WM/Vent signal, COMPCOR, global signal). We strongly recommend regression of slow time drifts. Everything else is optional. 
    * Group analyses: There will also be some residuals effect of motion, which you may want to regress out from connectivity measures at the group level. The number of acceptable time frames as well as a measure of residual motion (called frame displacement, as described by Power et al., Neuroimage 2012), can be found in the variables 'Frames OK' and 'FD scrubbed' in 'phenotypic_data.tsv.gz'. Finally, the simplest use case with these data is to predict the overall presence of a diagnosis of schizophrenia (values 'Control' or 'Patient' in the phenotypic variable 'Subject Type'). You may want to try to match the control and patient samples in terms of amounts of motion, as well as age and sex. Note that more detailed diagnostic categories are available in the variable 'Diagnosis'. 


Preprocessing
-------------
The datasets were analysed using the NeuroImaging Analysis Kit (NIAK https://github.com/SIMEXP/niak) version 0.17, under CentOS version 6.3 with Octave(http://gnu.octave.org) version 4.0.2 and the Minc toolkit (http://www.bic.mni.mcgill.ca/ServicesSoftware/ServicesSoftwareMincToolKit) version 0.3.18.
Each fMRI dataset was corrected for inter-slice difference in acquisition time and the parameters of a rigid-body motion were estimated for each time frame. Rigid-body motion was estimated within as well as between runs, using the median volume of the first run as a target. The median volume of one selected fMRI run for each subject was coregistered with a T1 individual scan using Minctracc (Collins and Evans, 1998), which was itself non-linearly transformed to the Montreal Neurological Institute (MNI) template (Fonov et al., 2011) using the CIVET pipeline (Ad-Dabbagh et al., 2006). The MNI  symmetric template was generated from the ICBM152 sample of 152 young adults, after 40 iterations of non-linear coregistration. The rigid-body
transform, fMRI-to-T1 transform and T1-to-stereotaxic transform were all combined, and the functional volumes were resampled in the MNI space at a 6 mm isotropic resolution. 

Note that a number of confounding variables were estimated and are made available as part of the release. WARNING: no confounds were actually regressed from the data, so it can be done interactively by the user who will be able to explore different analytical paths easily. The “scrubbing” method of (Power et al., 2012), was used to identify the volumes with excessive motion (frame displacement greater than 0.5 mm). A minimum number of 60 unscrubbed volumes per run, corresponding to ~180 s of acquisition, is recommended for further analysis. The following nuisance parameters were estimated: slow time drifts (basis of discrete cosines with a 0.01 Hz high-pass cut-off), average signals in conservative masks of the white matter and the lateral ventricles as well as the six rigid-body motion parameters (Giove et al., 2009), anatomical COMPCOR signal in the ventricles and cerebrospinal fluid (Chai et al., 2012), PCA-based estimator of the global signal (Carbonell et al., 2011). The fMRI volumes were not spatially smoothed.


References
----------
Ad-Dab’bagh, Y., Einarson, D., Lyttelton, O., Muehlboeck, J. S., Mok, K., Ivanov, O., Vincent, R. D., Lepage, C., Lerch, J., Fombonne, E., Evans, A. C., 2006. The CIVET Image-Processing Environment: A Fully Automated Comprehensive Pipeline for Anatomical Neuroimaging Research. In: Corbetta, M. (Ed.), Proceedings of the 12th Annual Meeting of the Human Brain Mapping Organization. Neuroimage, Florence, Italy.

Bellec, P., Rosa-Neto, P., Lyttelton, O. C., Benali, H., Evans, A. C., Jul. 2010. Multi-level bootstrap analysis of stable clusters in resting-state fMRI. NeuroImage 51 (3), 1126–1139. URL http://dx.doi.org/10.1016/j.neuroimage.2010.02.082

F. Carbonell, P. Bellec, A. Shmuel. Validation of a superposition model of global and system-specific resting state activity reveals anti-correlated networks. Brain Connectivity 2011 1(6): 496-510. doi:10.1089/brain.2011.0065

Chai, X. J., Castan, A. N. N., Ongr, D., Whitfield-Gabrieli, S., Jan. 2012. Anticorrelations in resting state networks without global signal regression. NeuroImage 59 (2), 1420-1428. http://dx.doi.org/10.1016/j.neuroimage.2011.08.048
Collins, D. L., Evans, A. C., 1997. Animal: validation and applications of nonlinear registration-based segmentation. International Journal of Pattern Recognition and Artificial Intelligence 11, 1271–1294.

Fonov, V., Evans, A. C., Botteron, K., Almli, C. R., McKinstry, R. C., Collins, D. L., Jan. 2011. Unbiased average age-appropriate atlases for pediatric studies. NeuroImage 54 (1), 313–327.
URL http://dx.doi.org/10.1016/j.neuroimage.2010.07.033

Giove, F., Gili, T., Iacovella, V., Macaluso, E., Maraviglia, B., Oct. 2009. Images-based suppression of unwanted global signals in resting-state functional connectivity studies. Magnetic resonance imaging 27 (8), 1058–1064. URL http://dx.doi.org/10.1016/j.mri.2009.06.004

Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L., Petersen, S. E., Feb. 2012. Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. NeuroImage 59 (3), 2142–2154. URL http://dx.doi.org/10.1016/j.neuroimage.2011.10.018
