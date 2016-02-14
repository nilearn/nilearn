COBRE datasets preprocessed using NIAK 0.12.4 version pipeline


Content
-------
This work is a derivative from the COBRE sample found in the [International
Neuroimaging Data-sharing Initiative
(INDI)](http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html), originally
released under Creative Commons -- Attribution Non-Commercial. It includes
preprocessed resting-state functional magnetic resonance images for 72
patients diagnosed with schizophrenia (58 males, age range = 18-65 yrs) and 74
healthy controls (51 males, age range = 18-65 yrs). The fMRI dataset for each
subject are single nifti files (.nii.gz), featuring 150 EPI blood-oxygenation
level dependent (BOLD) volumes were obtained in 5 mns (TR = 2 s, TE = 29 ms,
FA = 75 degrees, 32 slices, voxel size = 3x3x4 mm3 , matrix size = 64x64, FOV = mm2).


The COBRE preprocessed fMRI release more specifically contains the following
files:
    :'description': a markdown (text) description of the release.
    :'phenotypic': numpy array
    contains a comma-separated values, with the sz (1: patient with
    schizophrenia, 0: control), age, sex, and FD (frame displacement,
    as defined by Power et al. 2012) variables. Each column codes for
    one variable, starting with the label, and each line has the label of the
    corresponding subject.
    :'func': contains list of filenames to functional datasets
    fmri_szxxxSUBJECT_session1_run1.nii.gz, a 3D+t nifti volume at 3 mm
    isotropic resolution, in the MNI non-linear 2009a symmetric space
    (http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009).
    Note that the number of time samples may vary, as some samples have been
    removed if tagged with excessive motion. See the _extra.mat for more info.
    :'mat_files': contains list of filenames to .mat files
    fmri_szxxxSUBJECT_session1_run1_extra.mat, a matlab/octave file for each
    subject.

    Each .mat file contains the following variables:
    * confounds: a TxK array. Each row corresponds to a time sample, and each
      column to one confound that was regressed out from the time series
      during preprocessing.
    * labels_confounds: cell of strings. Each entry is the label of a
      confound that was regressed out from the time series.
    * mask_suppressed: a T2x1 vector. T2 is the number of time samples in
      the raw time series (before preprocessing), T2=119. Each entry
      corresponds to a time sample, and is 1 if the corresponding sample
      was removed due to excessive motion (or to wait for magnetic
      equilibrium at the beginning of the series). Samples that were kept
      are tagged with 0s.
    * time_frames: a Tx1 vector. Each entry is the time of acquisition
      (in s) of the corresponding volume.


Preprocessing
-------------
The datasets were analysed using the NeuroImaging Analysis Kit (NIAK
https://github.com/SIMEXP/niak) version 0.12.14, under CentOS version 6.3 with
Octave(http://gnu.octave.org) version 3.8.1 and the Minc toolkit
(http://www.bic.mni.mcgill.ca/ServicesSoftware/ServicesSoftwareMincToolKit)
version 0.3.18.
Each fMRI dataset was corrected for inter-slice difference in acquisition time
and the parameters of a rigid-body motion were estimated for each time frame.
Rigid-body motion was estimated within as well as between runs, using the
median volume of the first run as a target. The median volume of one selected
fMRI run for each subject was coregistered with a T1 individual scan using
Minctracc (Collins and Evans, 1998), which was itself non-linearly transformed
to the Montreal Neurological Institute (MNI) template (Fonov et al., 2011)
using the CIVET pipeline (Ad-Dabbagh et al., 2006). The MNI  symmetric
template was generated from the ICBM152 sample of 152 young adults, after 40
iterations of non-linear coregistration. The rigid-body
transform, fMRI-to-T1 transform and T1-to-stereotaxic transform were all
combined, and the functional volumes were resampled in the MNI space at a 3 mm
isotropic resolution. The "scrubbing" method of (Power et al., 2012), was used
to remove the volumes with excessive motion (frame displacement greater than
0.5 mm). A minimum number of 60 unscrubbed volumes per run, corresponding to
~180 s of acquisition, was then required for further analysis. For this
reason, 16 controls and 29 schizophrenia patients were rejected from the
subsequent analyses. The following nuisance parameters were regressed out from
the time series at each voxel: slow time drifts (basis of discrete cosines
with a 0.01 Hz high-pass cut-off), average signals in conservative masks of
the white matter and the lateral ventricles as well as the first principal
components (95% energy) of the six rigid-body motion parameters and their
squares (Giove et al., 2009). The fMRI volumes were finally spatially smoothed
with a 6 mm isotropic Gaussian blurring kernel.


References
----------
Ad-Dab'bagh Y, Einarson D, Lyttelton O, Muehlboeck J S, Mok K, Ivanov O,
Vincent R D, Lepage C, Lerch J, Fombonne E, Evans A C, 2006.
The CIVET Image-Processing Environment: A Fully Automated Comprehensive
Pipeline for Anatomical Neuroimaging Research. In: Corbetta M. (Ed.),
Proceedings of the 12th Annual Meeting of the Human Brain Mapping
Organization. Neuroimage, Florence, Italy.

Bellec P, Rosa-Neto P, Lyttelton O C, Benali H, Evans A C, Jul. 2010.
Multi-level bootstrap analysis of stable clusters in resting-state fMRI.
NeuroImage 51 (3), 1126–1139.
URL http://dx.doi.org/10.1016/j.neuroimage.2010.02.082

Collins D L, Evans A C, 1997. Animal: validation and applications of
nonlinear registration-based segmentation. International Journal of Pattern
Recognition and Artificial Intelligence 11, 1271-1294.

Fonov V, Evans A C, Botteron K, Almli C R, McKinstry R C, Collins D L,
Jan. 2011. Unbiased average age-appropriate atlases for pediatric studies.
NeuroImage 54 (1), 313-327.
URL http://dx.doi.org/10.1016/j.neuroimage.2010.07.033

Giove F, Gili T, Iacovella V, Macaluso E, Maraviglia B, Oct. 2009.
Images-based suppression of unwanted global signals in resting-state
functional connectivity studies. Magnetic resonance imaging 27 (8), 1058-1064.
URL http://dx.doi.org/10.1016/j.mri.2009.06.004

Power J D, Barnes K A, Snyder A Z, Schlaggar B L, Petersen S E, Feb. 2012.
Spurious but systematic correlations in functional connectivity MRI
networks arise from subject motion. NeuroImage 59 (3), 2142-2154.
URL http://dx.doi.org/10.1016/j.neuroimage.2011.10.018


Other derivatives
-----------------
This dataset was used in a publication, see the link below.
https://github.com/SIMEXP/glm_connectome
