An atlas of multiscale brain parcellations


Content
-------
This work is a derivative from the Cambridge sample found in the [1000
functional connectome project]
(http://fcon_1000.projects.nitrc.org/fcpClassic/FcpTable.html) (Liu et
al., 2009), originally released under Creative Commons -- Attribution
Non-Commercial. It includes group brain parcellations generated from
resting-state functional magnetic resonance images for about 200 young
healthy subjects. Multiple scales (number of networks) are available,
and includes 7, 12, 20, 36, 64, 122, 197, 325, 444. The brain parcellations
have been generated using a method called bootstrap analysis of stable clusters
(BASC, Bellec et al., 2010) and the scales have been selected using a data-driven
method called MSTEPS (Bellec, 2013).


This release more specifically contains the following files:
    :'description': a markdown (text) description of the release.
    :'scale007', 'scale012', 'scale020', 'scale036', 'scale064',
     'scale122', 'scale197', 'scale325', 'scale444'
brain_parcellation_cambridge_basc_multiscale_(sym,asym)_scale(NNN).nii.gz:
a 3D volume .nii format at 3 mm isotropic resolution, in the MNI non-linear
2009a space (http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009).
Region number I is filled with Is (background is filled with 0s).


Note that two versions of the template are available, ending with either
nii_sym or nii_asym. The asym flavor contains brain images that have been
registered in the asymmetric version of the MNI brain template (reflecting
that the brain is asymmetric), while with the sym flavor they have been
registered in the symmetric version of the MNI template. The symmetric
template has been forced to be symmetric anatomically, and is therefore
ideally suited to study homotopic functional connections in fMRI: finding
homotopic regions simply consists of flipping the x-axis of the template. 


Preprocessing
-------------
The datasets were analysed using the NeuroImaging Analysis Kit (NIAK
https://github.com/SIMEXP/niak) version 0.12.14, under CentOS version 6.3 with
Octave (http://gnu.octave.org) version 3.8.1 and the Minc toolkit
(http://www.bic.mni.mcgill.ca/ServicesSoftware/ServicesSoftwareMincToolKit)
version 0.3.18.
Each fMRI dataset was corrected for inter-slice difference in acquisition time
and the parameters of a rigid-body motion were estimated for each time frame.
Rigid-body motion was estimated within as well as between runs, using the
median volume of the first run as a target. The median volume of one selected
fMRI run for each subject was coregistered with a T1 individual scan using
Minctracc (Collins and Evans, 1998), which was itself non-linearly transformed
to the Montreal Neurological Institute (MNI) template (Fonov et al., 2011)
using the CIVET pipeline (Ad-Dabbagh et al., 2006). The MNI symmetric template
was generated from the ICBM152 sample of 152 young adults, after 40 iterations
of non-linear coregistration. The rigid-body transform, fMRI-to-T1 transform
and T1-to-stereotaxic transform were all combined, and the functional volumes
were resampled in the MNI space at a 3 mm isotropic resolution. The
"scrubbing" method of (Power et al., 2012), was used to remove the volumes
with excessive motion (frame displacement greater than 0.5 mm). A minimum
number of 60 unscrubbed volumes per run, corresponding to ~180 s of
acquisition, was then required for further analysis. The following nuisance
parameters were regressed out from the time series at each voxel: slow time
drifts (basis of discrete cosines with a 0.01 Hz high-pass cut-off), average
signals in conservative masks of the white matter and the lateral ventricles
as well as the first principal components (95% energy) of the
six rigid-body motion parameters and their squares (Giove et al., 2009). The
fMRI volumes were finally spatially smoothed with a 6 mm isotropic Gaussian
blurring kernel.


Bootstrap Analysis of Stable Clusters
-------------------------------------
Brain parcellations were derived using BASC (Bellec et al. 2010). A region
growing algorithm was first applied to reduce the brain into regions of
roughly equal size, set to 1000 mm3. The BASC used 100 replications of a
hierarchical clustering with Ward's criterion on resampled individual time
series, using circular block bootstrap. A consensus clustering (hierarchical
with Ward's criterion) was generated across all the individual clustering
replications pooled together, hence generating group clusters. The generation
of group clusters was itself replicated by bootstraping subjects 500 times,
and a (final) consensus clustering (hierarchical Ward's criterion) was
generated on the replicated group clusters. The MSTEPS procedure (Bellec et
al., 2013) was implemented to select a data-driven subset of scales in the
range 5-500, approximating the group stability matrices up to 5% residual
energy, through linear interpolation over selected scales. Note that the
number of scales itself was selected by the MSTEPS procedure in a data-driven
fashion, and that the number of individual, group and final (consensus) number
of clusters were not necessarily identical.


References
----------
Ad-Dabbagh Y, Einarson D, Lyttelton O, Muehlboeck J S, Mok K,
Ivanov O, Vincent R D, Lepage C, Lerch J, Fombonne E, Evans A C,
2006. The CIVET Image-Processing Environment: A Fully Automated
Comprehensive Pipeline for Anatomical Neuroimaging Research.
In: Corbetta, M. (Ed.), Proceedings of the 12th Annual Meeting
of the Human Brain Mapping Organization. Neuroimage, Florence, Italy.

Bellec P, Rosa-Neto P, Lyttelton O C, Benali H, Evans A C, Jul. 2010
Multi-level bootstrap analysis of stable clusters in resting-state fMRI.
NeuroImage 51 (3), 1126-1139.
URL http://dx.doi.org/10.1016/j.neuroimage.2010.02.082

Bellec P, Jun. 2013. Mining the Hierarchy of Resting-State Brain Networks:
Selection of Representative Clusters in a Multiscale Structure. In: Pattern
Recognition in Neuroimaging (PRNI), 2013 International Workshop on. pp.
54-57.

Collins D L, Evans A C, 1997. Animal: validation and applications of
nonlinear registration-based segmentation. International Journal of
Pattern Recognition and Artificial Intelligence 11, 1271-1294.

Fonov V, Evans A C, Botteron K, Almli C R, McKinstry, R C, Collins D L,
Jan. 2011. Unbiased average age-appropriate atlases for pediatric
studies. NeuroImage 54 (1), 313-327.
URL http://dx.doi.org/10.1016/j.neuroimage.2010.07.033

Giove F, Gili T, Iacovella V, Macaluso E, Maraviglia B, Oct. 2009.
Images-based suppression of unwanted global signals in resting-state
functional connectivity studies. Magnetic resonance imaging 27 (8), 1058-1064.
URL http://dx.doi.org/10.1016/j.mri.2009.06.004

Liu H, Stufflebeam S M, Sepulcre J, Hedden T, Buckner R L, Dec. 2009
Evidence from intrinsic activity that asymmetry of the human brain
is controlled by multiple factors. Proceedings of the National Academy
of Sciences 106 (48), 20499-20503.
URL http://dx.doi.org/10.1073/pnas.0908073106

Power J D, Barnes K A, Snyder A Z, Schlaggar B L, Petersen S E, Feb 2012
Spurious but systematic correlations in functional connectivity 
MRI networks arise from subject motion. NeuroImage 59 (3), 2142-2154.
URL http://dx.doi.org/10.1016/j.neuroimage.2011.10.018
