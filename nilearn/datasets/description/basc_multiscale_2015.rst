.. _basc_atlas:

BASC multiscale atlas
=====================

Access
------
See :func:`nilearn.datasets.fetch_atlas_basc_multiscale_2015`.

Content
-------
This work is a derivative from the Cambridge sample found
in the `1000 functional connectome project <https://fcon_1000.projects.nitrc.org/fcpClassic/FcpTable.html>`_
(:footcite:t:`Liu2009`), originally released under Creative Commons -- Attribution Non-Commercial.
It includes group brain parcellations generated
from :term:`resting-state` functional magnetic resonance images
for about 200 young healthy subjects. Multiple scales (number of networks) are available,
and includes 7, 12, 20, 36, 64, 122, 197, 325, 444.
The brain parcellations have been generated using a method called bootstrap analysis of stable clusters
(BASC, :footcite:t:`Bellec2010`) and the scales have been selected
using a data-driven method called MSTEPS (:footcite:t:`Bellec2013`).


This release more specifically contains the following files:

    :'description': a markdown (text) description of the release.
    :'scale007', 'scale012', 'scale020', 'scale036', 'scale064', 'scale122', 'scale197', 'scale325', 'scale444':
        brain_parcellation_cambridge_basc_multiscale_(sym,asym)_scale(NNN).nii.gz:
        a 3D volume .nii format at 3 mm isotropic resolution,
        in the :term:`MNI` `non-linear 2009a space <https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009>`_.
        Region number I is filled with Is (background is filled with 0s).


Note that two versions of the template are available, ending with either
nii_sym or nii_asym. The asym flavor contains brain images that have been
registered in the asymmetric version of the :term:`MNI` brain template (reflecting
that the brain is asymmetric), while with the sym flavor they have been
registered in the symmetric version of the :term:`MNI` template. The symmetric
template has been forced to be symmetric anatomically, and is therefore
ideally suited to study homotopic functional connections in fMRI: finding
homotopic regions simply consists of flipping the x-axis of the template.

Preprocessing
-------------
The datasets were analyzed using
the NeuroImaging Analysis Kit (`NIAK <https://github.com/SIMEXP/niak>`_) version 0.12.14,
under CentOS version 6.3
with `Octave <https://octave.org>`_ version 3.8.1
and the `Minc toolkit <https://bic-mni.github.io/>`_ version 0.3.18.
Each :term:`fMRI` dataset was corrected for inter-slice difference in acquisition time
and the parameters of a rigid-body motion were estimated for each time frame.
Rigid-body motion was estimated within as well as between runs, using the
median volume of the first run as a target. The median volume of one selected
:term:`fMRI` run for each subject was coregistered with a T1 individual scan using
Minctracc (:footcite:t:`Collins1997`), which was itself non-linearly transformed
to the Montreal Neurological Institute (:term:`MNI`) template (:footcite:t:`Fonov2011`)
using the CIVET pipeline (:footcite:t:`AdDabbagh2006`). The :term:`MNI` symmetric template
was generated from the ICBM152 sample of 152 young adults, after 40 iterations
of non-linear coregistration. The rigid-body transform, fMRI-to-T1 transform
and T1-to-stereotaxic transform were all combined, and the functional volumes
were resampled in the :term:`MNI` space at a 3 mm isotropic resolution. The
"scrubbing" method of (:footcite:t:`Power2012`), was used to remove the volumes
with excessive motion (frame displacement greater than 0.5 mm). A minimum
number of 60 unscrubbed volumes per run, corresponding to ~180 s of
acquisition, was then required for further analysis. The following nuisance
parameters were regressed out from the time series at each voxel: slow time
drifts (basis of discrete cosines with a 0.01 Hz high-pass cut-off), average
signals in conservative masks of the white matter and the lateral ventricles
as well as the first principal components (95% energy) of the
six rigid-body motion parameters and their squares (:footcite:t:`Giove2009`).
The :term:`fMRI` volumes were finally spatially smoothed
with a 6 mm isotropic Gaussian blurring kernel.

Bootstrap Analysis of Stable Clusters
-------------------------------------
Brain parcellations were derived using BASC (:footcite:t:`Bellec2010`). A region
growing algorithm was first applied to reduce the brain into regions of
roughly equal size, set to 1000 mm3. The BASC used 100 replications of a
hierarchical clustering with Ward's criterion on resampled individual time
series, using circular block bootstrap. A consensus clustering (hierarchical
with Ward's criterion) was generated across all the individual clustering
replications pooled together, hence generating group clusters. The generation
of group clusters was itself replicated by bootstrapping subjects 500 times,
and a (final) consensus clustering (hierarchical Ward's criterion) was
generated on the replicated group clusters.
The MSTEPS procedure (:footcite:t:`Bellec2013`) was implemented
to select a data-driven subset of scales in the range 5-500,
approximating the group stability matrices up to 5% residual energy,
through linear interpolation over selected scales.
Note that the
number of scales itself was selected by the MSTEPS procedure in a data-driven
fashion, and that the number of individual, group and final (consensus) number
of clusters were not necessarily identical.

References
----------

.. footbibliography::


License
-------
unknown
