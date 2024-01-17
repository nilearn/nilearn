Harvard Oxford Atlas


Notes
-----
Probabilistic atlases covering 48 cortical and 21 subcortical structural areas,
derived from structural data and segmentations kindly
provided by the Harvard Center for Morphometric Analysis.

T1-weighted images of 21 healthy male and 16 healthy female subjects (ages 18-50)
were individually segmented by the CMA using semi-automated tools developed in-house.
The T1-weighted images were affine-registered to MNI152 space using FLIRT (FSL),
and the transforms then applied to the individual labels.
Finally, these were combined across subjects to form population probability maps for each label.

For more details: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases

Content
-------
    :'maps': nifti image containing regions or their probability
    :'labels': list of labels for the regions in the atlas.

References
----------
Makris N, Goldstein JM, Kennedy D, Hodge SM, Caviness VS, Faraone SV, Tsuang MT, Seidman LJ.
Decreased volume of left and total anterior insular lobule in schizophrenia.
Schizophr Res. 2006 Apr;83(2-3):155-71

Frazier JA, Chiu S, Breeze JL, Makris N, Lange N, Kennedy DN, Herbert MR, Bent EK,
Koneru VK, Dieterich ME, Hodge SM, Rauch SL, Grant PE, Cohen BM, Seidman LJ, Caviness VS, Biederman J.
Structural brain magnetic resonance imaging of limbic and thalamic volumes in pediatric bipolar disorder.
Am J Psychiatry. 2005 Jul;162(7):1256-65

Desikan RS, Ségonne F, Fischl B, Quinn BT, Dickerson BC, Blacker D, Buckner RL,
Dale AM, Maguire RP, Hyman BT, Albert MS, Killiany RJ.
An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest.
Neuroimage. 2006 Jul 1;31(3):968-80.

Goldstein JM, Seidman LJ, Makris N, Ahern T, O'Brien LM, Caviness VS Jr,
Kennedy DN, Faraone SV, Tsuang MT.
Hypothalamic abnormalities in schizophrenia: sex effects and genetic vulnerability.
Biol Psychiatry. 2007 Apr 15;61(8):935-45


License
-------
See https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Licence
