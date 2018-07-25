=====================================
Introduction: nistats in a nutshell
=====================================

.. contents:: **Contents**
    :local:
    :depth: 1


What is nistats?
===========================================================================

.. topic:: **What is nistats?**

           Nistats is a python module to perform voxel-wise analyses of functional magnetic resonance images (fMRI) using linear models. It provides functions to create design matrices, at the subject and group levels, to estimate them from images series and to compute statistical maps (contrasts). It allows to perform the same statistical analyses than SPM or FSL (but it does not provide tools for preprocessing stages (realignement, spatial normalization, etc.); for this, see nipype???).



A primer on BOLD-fMRI data analysis
===================================

Functional magnetic resonance imaging (fMRI) is based on the fact that when local neural activity increases, increases in metabolism and blood flow lead to fluctuations of the relative concentrations of oxyhemoglobine (the red cells in the blood that carry oxygen) and deoxyhemoglobine (the same red cells after they have delivered the oxygen). Because oxy and deoxy hemoglobine have different magnetic properties (one is diamagnetic while the other is paramagnetic), they affect the local magnetic field in different ways. The signal picked up by the MRI scanner is sensitive to these modifications of the local magnetic field. To record cerebral activity, during functional sessions, the scanner is tuned to detect this "Blood Oxygen Level Dependent" (BOLD) contrast.

Brain activity is measured in sessions that last several minutes, during which the participant performs some kind of cognitive task while the scanner acquires brain images, typically every 2 or 3 seconds (the time between two successive image acquisition is called the Time of Repetition, or TR).

A cerebral MR image provides a 3D image of the brain that can be decomposed in voxels (the equivalent of pixels, but in 3 dimensions).

 INSERT HERE A PIC OF A BRAIN WITH A VOXEL GRID SUPERIMPOSED

A typical step in the preprocessing of MR images, involves spatially morphing these images onto a standard template (e.g. the MNI152 template from the Montreal Neurological Institute). One this is done, the coordinates of a voxel are in the same space as the template and can be used to estimate its brain location using brain atlases.

The series of images acquired during a functional session provide, in each voxel, a time series, sampled at the TR.

INSERT HERE A SAMPLE OF A TIME SERIES IN A VOXEL (or several voxels)

One way to analyze such times series consists in comparing them to a *model* buld from our knowledge of the events that occurred during the functional session. Events can correspond to actions of the participant (e.g. button presses), presentations of sensory stimui (e.g. sound, images), or hypothesized internal processes (e.g. memorization of a stimulus), ...

.. INSERT HERE AN IMAGE OF a TIME DIAGRAM OF EVENTS DURING A RUN

.. figure:: images/stimulation-time-diagram.png


One expects that a brain region involved in the processing of a certain type of event (e.g. the auditory cortex for sounds), would show a time course of activation that correlates with the time-diagram of these events. If the fMRI signal directly showed neural activity and did not contain any noise, we could just look at it in various voxel and detect those who conform to the time-diagrams.

Yet, we know, from previous measurements, that the BOLD signal does not follow the exact time course of stimulus processing and the underlying neural activity. The BOLD response, reflecting changes in blood flow and concentrations in oxy-deoxy hemoglobin. In reality, it is sluggish and long-lasting.

.. INSERT A FIG of the iHRF

.. figure:: images/spm_iHRF.png


From the knowledge of the impulse haemodynamic response (FIG), we can build a predicted time course from the time-diagram of a type of event (The operation is known a a convolution. Remark: it assumes linearity of the BOLD response, a assumption that may be wrong in some scenarios). It is this predicted time course, also known as a *predictor*, that is compared to the actual fMRI signal. If the correlation between the predictor and the signal is higher than expected by chance, the voxel is said to exhibit a significant response to the event type. 

.. INSERT A FIG SHOWING SIGNAL AND PREDICTOR AND their CORRELATION

.. figure:: images/time-course-and-model-fit-in-a-voxel.png

Correlations are computed separately at each voxel and a correlation map can be produced.

INSERT A CORRELATION (BRAIN) MAP HERE. With a certain Threshold.

In most fMRI experiments, several predictors described the events. To find the effect specific to each predictor, a multiple regression approach is typically used: all predictors are entered as columns in a *design-matrix* and the software find the linear combination of these columns that best fits the signal.  The weight assigned to each predictor by this linear cobination are estimates of the contribution of this predictor to the response in the voxel. One can plot this effect size maps or, maps showing their statistical significance (how unlikely theye are under the null hypothesis of no effect)

SHOW a beta map and SPMt map side by side.

.. figure:: images/example-spmZ_map.png

In brief, the analysis of fMRI images involves:

1. describing the paradigm in terms of events of various types occuring at certain times and having a certain duration.
2. from these time diagram, creating predictors for each type of event, typically using a convolution by the impulse HRF

3. assembling these predictors in a design-matrix, thus providing a *model*
4. estimate the parameters of the model, that is, the weights associated to each predictors, at each voxel, using multiple regression.
5. display the coefficients and/or their statistical significance. 




Tutorials
=========

    For tutorials, please check out the `Examples <auto_examples/index.html>`_ gallery

.. _installation:

Installing nistats
====================

.. raw:: html
   :file: install_doc_component.html
