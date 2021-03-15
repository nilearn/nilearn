Glossary
========

.. currentmodule:: nilearn

The Glossary provides short definitions of neuro-imaging concepts as well
as Nilearn specific vocabulary. 

If you wish to add a missing term, please `create a new issue`_ or
`open a Pull Request`_.

.. glossary::
    :sorted:

    
    BIDS
    bids
        `Brain Imaging Data Structure`_ is a simple and easy to adopt way
        of organizing neuroimaging and behavioral data.
    
    BOLD
    Bold
    bold
        Blood oxygenation level dependent.

    CanICA
        Canonical independant component analysis.

    Contrast
    contrast
    Contrasts
    contrasts
        A `contrast`_ is a linear combination of variables (parameters or
        statistics) whose coefficients add up to zero, allowing comparison
        of different treatments.

    Decoding
        `Decoding`_ consists in predicting, from brain images, the conditions
        associated to trial.

    EEG
        Electroencephalography.
    
    EPI
        Echo-Planar Imaging.

    FDR correction
        `False discovery rate`_ controlling procedures are designed to control
        the expected proportion of "discoveries" (rejected null hypotheses)
        that are false (incorrect rejections of the null).

    FIR
        Finite impulse response.

    fMRI
    fmri
        Functional magnetic resonance imaging is based on the fact that 
        when local neural activity increases, increases in metabolism and
        blood flow lead to fluctuations of the relative concentrations of
        oxyhaemoglobin (the red cells in the blood that carry oxygen) and
        deoxyhaemoglobin (the same red cells after they have delivered the
        oxygen). Oxyhaemoglobin and deoxyhaemoglobin have different magnetic
        properties (diamagnetic and paramagnetic, respectively), and they
        affect the local magnetic field in different ways.
        The signal picked up by the MRI scanner is sensitive to these
        modifications of the local magnetic field.

    FPR correction
    FPR
    fpr
        False positive rate correction.

    functional connectivity
        TODO.

    FWER correction
        `Family-wise error rate`_ is the probability of making one or more
        false discoveries, or type I errors when performing multiple
        hypotheses tests.

    GLM
        General Linear Model.

    HRF
    hrf
        Haemodynamic response function.

    ICA
        Independant component analysis.

    MEG
        Magnetoencephalography.

    MNI
        MNI stands for "Montreal Neurological Institute". Usually, this is
        used to reference the MNI space/template. The current standard MNI 
        template is the ICBM152, which is the average of 152 normal MRI scans 
        that have been matched to the MNI305 using a 9 parameter affine transform.

    MVPA
        Mutli-Voxel Pattern Analysis.

    parcellation
    parcellations
    brain parcellation
    brain parcellations
        TODO.

    predictive modelling
        TODO.

    resting-state
        TODO.

    supervised learning
        `Supervised learning`_ is interested in predicting an output variable, 
        or target, y, from data X. Typically, we start from labeled data (the 
        training set). We need to know the y for each instance of X in order to 
        train the model. Once learned, this model is then applied to new unlabeled 
        data (the test set) to predict the labels (although we actually know them).
        There are essentially two possible types of problems:

        .. glossary::

            regression
                 In regression problems, the objective is to predict a continuous 
                 variable, such as participant age, from the data X.

            classification
                In classification problems, the objective is to predict a binary 
                variable that splits the observations into two groups, such as 
                patients versus controls.

        In neuroimaging research, supervised learning is typically used to derive an 
        underlying cognitive process (e.g., emotional versus non-emotional theory of 
        mind), a behavioral variable (e.g., reaction time or IQ), or diagnosis status 
        (e.g., schizophrenia versus healthy) from brain images.

    TR
        Repetition time.

    Unsupervised learning
    unsupervised learning
        `Unsupervised learning`_ is concerned with data X without any labels. It analyzes 
        the structure of a dataset to find coherent underlying structure, for instance 
        using clustering, or to extract latent factors, for instance using independent 
        components analysis (:term:`ICA`).

        In neuroimaging research, it is typically used to create functional and anatomical 
        brain atlases by clustering based on connectivity or to extract the main brain 
        networks from resting-state correlations. An important option of future research 
        will be the identification of potential neurobiological subgroups in psychiatric 
        and neurobiological disorders.

    VBM
        TODO.

    voxel
    voxels
        A voxel represents a value on a regular grid in 3D space.

.. LINKS

.. _`create a new issue`:
    https://github.com/nilearn/nilearn/issues/new/choose

.. _`open a Pull Request`:
    https://github.com/nilearn/nilearn/compare

.. _`Brain Imaging Data Structure`:
    https://bids.neuroimaging.io/

.. _`contrast`:
    https://en.wikipedia.org/wiki/Contrast_(statistics)

.. _`Decoding`:
    https://nilearn.github.io/decoding/decoding_intro.html

.. _`False discovery rate`:
    https://en.wikipedia.org/wiki/False_discovery_rate

.. _`Family-wise error rate`:
    https://en.wikipedia.org/wiki/Family-wise_error_rate

.. _`Supervised learning`:
    https://en.wikipedia.org/wiki/Supervised_learning

.. _`Unsupervised learning`:
    https://en.wikipedia.org/wiki/Unsupervised_learning
