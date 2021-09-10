Glossary
========

.. currentmodule:: nilearn

The Glossary provides short definitions of neuro-imaging concepts as well
as Nilearn specific vocabulary.

If you wish to add a missing term, please `create a new issue`_ or
`open a Pull Request`_.

.. glossary::
    :sorted:


    ANOVA
        `Analysis of variance`_ is a collection of statistical models and
        their associated estimation procedures used to analyze the differences
        among means.

    AUC
        `Area under the curve`_.

    BIDS
        `Brain Imaging Data Structure`_ is a simple and easy to adopt way
        of organizing neuroimaging and behavioral data.

    BOLD
        Blood oxygenation level dependent. This is the kind of signal measured
        by functional Magnetic Resonance Imaging.

    CanICA
        `Canonical independent component analysis`_.

    contrast
        A `contrast`_ is a linear combination of variables (parameters or
        statistics) whose coefficients add up to zero, allowing comparison
        of different treatments.

    Decoding
        `Decoding`_ consists in predicting, from brain images, the conditions
        associated to trial.

    EEG
        `Electroencephalography`_ is a monitoring method to record electrical
        activity of the brain.

    EPI
        Echo-Planar Imaging. This is the type of sequence used to acquire
        functional or diffusion MRI data.

    FDR correction
        `False discovery rate`_ controlling procedures are designed to control
        the expected proportion of "discoveries" (rejected null hypotheses)
        that are false (incorrect rejections of the null).

    FIR
        Finite impulse response. This is a type of free-form temporal filter
        that is used to link neural activity with hemodynamic response, when
        there is uncertainty on the true model.

    fMRI
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
        False positive rate correction. This refers to the methods employed to
        correct false positive rates such as the Bonferroni correction which
        divides the significance level by the number of comparisons made.

    FREM
        `FREM`_ means "Fast ensembling of REgularized Models". It uses an implicit
        spatial regularization through fast clustering and aggregates a high
        number of estimators trained on various splits of the training set, thus
        returning a very robust decoder at a lower computational cost than other
        spatially regularized methods.

    functional connectivity
        Functional connectivity is a measure of the similarity of the response
        patterns in two or more regions.

    functional connectome
        A `functional connectome`_ is a set of connections representing brain
        interactions between regions.

    FWER correction
        `Family-wise error rate`_ is the probability of making one or more
        false discoveries, or type I errors when performing multiple
        hypotheses tests.

    GLM
        General Linear Model. This is the name of the models traditionally fit
        to fMRI data, where one linear model is fit to each voxel time course.

    HRF
        Haemodynamic response function. This is a temporal filter that converts
        neural signals to hemodynamic signals observable with :term:`fMRI`.

    ICA
        `Independent component analysis`_ is a computational method for separating
        a multivariate signal into additive subcomponents.

    MEG
        `Magnetoencephalography`_ is a functional neuroimaging technique for mapping
        brain activity by recording magnetic fields produced by electrical currents
        occurring naturally in the brain.

    MNI
        MNI stands for "Montreal Neurological Institute". Usually, this is
        used to reference the MNI space/template. The current standard MNI
        template is the ICBM152, which is the average of 152 normal MRI scans
        that have been matched to the MNI305 using a 9 parameter affine transform.

    MVPA
        Mutli-Voxel Pattern Analysis. This is the way :term:`supervised learning`
        methods are called in the field of brain imaging.

    Neurovault
        `Neurovault`_ is a public repository of unthresholded statistical maps,
        parcellations, and atlases of the human brain.

    parcellation
        Act of dividing the brain into smaller regions, i.e. parcels. Parcellations
        can be defined by many different criteria including anatomical or functional
        characteristics. Parcellations can either be composed of "hard" deterministic
        parcels with no overlap between individual regions or "soft" probabilistic
        parcels with a non-zero probability of overlap.

    predictive modelling
        `Predictive modelling`_ uses statistics to predict outcomes.

    ReNA
        `Recursive nearest agglomeration`_.

    resting-state
        `Resting state`_ :term:`fMRI` is a method of functional magnetic resonance
        imaging that is used in brain mapping to evaluate regional interactions that
        occur in a resting or task-negative state, when an explicit task is not being
        performed.

    ROC
        The `receiver operating characteristic curve`_ plots the true positive rate
        (TPR) against the false positive rate (FPR) at various threshold settings.

    Searchlight
        `Searchlight analysis`_ consists of scanning the brain with a searchlight.
        That is, a ball of given radius is scanned across the brain volume and the
        prediction accuracy of a classifier trained on the corresponding voxels is measured.

    SpaceNet
        `SpaceNet`_ is a decoder implementing spatial penalties which improve brain
        decoding power as well as decoder maps.

    SPM
        `Statistical Parametric Mapping`_ is a statistical technique for examining
        differences in brain activity recorded during functional neuroimaging
        experiments. It may alternatively refer to a `software`_ created by the Wellcome
        Department of Imaging Neuroscience at University College London to carry out
        such analyses.

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

    SVM
        `Support vector machines`_ are a set of :term:`supervised learning` methods used
        for :term:`classification`, :term:`regression` and outliers detection.

    TR
        Repetition time. This is the time in seconds between the beginning of an
        acquisition of one volume and the beginning of acquisition of the volume following it.

    Unsupervised learning
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
        `Voxel-Based Morphometry`_ measures differences in local concentrations of brain
        tissue, through a voxel-wise comparison of multiple brain images.

    voxel
        A voxel represents a value on a regular grid in 3D space.

    Ward clustering
        Ward’s algorithm is a hierarchical clustering algorithm: it recursively merges voxels,
        then clusters that have similar signal (parameters, measurements or time courses).


.. LINKS

.. _`create a new issue`:
    https://github.com/nilearn/nilearn/issues/new/choose

.. _`open a Pull Request`:
    https://github.com/nilearn/nilearn/compare

.. _`Analysis of variance`:
    https://en.wikipedia.org/wiki/Analysis_of_variance

.. _`Area under the curve`:
    https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics

.. _`Brain Imaging Data Structure`:
    https://bids.neuroimaging.io/

.. _`Canonical independent component analysis`:
    https://arxiv.org/abs/1006.2300

.. _`contrast`:
    https://en.wikipedia.org/wiki/Contrast_(statistics)

.. _`Decoding`:
    https://nilearn.github.io/decoding/decoding_intro.html

.. _`Electroencephalography`:
    https://en.wikipedia.org/wiki/Electroencephalography

.. _`False discovery rate`:
    https://en.wikipedia.org/wiki/False_discovery_rate

.. _`Family-wise error rate`:
    https://en.wikipedia.org/wiki/Family-wise_error_rate

.. _`FREM`:
    https://www.sciencedirect.com/science/article/abs/pii/S1053811917308182

.. _`functional connectome`:
    https://nilearn.github.io/connectivity/functional_connectomes.html

.. _`Independent component analysis`:
    https://en.wikipedia.org/wiki/Independent_component_analysis

.. _`Magnetoencephalography`:
    https://en.wikipedia.org/wiki/Magnetoencephalography

.. _`Neurovault`:
    https://www.neurovault.org/

.. _`Predictive modelling`:
    https://en.wikipedia.org/wiki/Predictive_modelling

.. _`Recursive nearest agglomeration`:
    https://hal.archives-ouvertes.fr/hal-01366651/

.. _`receiver operating characteristic curve`:
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic

.. _`Resting state`:
    https://en.wikipedia.org/wiki/Resting_state_fMRI

.. _`Searchlight analysis`:
    https://nilearn.github.io/decoding/searchlight.html

.. _`software`:
    https://www.fil.ion.ucl.ac.uk/spm/software/

.. _`SpaceNet`:
    https://nilearn.github.io/decoding/space_net.html

.. _`Statistical Parametric Mapping`:
    https://en.wikipedia.org/wiki/Statistical_parametric_mapping

.. _`Supervised learning`:
    https://en.wikipedia.org/wiki/Supervised_learning

.. _`Unsupervised learning`:
    https://en.wikipedia.org/wiki/Unsupervised_learning

.. _`Support vector machines`:
    https://scikit-learn.org/stable/modules/svm.html

.. _`Voxel-Based Morphometry`:
    https://en.wikipedia.org/wiki/Voxel-based_morphometry
