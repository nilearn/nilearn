Glossary
========

.. currentmodule:: nilearn

The Glossary provides short definitions of neuro-imaging concepts as well
as Nilearn specific vocabulary.

If you wish to add a missing term, please
:nilearn-gh:`create an issue <issues/new/choose>` or
:nilearn-gh:`open a Pull Request <compare>`.

.. glossary::
    :sorted:


    ANOVA
        `Analysis of variance`_ is a collection of statistical models and
        their associated estimation procedures used to analyze the differences
        among means.

    AUC
        :sklearn:`Area under the curve <modules/model_evaluation.html#roc-metrics>`.

    Beta
        See :term:`Parameter estimate`.

    BIDS
        `Brain Imaging Data Structure`_ is a simple and easy to adopt way
        of organizing neuroimaging and behavioral data.

    BOLD
        Blood oxygenation level dependent. This is the kind of signal measured
        by functional Magnetic Resonance Imaging.

    CanICA
        `Canonical independent component analysis`_.

    Closing
        `Closing`_ is, together with :term:`opening<Opening>`, one of the basic
        operations of `mathematical morphology`_. The closing of a binary image
        by a structuring element is defined as the :term:`erosion<Erosion>` of
        the :term:`dilation<Dilation>` of that set.

    contrast
        A `contrast`_ is a linear combination of variables (parameters or
        statistics) whose coefficients add up to zero, allowing comparison
        of different treatments.

    Decoding
        :ref:`Decoding <decoding_intro>` consists in predicting, from brain
        images, the conditions associated to trial.

    Deterministic atlas
        A deterministic atlas is a hard parcellation of the brain into
        non-overlaping regions, that might have been obtained by segmentation or clustering methods.
        These objects are represented as 3D images of the brain composed of
        integer values, called 'labels', which define the different regions.
        In such atlases, and contrary to
        :term:`probabilistic atlases<Probabilistic atlas>`, a :term:`voxel`
        belongs to one, and only one, region.

    Dictionary learning
        `Dictionary learning`_ (or sparse coding) is a representation learning
        method aiming at finding a sparse representation of the input data as
        a linear combination of basic elements called atoms. The identification
        of these atoms composing the dictionary relies on a sparsity principle:
        maximally sparse representations of the dataset are sought for. Atoms
        are not required to be orthogonal.

    Dilation
        `Dilation`_ is, with :term:`erosion<Erosion>` one of the fundamental
        operations of `mathematical morphology`_ from which other operations
        like :term:`opening<Opening>` or :term:`closing<Closing>` are based.
        Dilation uses a structuring element for probing and expanding the
        shapes contained in the input image.

    EEG
        `Electroencephalography`_ is a monitoring method to record electrical
        activity of the brain.

    EPI
        Echo-Planar Imaging. This is the type of sequence used to acquire
        functional or diffusion MRI data.

    Erosion
        `Erosion`_ is, with :term:`dilation<Dilation>`, one of the fundamental
        operations in `mathematical morphology`_ from which other operations
        like :term:`opening<Opening>` or :term:`closing<Closing>` are based.
        Erosion uses a structuring element for probing and reducing the shapes
        contained in the input image.

    faces
        When referring to surface data, a face corresponds to one of the triangles
        of a triangular :term:`mesh`.

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

    fMRIPrep
        `fMRIPrep`_ is a :term:`fMRI` data preprocessing pipeline designed
        to provide an interface robust to variations in scan acquisition
        protocols with minimal user input. It performs basic processing
        steps (coregistration, normalization, unwarping, noise component
        extraction, segmentation, skullstripping etc.) providing outputs,
        often called confounds or nuisance parameters, that can be easily
        submitted to a variety of group level analyses, including task-based
        or resting-state :term:`fMRI`, graph theory measures, surface or
        volume-based statistics, etc.

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
        A :ref:`functional connectome <functional_connectomes>` is a set of
        connections representing brain interactions between regions.

    FWER correction
        `Family-wise error rate`_ is the probability of making one or more
        false discoveries, or type I errors when performing multiple
        hypotheses tests.

    FWHM
        `FWHM`_ stands for "full width at half maximum". In a distribution, it
        refers to the width of a filter, expressed as the diameter of the area
        on which the filter value is above half its maximal value.

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

    mesh
        In the context of brain surface data, a mesh refers to a 3D representation
        of the brain's surface geometry.
        It is a collection of vertices, edges, and faces
        that define the shape and structure of the brain's outer surface.
        Each :term:`vertex` represents a point in 3D space,
        and edges connect these vertices to form a network.
        :term:`Faces<faces>` are then created by connecting
        three or more vertices to form triangles.

    MNI
        MNI stands for "Montreal Neurological Institute". Usually, this is
        used to reference the MNI space/template. The current standard MNI
        template is the ICBM152, which is the average of 152 normal MRI scans
        that have been matched to the MNI305 using a 9 parameter affine transform.

    MVPA
        Multi-Voxel Pattern Analysis. This is the way :term:`supervised learning`
        methods are called in the field of brain imaging.

    Neurovault
        `Neurovault`_ is a public repository of unthresholded statistical maps,
        parcellations, and atlases of the human brain.

    Opening
        `Opening`_ is, together with :term:`closing<Closing>`, one of the basic
        operations of `mathematical morphology`_. It is defined as the
        :term:`dilation<Dilation>` of the :term:`erosion<Erosion>` of a set by a
        structuring element.

    Parameter estimate
        In the context of a :term:`GLM`, each :term:`contrast` comparing rows in the
        design matrix results in a parameter estimate (PE) that signifies how
        well the underlying model fits the data at each :term:`voxel`. For statistical
        inferences the parameter estimate, sometimes also referred to as
        :term:`beta`, is commonly converted to either a t-, or z-statistic. In
        nilearn the parameter estimate (or beta) is referred to as
        ``effect_size``.

    parcellation
        Act of dividing the brain into smaller regions, i.e. parcels. Parcellations
        can be defined by many different criteria including anatomical or functional
        characteristics. Parcellations can either be composed of "hard" deterministic
        parcels with no overlap between individual regions or "soft" probabilistic
        parcels with a non-zero probability of overlap.

    predictive modelling
        `Predictive modelling`_ uses statistics to predict outcomes.

    Probabilistic atlas
        Probabilistic atlases define soft parcellations of the brain in which
        the regions may overlap. In such atlases, and contrary to
        :term:`deterministic atlases<Deterministic atlas>`, a :term:`voxel`
        can belong to several components. These atlases are represented by 4D
        images where the 3D components, also called 'spatial maps', are
        stacked along one dimension (usually the 4th dimension). In each
        3D component, the value at a given :term:`voxel` indicates how
        strongly this :term:`voxel` is related to this component.

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
        :ref:`Searchlight analysis <searchlight>` consists of scanning the brain with a searchlight.
        That is, a ball of given radius is scanned across the brain volume and the
        prediction accuracy of a classifier trained on the corresponding voxels is measured.

    SNR
        `SNR`_ stands for "Signal to Noise Ratio" and is a measure comparing the level
        of a given signal to the level of the background noise.

    SpaceNet
        :ref:`SpaceNet <space_net>` is a decoder implementing spatial penalties
        which improve brain decoding power as well as decoder maps.

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
        :sklearn:`Support vector machines <modules/svm.html>` are a set of
        :term:`supervised learning` methods used for :term:`classification`,
        :term:`regression` and outliers detection.

    TFCE
        Threshold-free cluster enhancement is a voxel-level metric that combines signal
        magnitude and cluster extent to enhance the importance of clusters that are large,
        have high magnitude, or both.

        For more information about TFCE, see :footcite:t:`Smith2009a` or
        `Benedikt Ehinger's tutorial <https://benediktehinger.de/blog/science/threshold-free-cluster-enhancement-explained/>`_.

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

    vertex
        A vertex (plural vertices) represents the coordinate
        of an angle of :term:`face<faces>`
        on a triangular :term:`mesh` in 3D space.

    voxel
        A voxel represents a value on a regular grid in 3D space.

    Ward clustering
        Ward's algorithm is a hierarchical clustering algorithm: it recursively merges voxels,
        then clusters that have similar signal (parameters, measurements or time courses).

References
----------

.. footbibliography::


.. LINKS

.. _`Analysis of variance`:
    https://en.wikipedia.org/wiki/Analysis_of_variance

.. _`Brain Imaging Data Structure`:
    https://bids.neuroimaging.io/

.. _`Canonical independent component analysis`:
    https://arxiv.org/abs/1006.2300

.. _`Closing`:
    https://en.wikipedia.org/wiki/Closing_(morphology)

.. _`contrast`:
    https://en.wikipedia.org/wiki/Contrast_(statistics)

.. _`Dictionary learning`:
    https://en.wikipedia.org/wiki/Sparse_dictionary_learning

.. _`Dilation`:
    https://en.wikipedia.org/wiki/Dilation_(morphology)

.. _`Electroencephalography`:
    https://en.wikipedia.org/wiki/Electroencephalography

.. _`Erosion`:
    https://en.wikipedia.org/wiki/Erosion_(morphology)

.. _`False discovery rate`:
    https://en.wikipedia.org/wiki/False_discovery_rate

.. _`Family-wise error rate`:
    https://en.wikipedia.org/wiki/Family-wise_error_rate

.. _`fMRIPrep`:
    https://fmriprep.org/en/stable/

.. _`FREM`:
    https://www.sciencedirect.com/science/article/abs/pii/S1053811917308182

.. _`FWHM`:
    https://en.wikipedia.org/wiki/Full_width_at_half_maximum

.. _`Independent component analysis`:
    https://en.wikipedia.org/wiki/Independent_component_analysis

.. _`Magnetoencephalography`:
    https://en.wikipedia.org/wiki/Magnetoencephalography

.. _`mathematical morphology`:
    https://en.wikipedia.org/wiki/Mathematical_morphology

.. _`Neurovault`:
    https://www.neurovault.org/

.. _`Opening`:
    https://en.wikipedia.org/wiki/Opening_(morphology)

.. _`Predictive modelling`:
    https://en.wikipedia.org/wiki/Predictive_modelling

.. _`Recursive nearest agglomeration`:
    https://hal.archives-ouvertes.fr/hal-01366651/

.. _`receiver operating characteristic curve`:
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic

.. _`Resting state`:
    https://en.wikipedia.org/wiki/Resting_state_fMRI

.. _`SNR`:
    https://en.wikipedia.org/wiki/Signal-to-noise_ratio

.. _`software`:
    https://www.fil.ion.ucl.ac.uk/spm/software/

.. _`Statistical Parametric Mapping`:
    https://en.wikipedia.org/wiki/Statistical_parametric_mapping

.. _`Supervised learning`:
    https://en.wikipedia.org/wiki/Supervised_learning

.. _`Unsupervised learning`:
    https://en.wikipedia.org/wiki/Unsupervised_learning

.. _`Voxel-Based Morphometry`:
    https://en.wikipedia.org/wiki/Voxel-based_morphometry
