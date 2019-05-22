# first line: 24
def filter_and_extract(imgs, extraction_function,
                       parameters,
                       memory_level=0, memory=Memory(cachedir=None),
                       verbose=0,
                       confounds=None,
                       copy=True,
                       dtype=None):
    """Extract representative time series using given function.

    Parameters
    ----------
    imgs: 3D/4D Niimg-like object
        Images to be masked. Can be 3-dimensional or 4-dimensional.

    extraction_function: function
        Function used to extract the time series from 4D data. This function
        should take images as argument and returns a tuple containing a 2D
        array with masked signals along with a auxiliary value used if
        returning a second value is needed.
        If any other parameter is needed, a functor or a partial
        function must be provided.

    For all other parameters refer to NiftiMasker documentation

    Returns
    -------
    signals: 2D numpy array
        Signals extracted using the extraction function. It is a scikit-learn
        friendly 2D array with shape n_samples x n_features.
    """
    # Since the calling class can be any *Nifti*Masker, we look for exact type
    if verbose > 0:
        class_name = enclosing_scope_name(stack_level=10)

    # If we have a string (filename), we won't need to copy, as
    # there will be no side effect
    if isinstance(imgs, _basestring):
        copy = False

    if verbose > 0:
        print("[%s] Loading data from %s" % (
            class_name,
            _utils._repr_niimgs(imgs)[:200]))
    imgs = _utils.check_niimg(imgs, atleast_4d=True, ensure_ndim=4,
                              dtype=dtype)

    sample_mask = parameters.get('sample_mask')
    if sample_mask is not None:
        imgs = image.index_img(imgs, sample_mask)

    target_shape = parameters.get('target_shape')
    target_affine = parameters.get('target_affine')
    if target_shape is not None or target_affine is not None:
        if verbose > 0:
            print("[%s] Resampling images" % class_name)
        imgs = cache(
            image.resample_img, memory, func_memory_level=2,
            memory_level=memory_level, ignore=['copy'])(
                imgs, interpolation="continuous",
                target_shape=target_shape,
                target_affine=target_affine,
                copy=copy)

    smoothing_fwhm = parameters.get('smoothing_fwhm')
    if smoothing_fwhm is not None:
        if verbose > 0:
            print("[%s] Smoothing images" % class_name)
        imgs = cache(
            image.smooth_img, memory, func_memory_level=2,
            memory_level=memory_level)(
                imgs, parameters['smoothing_fwhm'])

    if verbose > 0:
        print("[%s] Extracting region signals" % class_name)
    region_signals, aux = cache(extraction_function, memory,
                                func_memory_level=2,
                                memory_level=memory_level)(imgs)

    # Temporal
    # --------
    # Detrending (optional)
    # Filtering
    # Confounds removing (from csv file or numpy array)
    # Normalizing
    if verbose > 0:
        print("[%s] Cleaning extracted signals" % class_name)
    sessions = parameters.get('sessions')
    region_signals = cache(
        signal.clean, memory=memory, func_memory_level=2,
        memory_level=memory_level)(
            region_signals,
            detrend=parameters['detrend'],
            standardize=parameters['standardize'],
            t_r=parameters['t_r'],
            low_pass=parameters['low_pass'],
            high_pass=parameters['high_pass'],
            confounds=confounds,
            sessions=sessions)

    return region_signals, aux
