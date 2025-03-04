def get_create_display_fun(display_mode, class_dict):
    """Help for functions \
    :func:`~nilearn.plotting.displays.get_slicer` and \
    :func:`~nilearn.plotting.displays.get_projector`.
    """
    try:
        return class_dict[display_mode].init_with_figure
    except KeyError:
        message = (
            f"{display_mode} is not a valid display_mode. "
            f"Valid options are {sorted(class_dict.keys())}"
        )
        raise ValueError(message)

