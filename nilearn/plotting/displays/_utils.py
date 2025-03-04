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


def _get_index_from_direction(direction):
    """Return numerical index from direction."""
    directions = ["x", "y", "z"]
    try:
        # l and r are subcases of x
        index = 0 if direction in "lr" else directions.index(direction)
    except ValueError:
        message = (
            f"{direction} is not a valid direction. "
            "Allowed values are 'l', 'r', 'x', 'y' and 'z'"
        )
        raise ValueError(message)
    return index


def coords_3d_to_2d(coords_3d, direction, return_direction=False):
    """Project 3d coordinates into 2d ones given the direction of a cut."""
    index = _get_index_from_direction(direction)
    dimensions = [0, 1, 2]
    dimensions.pop(index)
    if return_direction:
        return coords_3d[:, dimensions], coords_3d[:, index]
    return coords_3d[:, dimensions]
