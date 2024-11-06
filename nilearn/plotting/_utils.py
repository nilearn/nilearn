from nilearn.experimental.surface import PolyMesh, SurfaceImage


def check_surface_plotting_inputs(
    surf_map,
    surf_mesh,
    hemi="left",
    bg_map=None,
    map_var_name="surf_map",
    mesh_var_name="surf_mesh",
):
    """Check inputs for surface plotting.

    Where possible this will 'convert' the inputs
    if SurfaceImage or PolyMesh objects are passed
    to be able to give them to the surface plotting functions.

    Returns
    -------
    surf_map : numpy.ndarray
        Description of the output.

    surf_mesh : numpy.ndarray
        Description of the output.

    bg_map : str | pathlib.Path | numpy.ndarray | None

    """
    if surf_mesh is None and surf_map is None:
        raise TypeError(
            f"{mesh_var_name} and {map_var_name} cannot both be None."
            f"If you want to pass {mesh_var_name}=None, "
            f"then {mesh_var_name} must be a SurfaceImage instance."
        )

    if surf_mesh is None and not isinstance(surf_map, SurfaceImage):
        raise TypeError(
            f"If you want to pass {mesh_var_name}=None, "
            f"then {mesh_var_name} must be a SurfaceImage instance."
        )

    if isinstance(surf_mesh, PolyMesh):
        _check_hemi_present(surf_mesh, hemi)
        surf_mesh = surf_mesh.parts[hemi]

    if isinstance(surf_map, SurfaceImage):
        if surf_mesh is None:
            surf_mesh = surf_map.mesh.parts[hemi]
        surf_map = surf_map.data.parts[hemi]

    bg_map = _check_bg_map(bg_map, hemi)

    return surf_map, surf_mesh, bg_map


def _check_bg_map(bg_map, hemi):
    """Get the requested hemisphere if bg_map is a SurfaceImage.

    bg_map: Any

    hemi: str

    Returns
    -------
    bg_map : str | pathlib.Path | numpy.ndarray | None
    """
    if isinstance(bg_map, SurfaceImage):
        assert bg_map.data.parts[hemi] is not None
        bg_map = bg_map.data.parts[hemi]
    return bg_map


def _check_hemi_present(mesh, hemi):
    """Check that a given hemisphere exists in a PolyMesh."""
    if hemi not in mesh.parts:
        raise ValueError(f"{hemi} must be present in mesh")
