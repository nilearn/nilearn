import numpy as np

from nilearn.surface import InMemoryMesh, PolyMesh, SurfaceImage


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

    surf_mesh : numpy.ndarray

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
        surf_mesh = _check_hemi_present(surf_mesh, hemi)

    if isinstance(surf_mesh, SurfaceImage):
        raise TypeError(
            "'surf_mesh' cannot be a SurfaceImage instance. ",
            "Accepted types are: str, list of two numpy.ndarray, "
            "InMemoryMesh, PolyMesh, or None.",
        )

    if isinstance(surf_map, SurfaceImage):
        if surf_mesh is None:
            surf_mesh = _check_hemi_present(surf_map.mesh, hemi)
        if len(surf_map.shape) > 1 and surf_map.shape[1] > 1:
            raise TypeError(
                "Input data has incompatible dimensionality. "
                f"Expected dimension is ({surf_map.shape[0]},) "
                f"or ({surf_map.shape[0]}, 1) "
                f"and you provided a {surf_map.shape} surface image."
            )
        # concatenate the left and right data if hemi is "both"
        if hemi == "both":
            surf_map = np.concatenate(
                (surf_map.data.parts["left"], surf_map.data.parts["right"])
            ).T
        else:
            surf_map = surf_map.data.parts[hemi].T

    bg_map = _check_bg_map(bg_map, hemi)

    return surf_map, surf_mesh, bg_map


def _check_bg_map(bg_map, hemi):
    """Get the requested hemisphere if bg_map is a SurfaceImage. If the
    hemisphere is not present, raise an error. If the hemisphere is "both",
    concatenate the left and right hemispheres.

    bg_map : Any

    hemi : str

    Returns
    -------
    bg_map : str | pathlib.Path | numpy.ndarray | None
    """
    if isinstance(bg_map, SurfaceImage):
        if len(bg_map.shape) > 1 and bg_map.shape[1] > 1:
            raise TypeError(
                "Input data has incompatible dimensionality. "
                f"Expected dimension is ({bg_map.shape[0]},) "
                f"or ({bg_map.shape[0]}, 1) "
                f"and you provided a {bg_map.shape} surface image."
            )
        if hemi == "both":
            bg_map = np.concatenate(
                (bg_map.data.parts["left"], bg_map.data.parts["right"])
            )
        else:
            assert bg_map.data.parts[hemi] is not None
            bg_map = bg_map.data.parts[hemi]
    return bg_map


def _check_hemi_present(mesh, hemi):
    """Check that a given hemisphere exists in a PolyMesh or is "both", and
    return the corresponding mesh. If "both" is requested, combine the left
    and right hemispheres.
    """
    if hemi not in [*list(mesh.parts), "both"]:
        raise ValueError(f"{hemi} must be one of left, right or both.")

    if hemi == "both":
        combined_coords = np.concatenate(
            (mesh.parts["left"].coordinates, mesh.parts["right"].coordinates)
        )
        combined_faces = np.concatenate(
            (
                mesh.parts["left"].faces,
                mesh.parts["right"].faces
                + mesh.parts["left"].coordinates.shape[0],
            )
        )
        surf_mesh = InMemoryMesh(combined_coords, combined_faces)
    else:
        surf_mesh = mesh.parts[hemi]

    return surf_mesh
