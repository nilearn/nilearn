"""Functions for surface visualization."""

from warnings import warn

import numpy as np
import pandas as pd

from nilearn import DEFAULT_DIVERGING_CMAP
from nilearn._utils.docs import fill_doc
from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg_conversions import check_niimg_3d
from nilearn._utils.param_validation import check_params
from nilearn.image import get_data
from nilearn.plotting._engine_utils import create_colormap_from_lut
from nilearn.plotting._utils import (
    DEFAULT_ENGINE,
    check_threshold_not_negative,
)
from nilearn.plotting.surface._utils import (
    DEFAULT_HEMI,
    check_hemispheres,
    check_surface_plotting_inputs,
    check_views,
    get_surface_backend,
)
from nilearn.surface import load_surf_data, load_surf_mesh, vol_to_surf
from nilearn.surface.surface import (
    FREESURFER_DATA_EXTENSIONS,
    check_extensions,
    check_mesh_is_fsaverage,
)

# subset of data format extensions supported
DATA_EXTENSIONS = (
    "gii",
    "gii.gz",
    "mgz",
)


@fill_doc
def plot_surf(
    surf_mesh=None,
    surf_map=None,
    bg_map=None,
    hemi=DEFAULT_HEMI,
    view=None,
    engine=DEFAULT_ENGINE,
    cmap=None,
    symmetric_cmap=None,
    colorbar=True,
    avg_method=None,
    threshold=None,
    alpha=None,
    bg_on_data=False,
    darkness=0.7,
    vmin=None,
    vmax=None,
    cbar_vmin=None,
    cbar_vmax=None,
    cbar_tick_format="auto",
    title=None,
    title_font_size=None,
    output_file=None,
    axes=None,
    figure=None,
):
    """Plot surfaces with optional background and data.

    .. versionadded:: 0.3

    Parameters
    ----------
    %(surf_mesh)s
        If `None` is passed, then ``surf_map`` must be a
        :obj:`~nilearn.surface.SurfaceImage` instance and the mesh from that
        :obj:`~nilearn.surface.SurfaceImage` instance will be used.

    surf_map : :obj:`str` or :class:`numpy.ndarray` or \
               :obj:`~nilearn.surface.SurfaceImage` or None, default=None
        Data to be displayed on the surface :term:`mesh`.
        Can be a file (valid formats are .gii, .mgz, .nii, .nii.gz, or
        Freesurfer specific files such as .thickness, .area, .curv, .sulc,
        .annot, .label) or a Numpy array with a value for each :term:`vertex`
        of the `surf_mesh`, or a :obj:`~nilearn.surface.SurfaceImage`
        instance.
        If `None` is passed for ``surf_mesh``, then ``surf_map`` must be a
        :obj:`~nilearn.surface.SurfaceImage` instance and its mesh will be
        used for plotting.

        When specified `surf_map` is of type :class:`numpy.ndarray`, to have a
        correct view, `hemi` should have a value corresponding to `surf_map`
        data.

    %(bg_map)s

    %(hemi)s

    %(view)s

    engine : {'matplotlib', 'plotly'}, default='matplotlib'

        .. versionadded:: 0.9.0

        Selects which plotting engine will be used by ``plot_surf``.
        Currently, only ``matplotlib`` and ``plotly`` are supported.

        .. note::
            To use the ``plotly`` engine, you need to have ``plotly``
            installed.

        .. note::
            To be able to save figures to disk with the ``plotly`` engine, you
            need to have ``kaleido`` installed.

        .. warning::
            The ``plotly`` engine is new and experimental. Please report bugs
            that you may encounter.

    %(cmap)s
        If `None`, ``matplotlib`` default will be chosen.

    symmetric_cmap : :obj:`bool`, default=None
        Whether to use a symmetric colormap or not.

        .. note::
            This option is currently only implemented for the ``plotly``
            engine.

        When using ``plotly`` as engine, ``symmetric_cmap`` will default to
        `False` if `None` is passed.

        .. versionadded:: 0.9.0

        .. versionchanged:: 0.12.0
            Default value changed to None.

    %(colorbar)s
        Default=True.

    %(avg_method)s

        .. note::
            This option is currently only implemented for the ``matplotlib``
            engine.

        When using ``matplotlib`` as engine, ``avg_method`` will default to
        ``"mean"`` if `None` is passed.

    %(threshold)s
        Default=None

    alpha : :obj:`float` or None, default=None
        Alpha level of the :term:`mesh` (not surf_data).

        If `'auto'` is chosen, ``alpha`` will default to `0.5` when no
        ``bg_map`` is passed and to `1` if a ``bg_map`` is passed.

        .. note::
            This option is currently only implemented for the ``matplotlib``
            engine.

        When using ``matplotlib`` as engine, ``alpha`` will default to `"auto"`
        if `None` is passed.

    %(bg_on_data)s

    %(darkness)s
        Default=1.

    %(vmin)s

    %(vmax)s

    cbar_vmin : :obj:`float` or None, default=None
        Lower bound for the colorbar.
        If `None`, the value will be set from the data.

        .. note::
            This option is currently only implemented for the ``matplotlib``
            engine.

    cbar_vmax : :obj:`float` or None, default=None
        Upper bound for the colorbar.
        If `None`, the value will be set from the data.

        .. note::
            This option is currently only implemented for the ``matplotlib``
            engine.

    %(cbar_tick_format)s
        Default="auto" which will select:

        - `'%%.2g'` (scientific notation) with ``matplotlib`` engine.
        - `'.1f'` (rounded floats) with ``plotly`` engine.

        .. versionadded:: 0.7.1

    %(title)s

    title_font_size : :obj:`int`, default=None
        Size of the title font

        .. note::
            This option is currently only implemented for the ``plotly``
            engine.

        When using ``plotly`` as engine, ``title_font_size`` will default to
        `18` if `None` is passed.

        .. versionadded:: 0.9.0

    %(output_file)s

    axes : instance of matplotlib axes or None, default=None
        The axes instance to plot to. The projection must be `"3d"` (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': "3d"})`,
        where axes should be passed.).
        If `None`, a new axes is created.

        .. note::
            This option is currently only implemented for the ``matplotlib``
            engine.

    %(figure)s

        .. note::
            This option is currently only implemented for the ``matplotlib``
            engine.

    Returns
    -------
    fig : :class:`~matplotlib.figure.Figure` or
    :class:`~nilearn.plotting.displays.PlotlySurfaceFigure`
        The surface figure. If ``engine='matplotlib'`` then a
        :class:`~matplotlib.figure.Figure` is returned.
        If ``engine='plotly'``, then a
        :class:`~nilearn.plotting.displays.PlotlySurfaceFigure`
        is returned

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf_roi : For plotting statistical maps on brain
        surfaces.

    nilearn.plotting.plot_surf_stat_map : for plotting statistical maps on
        brain surfaces.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.
    """
    check_params(locals())
    check_threshold_not_negative(threshold)
    surf_map, surf_mesh, bg_map = check_surface_plotting_inputs(
        surf_map, surf_mesh, hemi, bg_map
    )
    check_extensions(surf_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

    backend = get_surface_backend(engine)
    fig = backend._plot_surf(
        surf_mesh,
        surf_map=surf_map,
        bg_map=bg_map,
        hemi=hemi,
        view=view,
        cmap=cmap,
        symmetric_cmap=symmetric_cmap,
        colorbar=colorbar,
        avg_method=avg_method,
        threshold=threshold,
        alpha=alpha,
        bg_on_data=bg_on_data,
        darkness=darkness,
        vmin=vmin,
        vmax=vmax,
        cbar_vmin=cbar_vmin,
        cbar_vmax=cbar_vmax,
        cbar_tick_format=cbar_tick_format,
        title=title,
        title_font_size=title_font_size,
        output_file=output_file,
        axes=axes,
        figure=figure,
    )

    return fig


@fill_doc
def plot_surf_contours(
    surf_mesh=None,
    roi_map=None,
    hemi=DEFAULT_HEMI,
    levels=None,
    labels=None,
    colors=None,
    legend=False,
    cmap="tab20",
    title=None,
    output_file=None,
    axes=None,
    figure=None,
    **kwargs,
):
    """Plot contours of ROIs on a surface, optionally over a statistical map.

    Parameters
    ----------
    %(surf_mesh)s
        If None is passed, then ``roi_map`` must be a
        :obj:`~nilearn.surface.SurfaceImage` instance and the mesh from that
        :obj:`~nilearn.surface.SurfaceImage` instance will be used.

    roi_map : :obj:`str` or :class:`numpy.ndarray` or \
              :obj:`~nilearn.surface.SurfaceImage` or None, default=None
        ROI map to be displayed on the surface mesh, can be a file (valid
        formats are .gii, .mgz, or Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label) or a Numpy array with
        a value for each :term:`vertex` of the `surf_mesh`.
        The value at each :term:`vertex` one inside the ROI and zero inside
        ROI, or an integer giving the label number for atlases.
        If None is passed for ``surf_mesh`` then ``roi_map`` must be a
        :obj:`~nilearn.surface.SurfaceImage` instance and its the mesh will be
        used for plotting.

        When specified `roi_map` is of type :class:`numpy.ndarray`, to have a
        correct view, `hemi` should have a value corresponding to `roi_map`
        data.

    %(hemi)s
        It is only used if ``roi_map`` is :obj:`~nilearn.surface.SurfaceImage`
        and / or ``surf_mesh`` is :obj:`~nilearn.surface.PolyMesh`.
        Otherwise a warning will be displayed.

        .. versionadded:: 0.11.0

    levels : :obj:`list` of :obj:`int`, or None, default=None
        A list of indices of the regions that are to be outlined.
        Every index needs to correspond to one index in ``roi_map``.
        If `None`, all regions in ``roi_map`` are used.

    labels : :obj:`list` of :obj:`str` or None, or None, default=None
        A list of labels for the individual regions of interest.
        Provide `None` as list entry to skip showing the label of that region.
        If `None`, no labels are used.

    colors : :obj:`list` of matplotlib color names or RGBA values, or None,
        default=None
        Colors to be used.

    legend : :obj:`bool`,  default=False
        Whether to plot a legend of region's labels.

    %(cmap)s
        Default='tab20'.

    %(title)s

    %(output_file)s

    axes : instance of matplotlib axes or None, default=None
        The axes instance to plot to. The projection must be `"3d"` (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': "3d"})`,
        where axes should be passed.).
        If `None`, uses axes from figure if available, else creates new axes.

    %(figure)s

    kwargs : extra keyword arguments, optional
        Extra keyword arguments passed to :func:`~nilearn.plotting.plot_surf`.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf_stat_map : for plotting statistical maps on
        brain surfaces.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.
    """
    roi_map, surf_mesh, _ = check_surface_plotting_inputs(
        roi_map, surf_mesh, hemi, map_var_name="roi_map"
    )
    check_extensions(roi_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

    if "darkness" not in kwargs:
        # TODO (nilearn >= 0.13.0) remove when darkness is removed
        # added now to prevent some extra warnings
        kwargs["darkness"] = None

    backend = get_surface_backend(DEFAULT_ENGINE)
    fig = backend._plot_surf_contours(
        surf_mesh=surf_mesh,
        roi_map=roi_map,
        hemi=hemi,
        levels=levels,
        labels=labels,
        colors=colors,
        legend=legend,
        cmap=cmap,
        title=title,
        output_file=output_file,
        axes=axes,
        figure=figure,
        **kwargs,
    )
    return fig


@fill_doc
def plot_surf_stat_map(
    surf_mesh=None,
    stat_map=None,
    bg_map=None,
    hemi=DEFAULT_HEMI,
    view=None,
    engine=DEFAULT_ENGINE,
    cmap=DEFAULT_DIVERGING_CMAP,
    colorbar=True,
    avg_method=None,
    threshold=None,
    alpha=None,
    bg_on_data=False,
    darkness=0.7,
    vmin=None,
    vmax=None,
    symmetric_cbar="auto",
    cbar_tick_format="auto",
    title=None,
    title_font_size=None,
    output_file=None,
    axes=None,
    figure=None,
    **kwargs,
):
    """Plot a stats map on a surface :term:`mesh` with optional background.

    .. versionadded:: 0.3

    Parameters
    ----------
    %(surf_mesh)s
        If None is passed, then ``stat_map`` must be a
        :obj:`~nilearn.surface.SurfaceImage` instance and the mesh from
        that :obj:`~nilearn.surface.SurfaceImage` instance will be used.

    stat_map : :obj:`str` or :class:`numpy.ndarray` or None, default=None
        Statistical map to be displayed on the surface :term:`mesh`,
        can be a file
        (valid formats are .gii, .mgz, or
        Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label) or
        a Numpy array with a value for each :term:`vertex` of the `surf_mesh`.
        If None is passed for ``surf_mesh``
        then ``stat_map``
        must be a :obj:`~nilearn.surface.SurfaceImage` instance
        and its the mesh will be used for plotting.

        When specified `surf_map` is of type :class:`numpy.ndarray`, to have a
        correct view, `hemi` should have a value corresponding to `surf_map`
        data.

    %(bg_map)s

    %(hemi)s

    %(view)s

    engine : {'matplotlib', 'plotly'}, default='matplotlib'

        .. versionadded:: 0.9.0

        Selects which plotting engine will be used by ``plot_surf_stat_map``.
        Currently, only ``matplotlib`` and ``plotly`` are supported.

        .. note::
            To use the ``plotly`` engine you need to
            have ``plotly`` installed.

        .. note::
            To be able to save figures to disk with the ``plotly``
            engine you need to have ``kaleido`` installed.

        .. warning::
            The ``plotly`` engine is new and experimental.
            Please report bugs that you may encounter.


    %(cmap)s
        default="RdBu_r"

    %(colorbar)s

        .. note::
            This function uses a symmetric colorbar for the statistical map.

        Default=True.

    %(avg_method)s

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

        When using matplotlib as engine,
        `avg_method` will default to ``"mean"`` if ``None`` is passed.

        .. versionadded:: 0.10.3

    %(threshold)s
        Default=None

    alpha : :obj:`float` or 'auto' or None, default=None
        Alpha level of the :term:`mesh` (not the stat_map).
        Will default to ``"auto"`` if ``None`` is passed.
        If 'auto' is chosen, alpha will default to .5 when no bg_map is
        passed and to 1 if a bg_map is passed.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(bg_on_data)s

    %(darkness)s
        Default=1.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(vmin)s

    %(vmax)s

    %(symmetric_cbar)s

    %(cbar_tick_format)s
        Default="auto" which will select:

            - '%%.2g' (scientific notation) with ``matplotlib`` engine.
            - '.1f' (rounded floats) with ``plotly`` engine.

        .. versionadded:: 0.7.1

    %(title)s

    title_font_size : :obj:`int`, default=None
        Size of the title font (only implemented for the plotly engine).

        .. versionadded:: 0.9.0

    %(output_file)s

    axes : instance of matplotlib axes or None, default=None
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': '3d'})`,
        where axes should be passed.).
        If None, a new axes is created.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(figure)s

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    kwargs : :obj:`dict`, optional
        Keyword arguments passed to :func:`nilearn.plotting.plot_surf`.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage: For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf: For brain surface visualization.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.
    """
    check_params(locals())
    check_threshold_not_negative(threshold)

    stat_map, surf_mesh, bg_map = check_surface_plotting_inputs(
        stat_map, surf_mesh, hemi, bg_map, map_var_name="stat_map"
    )
    check_extensions(stat_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)
    loaded_stat_map = load_surf_data(stat_map)

    backend = get_surface_backend(engine)
    # derive symmetric vmin, vmax and colorbar limits depending on
    # symmetric_cbar settings
    cbar_vmin, cbar_vmax, vmin, vmax = (
        backend._adjust_colorbar_and_data_ranges(
            loaded_stat_map,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=symmetric_cbar,
        )
    )

    fig = plot_surf(
        surf_mesh,
        surf_map=loaded_stat_map,
        bg_map=bg_map,
        hemi=hemi,
        view=view,
        engine=engine,
        cmap=cmap,
        colorbar=colorbar,
        avg_method=avg_method,
        threshold=threshold,
        alpha=alpha,
        bg_on_data=bg_on_data,
        darkness=darkness,
        vmin=vmin,
        vmax=vmax,
        cbar_vmin=cbar_vmin,
        cbar_vmax=cbar_vmax,
        cbar_tick_format=cbar_tick_format,
        title=title,
        title_font_size=title_font_size,
        output_file=output_file,
        axes=axes,
        figure=figure,
        **kwargs,
    )
    return fig


@fill_doc
def plot_img_on_surf(
    stat_map,
    surf_mesh="fsaverage5",
    mask_img=None,
    hemispheres=None,
    views=None,
    cmap=DEFAULT_DIVERGING_CMAP,
    colorbar=True,
    threshold=None,
    bg_on_data=False,
    inflate=False,
    vmin=None,
    vmax=None,
    symmetric_cbar="auto",
    cbar_tick_format="%i",
    title=None,
    output_file=None,
    **kwargs,
):
    """Plot multiple views of plot_surf_stat_map \
    in a single figure.

    It projects stat_map into meshes and plots views of
    left and right hemispheres. The *views* argument defines the views
    that are shown. This function returns the fig, axes elements from
    matplotlib unless kwargs sets and output_file, in which case nothing
    is returned.

    Parameters
    ----------
    stat_map : :obj:`str` or :class:`pathlib.Path` or 3D Niimg-like object
        See :ref:`extracting_data`.

    surf_mesh : :obj:`str`, :obj:`dict`, or None, default='fsaverage5'
        If str, either one of the two:
        'fsaverage5': the low-resolution fsaverage5 :term:`mesh` (10242 nodes)
        'fsaverage': the high-resolution fsaverage :term:`mesh` (163842 nodes)
        If dict, a dictionary with keys: ['infl_left', 'infl_right',
        'pial_left', 'pial_right', 'sulc_left', 'sulc_right'], where
        values are surface :term:`mesh` geometries as accepted
        by plot_surf_stat_map.

    mask_img : Niimg-like object or None, default=None
        The mask is passed to vol_to_surf.
        Samples falling out of this mask or out of the image are ignored
        during projection of the volume to the surface.
        If ``None``, don't apply any mask.

    hemispheres : :obj:`list` of :obj:`str`, default=None
        Hemispheres to display.
        Will default to ``['left', 'right']`` if ``None`` or "both" is passed.

    views : :obj:`list` of :obj:`str`, default=None
        A list containing all views to display.
        The montage will contain as many rows as views specified by
        display mode. Order is preserved, and left and right hemispheres
        are shown on the left and right sides of the figure.
        Will default to ``['lateral', 'medial']`` if ``None`` is passed.

    %(cmap)s
        Default="RdBu_r".

    %(colorbar)s

        .. note::
            This function uses a symmetric colorbar for the statistical map.

        Default=True.

    %(threshold)s
        Default=None

    %(bg_on_data)s

    inflate : :obj:`bool`, default=False
        If True, display images in inflated brain.
        If False, display images in pial surface.

    %(vmin)s

    %(vmax)s

    %(symmetric_cbar)s

    %(cbar_tick_format)s

    %(title)s

    %(output_file)s

    kwargs : :obj:`dict`, optional
        keyword arguments passed to plot_surf_stat_map.

        .. note::
            Parameters "figure", "axes", and "engine" which are valid for
            ``plot_surf_stat_map`` are not valid for ``plot_img_on_surf``.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as the default background map for this plotting function.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.

    nilearn.plotting.plot_surf_stat_map : For info on kwargs options
        accepted by plot_img_on_surf.

    """
    check_params(locals())
    check_threshold_not_negative(threshold)

    for arg in ("figure", "axes", "engine"):
        if arg in kwargs:
            raise ValueError(
                f"plot_img_on_surf does not accept {arg} as an argument"
            )

    if hemispheres in (None, "both", ["both"]):
        hemispheres = ["left", "right"]
    if not isinstance(hemispheres, list):
        hemispheres = [hemispheres]
    hemis = check_hemispheres(hemispheres)

    if views is None:
        views = ["lateral", "medial"]
    modes = check_views(views)

    stat_map = check_niimg_3d(stat_map, dtype="auto")
    surf_mesh = check_mesh_is_fsaverage(surf_mesh)

    mesh_prefix = "infl" if inflate else "pial"
    surf = {
        "left": surf_mesh[f"{mesh_prefix}_left"],
        "right": surf_mesh[f"{mesh_prefix}_right"],
    }
    texture = {
        "left": vol_to_surf(
            stat_map, surf_mesh["pial_left"], mask_img=mask_img
        ),
        "right": vol_to_surf(
            stat_map, surf_mesh["pial_right"], mask_img=mask_img
        ),
    }

    backend = get_surface_backend(DEFAULT_ENGINE)
    # get vmin and vmax for entire data (all hemis)
    _, _, vmin, vmax = backend._adjust_colorbar_and_data_ranges(
        get_data(stat_map),
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
    )

    fig = backend._plot_img_on_surf(
        surf,
        surf_mesh=surf_mesh,
        stat_map=stat_map,
        texture=texture,
        hemis=hemis,
        modes=modes,
        bg_on_data=bg_on_data,
        inflate=inflate,
        output_file=output_file,
        title=title,
        colorbar=colorbar,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        symmetric_cbar=symmetric_cbar,
        cmap=cmap,
        cbar_tick_format=cbar_tick_format,
        **kwargs,
    )
    return fig


@fill_doc
def plot_surf_roi(
    surf_mesh=None,
    roi_map=None,
    bg_map=None,
    hemi=DEFAULT_HEMI,
    view=None,
    engine=DEFAULT_ENGINE,
    cmap="gist_ncar",
    colorbar=True,
    avg_method=None,
    threshold=None,
    alpha=None,
    bg_on_data=False,
    darkness=0.7,
    vmin=None,
    vmax=None,
    cbar_tick_format="auto",
    title=None,
    title_font_size=None,
    output_file=None,
    axes=None,
    figure=None,
    **kwargs,
):
    """Plot ROI on a surface :term:`mesh` with optional background.

    .. versionadded:: 0.3

    Parameters
    ----------
    %(surf_mesh)s
        If None is passed, then ``roi_map`` must be a
        :obj:`~nilearn.surface.SurfaceImage` instance and the mesh from that
        :obj:`~nilearn.surface.SurfaceImage` instance will be used.

    roi_map : :obj:`str` or :class:`numpy.ndarray` or \
              :obj:`list` of :class:`numpy.ndarray` or \
              :obj:`~nilearn.surface.SurfaceImage` or None, \
              default=None
        ROI map to be displayed on the surface :term:`mesh`,
        can be a file
        (valid formats are .gii, .mgz, or
        Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label) or
        a Numpy array with a value for each :term:`vertex` of the `surf_mesh`
        or a :obj:`~nilearn.surface.SurfaceImage` instance.
        The value at each vertex one inside the ROI and zero inside ROI, or an
        integer giving the label number for atlases.
        If None is passed for ``surf_mesh``
        then ``roi_map``
        must be a :obj:`~nilearn.surface.SurfaceImage` instance
        and its the mesh will be used for plotting.

        When specified `roi_map` is of type :class:`numpy.ndarray`, to have a
        correct view, `hemi` should have a value corresponding to `roi_map`
        data.

    %(bg_map)s

    %(hemi)s

    %(view)s

    engine : {'matplotlib', 'plotly'}, default='matplotlib'

        .. versionadded:: 0.9.0

        Selects which plotting engine will be used by ``plot_surf_roi``.
        Currently, only ``matplotlib`` and ``plotly`` are supported.

        .. note::
            To use the ``plotly`` engine you need to have
            ``plotly`` installed.

        .. note::
            To be able to save figures to disk with ``plotly`` engine
            you need to have ``kaleido`` installed.

        .. warning::
            The ``plotly`` engine is new and experimental.
            Please report bugs that you may encounter.

    %(cmap_lut)s
        Default='gist_ncar'.

    %(colorbar)s
        Default=True

    %(avg_method)s

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

        When using matplotlib as engine,
        `avg_method` will default to ``"median"`` if ``None`` is passed.

    %(threshold)s
        Default=None

        .. note::
            By default, the regions that are labeled 0 are not thresholded.
            Threshold should be set to a very small number, ex. 1e-14 to
            threshold the those 0 labeled regions.

    alpha : :obj:`float` or 'auto' or None, default=None
        Alpha level of the :term:`mesh` (not surf_data).
        When using matplotlib as engine,
        `alpha` will default to ``"auto"`` if ``None`` is passed.
        If 'auto' is chosen, alpha will default to 0.5 when no bg_map
        is passed and to 1 if a bg_map is passed.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(bg_on_data)s

    %(darkness)s
        Default=1.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(vmin)s

    %(vmax)s

    %(cbar_tick_format)s
        Default="auto" which defaults to integers format:

            - "%%i" for ``matplotlib`` engine.
            - "." for ``plotly`` engine.

        .. versionadded:: 0.7.1

    %(title)s

    title_font_size : :obj:`int`, default=None
        Size of the title font (only implemented for the plotly engine).

        .. versionadded:: 0.9.0

    %(output_file)s

    axes : Axes instance or None, default=None
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `plt.subplots(subplot_kw={'projection': '3d'})`).
        If None, a new axes is created.

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    %(figure)s

        .. note::
            This option is currently only implemented for the
            ``matplotlib`` engine.

    kwargs : :obj:`dict`, optional
        Keyword arguments passed to :func:`nilearn.plotting.plot_surf`.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage: For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf: For brain surface visualization.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.
    """
    check_params(locals())
    check_threshold_not_negative(threshold)
    roi_map, surf_mesh, bg_map = check_surface_plotting_inputs(
        roi_map, surf_mesh, hemi, bg_map
    )
    check_extensions(roi_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

    # preload roi and mesh to determine vmin, vmax and give more useful
    # error messages in case of wrong inputs
    roi = load_surf_data(roi_map)

    if roi.ndim != 1:
        raise ValueError(
            "roi_map can only have one dimension but has "
            f"{roi.ndim} dimensions"
        )
    if (roi < 0).any():
        # TODO (nilearn >= 0.13.0) raise ValueError
        warn(
            (
                "Negative values in roi_map will no longer be allowed in"
                " Nilearn version 0.13"
            ),
            DeprecationWarning,
            stacklevel=find_stack_level(),
        )

    mesh = load_surf_mesh(surf_mesh)
    if roi.shape[0] != mesh.n_vertices:
        raise ValueError(
            "roi_map does not have the same number of vertices "
            "as the mesh. If you have a list of indices for the "
            "ROI you can convert them into a ROI map like this:\n"
            "roi_map = np.zeros(n_vertices)\n"
            "roi_map[roi_idx] = 1"
        )

    idx_not_na = ~np.isnan(roi)
    if vmin is None:
        vmin = float(np.nanmin(roi))
    if vmax is None:
        vmax = float(1 + np.nanmax(roi))

    if not np.array_equal(roi[idx_not_na], roi[idx_not_na].astype(int)):
        # TODO (nilearn >= 0.13.0) raise ValueError
        warn(
            (
                "Non-integer values in roi_map will no longer be allowed "
                "in Nilearn version 0.13"
            ),
            DeprecationWarning,
            stacklevel=find_stack_level(),
        )
    if isinstance(cmap, pd.DataFrame):
        cmap = create_colormap_from_lut(cmap)

    params = {
        "avg_method": avg_method,
        "cbar_tick_format": cbar_tick_format,
    }

    backend = get_surface_backend(engine)
    backend._adjust_plot_roi_params(params)

    fig = backend._plot_surf(
        mesh,
        surf_map=roi,
        bg_map=bg_map,
        hemi=hemi,
        view=view,
        cmap=cmap,
        colorbar=colorbar,
        avg_method=params["avg_method"],
        threshold=threshold,
        alpha=alpha,
        bg_on_data=bg_on_data,
        darkness=darkness,
        vmin=vmin,
        vmax=vmax,
        cbar_tick_format=params["cbar_tick_format"],
        title=title,
        title_font_size=title_font_size,
        output_file=output_file,
        axes=axes,
        figure=figure,
        **kwargs,
    )
    return fig
