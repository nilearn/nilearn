"""
Visualizing 3D stat maps in a Brainsprite viewer
"""
import os
from pathlib import Path

from nilearn.plotting import cm
from nilearn.externals import tempita
from brainsprite import viewer_substitute


def view_img(stat_map_img, bg_img='MNI152',
             cut_coords=None,
             colorbar=True,
             title=None,
             threshold=1e-6,
             annotate=True,
             draw_cross=True,
             black_bg='auto',
             cmap=cm.cold_hot,
             symmetric_cmap=True,
             dim='auto',
             vmax=None,
             vmin=None,
             resampling_interpolation='continuous',
             opacity=1,
             **kwargs
             ):
    """
    Interactive html viewer of a statistical map, with optional background

    Parameters
    ----------
    stat_map_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The statistical map image. Can be either a 3D volume or a 4D volume
        with exactly one time point.
    bg_img : Niimg-like object (default='MNI152')
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the stat map will be plotted on top of.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".
    cut_coords : None, or a tuple of floats (default None)
        The MNI coordinates of the point where the cut is performed
        as a 3-tuple: (x, y, z). If None is given, the cuts are calculated
        automaticaly.
    colorbar : boolean, optional (default True)
        If True, display a colorbar on top of the plots.
    title : string or None (default=None)
        The title displayed on the figure (or None: no title).
    threshold : string, number or None  (default=1e-6)
        If None is given, the image is not thresholded.
        If a string of the form "90%" is given, use the 90-th percentile of
        the absolute value in the image.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        automatically.
    annotate : boolean (default=True)
        If annotate is True, current cuts are added to the viewer.
    draw_cross : boolean (default=True)
        If draw_cross is True, a cross is drawn on the plot to
        indicate the cuts.
    black_bg : boolean (default='auto')
        If True, the background of the image is set to be black.
        Otherwise, a white background is used.
        If set to auto, an educated guess is made to find if the background
        is white or black.
    cmap : matplotlib colormap, optional
        The colormap for specified image.
    symmetric_cmap : bool, optional (default=True)
        True: make colormap symmetric (ranging from -vmax to vmax).
        False: the colormap will go from the minimum of the volume to vmax.
        Set it to False if you are plotting a positive volume, e.g. an atlas
        or an anatomical image.
    dim : float, 'auto' (default='auto')
        Dimming factor applied to background image. By default, automatic
        heuristics are applied based upon the background image intensity.
        Accepted float values, where a typical scan is between -2 and 2
        (-2 = increase constrast; 2 = decrease contrast), but larger values
        can be used for a more pronounced effect. 0 means no dimming.
    vmax : float, or None (default=None)
        max value for mapping colors.
        If vmax is None and symmetric_cmap is True, vmax is the max
        absolute value of the volume.
        If vmax is None and symmetric_cmap is False, vmax is the max
        value of the volume.
    vmin : float, or None (default=None)
        min value for mapping colors.
        If `symmetric_cmap` is `True`, `vmin` is always equal to `-vmax` and
        cannot be chosen.
        If `symmetric_cmap` is `False`, `vmin` defaults to the min of the
        image, or 0 when a threshold is used.
    resampling_interpolation : string, optional (default continuous)
        The interpolation method for resampling.
        Can be 'continuous', 'linear', or 'nearest'.
        See nilearn.image.resample_img
    opacity : float in [0,1] (default 1)
        The level of opacity of the overlay (0: transparent, 1: opaque)

    Returns
    -------
    html_view : the html viewer object.
        It can be saved as an html page `html_view.save_as_html('test.html')`,
        or opened in a browser `html_view.open_in_browser()`.
        If the output is not requested and the current environment is a Jupyter
        notebook, the viewer will be inserted in the notebook.

    See Also
    --------
    nilearn.plotting.plot_stat_map:
        static plot of brain volume, on a single or multiple planes.
    nilearn.plotting.view_connectome:
        interactive 3d view of a connectome.
    nilearn.plotting.view_markers:
        interactive plot of colored markers.
    nilearn.plotting.view_surf, nilearn.plotting.view_img_on_surf:
        interactive view of statistical maps or surface atlases on the cortical
        surface.
    """

    # Load template
    resource_path = Path(__file__).resolve().parent.joinpath("data", "html")
    file_template = resource_path.joinpath("stat_map_template.html")
    tpl = tempita.Template.from_filename(str(file_template), encoding="utf-8")

    # Initialize namespace for substitution
    namespace = {}
    namespace["title"] = title

    js_dir = os.path.join(os.path.dirname(__file__), "data", "js")
    with open(os.path.join(js_dir, "jquery.min.js")) as f:
        namespace["jquery_js"] = f.read()

    # Initialize the template substitution tool
    bsprite = viewer_substitute(
        cut_coords=cut_coords,
        colorbar=colorbar,
        title=title,
        threshold=threshold,
        annotate=annotate,
        draw_cross=draw_cross,
        black_bg=black_bg,
        cmap=cmap,
        symmetric_cmap=symmetric_cmap,
        dim=dim,
        vmax=vmax,
        vmin=vmin,
        resampling_interpolation=resampling_interpolation,
        opacity=opacity,
        base64=True,
    )

    # build sprites and meta-data
    bsprite.fit(stat_map_img, bg_img=bg_img)

    # Populate template
    return bsprite.transform(
        tpl, javascript="js", html="html", library="library", namespace=namespace
    )
