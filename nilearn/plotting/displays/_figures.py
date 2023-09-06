import numpy as np
from scipy.spatial import distance_matrix

from nilearn.surface.surface import load_surf_data


class SurfaceFigure:
    """Abstract class for surface figures.

    Parameters
    ----------
    figure : Figure instance or ``None``, optional
        Figure to be wrapped.

    output_file : :obj:`str` or ``None``, optional
        Path to output file.
    """

    def __init__(self, figure=None, output_file=None):
        self.figure = figure
        self.output_file = output_file

    def show(self):
        """Show the figure."""
        raise NotImplementedError

    def _check_output_file(self, output_file=None):
        """If an output file is provided, \
        set it as the new default output file.

        Parameters
        ----------
        output_file : :obj:`str` or ``None``, optional
            Path to output file.
        """
        if output_file is None:
            if self.output_file is None:
                raise ValueError(
                    "You must provide an output file "
                    "name to save the figure."
                )
        else:
            self.output_file = output_file

    def add_contours(self):
        """Draw boundaries around roi."""
        raise NotImplementedError


class PlotlySurfaceFigure(SurfaceFigure):
    """Implementation of a surface figure obtained with `plotly` engine.

    Parameters
    ----------
    figure : Plotly figure instance or ``None``, optional
        Plotly figure instance to be used.

    output_file : :obj:`str` or ``None``, optional
        Output file path.

    Attributes
    ----------
    figure : Plotly figure instance
        Plotly figure. Use this attribute to access the underlying
        plotly figure for further customization and use plotly
        functionality.

    output_file : :obj:`str`
        Output file path.

    """

    def __init__(self, figure=None, output_file=None):
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "Plotly is required to use `PlotlySurfaceFigure`."
            )
        if figure is not None and not isinstance(figure, go.Figure):
            raise TypeError(
                "`PlotlySurfaceFigure` accepts only plotly figure objects."
            )
        super().__init__(figure=figure, output_file=output_file)

    def show(self, renderer="browser"):
        """Show the figure.

        Parameters
        ----------
        renderer : :obj:`str`, optional
            Plotly renderer to be used.
            Default='browser'.
        """
        if self.figure is not None:
            self.figure.show(renderer=renderer)
            return self.figure

    def savefig(self, output_file=None):
        """Save the figure to file.

        Parameters
        ----------
        output_file : :obj:`str` or ``None``, optional
            Path to output file.
        """
        try:
            import kaleido  # noqa: F401
        except ImportError:
            raise ImportError(
                "`kaleido` is required to save plotly figures to disk."
            )
        self._check_output_file(output_file=output_file)
        if self.figure is not None:
            self.figure.write_image(self.output_file)

    def add_contours(self, roi_map, levels=None, labels=None, lines=None):
        """
        Draw boundaries around roi.

        Parameters
        ----------
        roi_map : str or :class:`numpy.ndarray` or list of
            :class:`numpy.ndarray` ROI map to be displayed on the surface
            mesh, can be a file (valid formats are .gii, .mgz, .nii,
            .nii.gz, or Freesurfer specific files such as .annot or .label),
            or a Numpy array with a value for each vertex of the surf_mesh.
            The value at each vertex one inside the ROI and zero inside ROI,
            or an integer giving the label number for atlases.

        levels : list of integers, or None, default=None
            A list of indices of the regions that are to be outlined.
            Every index needs to correspond to one index in roi_map.
            If None, all regions in roi_map are used.

        labels : list of strings or None, or None, optional
            A list of labels for the individual regions of interest. Provide
            None as list entry to skip showing the label of that region. If
            None, no labels are used.

        lines : list of dict giving the properties of the contours, or None,
            optional. For valid keys, see
            :attr:`plotly.graph_objects.Scatter3d.line`. If length 1, the
            properties defined in that element will be used to draw all
            requested contours.
        """
        import plotly.graph_objects as go

        if levels is None:
            levels = np.unique(roi_map)
        if labels is None:
            labels = [f"Region {i}" for i, _ in enumerate(levels)]
        if lines is None:
            lines = [None] * len(levels)
        elif len(lines) == 1 and len(levels) > 1:
            lines *= len(levels)
        if not (len(levels) == len(labels)):
            raise ValueError(
                "levels and labels need to be either the same length or None."
            )
        if not (len(levels) == len(lines)):
            raise ValueError(
                "levels and lines need to be either the same length or None."
            )
        roi = load_surf_data(roi_map)

        traces = []
        for level, label, line in zip(levels, labels, lines):
            parc_idx = np.where(roi == level)[0]
            sorted_vertices = self._get_vertices_on_edge(parc_idx)
            traces.append(
                go.Scatter3d(
                    x=sorted_vertices[:, 0],
                    y=sorted_vertices[:, 1],
                    z=sorted_vertices[:, 2],
                    mode="lines",
                    line=line,
                    name=label,
                )
            )
        self.figure.add_traces(data=traces)

    def _get_vertices_on_edge(self, parc_idx):
        """
        Identify which vertices lie on the outer edge of a parcellation.

        Parameters
        ----------
        parc_idx : numpy.ndarray, indices of the vertices of the region to be
            plotted.

        Returns
        -------
        data : :class:`numpy.ndarray` (n_vertices, s) x,y,z coordinates of
            vertices that trace region of interest.

        """
        faces = np.vstack(
            [self.figure._data[0].get(d) for d in ["i", "j", "k"]]
        ).T

        # count how many vertices belong to the given parcellation in each face
        verts_per_face = np.isin(faces, parc_idx).sum(axis=1)

        # test if parcellation forms regions
        if np.all(verts_per_face < 2):
            raise ValueError("Vertices in parcellation do not form region.")

        vertices_on_edge = np.intersect1d(
            np.unique(faces[verts_per_face == 2]), parc_idx
        )

        # now that we know where to draw the lines, we need to know in which
        # order. If we pick a vertex to start and move to the closest one, and
        # then to the closest remaining one and so forth, we should get the
        # whole ROI
        coords = np.vstack(
            [self.figure._data[0].get(d) for d in ["x", "y", "z"]]
        ).T
        vertices = coords[vertices_on_edge]

        # Start with the first vertex
        current_vertex = 0
        visited_vertices = {current_vertex}

        sorted_vertices = [vertices[0]]

        # Loop over the remaining vertices in order of distance from the
        # current vertex
        while len(visited_vertices) < len(vertices):
            remaining_vertices = np.array(
                [
                    vertex
                    for vertex in range(len(vertices))
                    if vertex not in visited_vertices
                ]
            )
            remaining_distances = distance_matrix(
                vertices[current_vertex].reshape(1, -1),
                vertices[remaining_vertices],
            )
            closest_index = np.argmin(remaining_distances)
            closest_vertex = remaining_vertices[closest_index]
            visited_vertices.add(closest_vertex)
            sorted_vertices.append(vertices[closest_vertex])
            # Move to the closest vertex and repeat the process
            current_vertex = closest_vertex

        # at the end we append the first one again to close the outline
        sorted_vertices.append(vertices[0])
        sorted_vertices = np.asarray(sorted_vertices)

        return sorted_vertices
