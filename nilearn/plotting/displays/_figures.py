import warnings

import numpy as np
from scipy import linalg
from scipy.spatial import distance_matrix

from nilearn._utils.helpers import is_kaleido_installed, is_plotly_installed
from nilearn.surface import SurfaceImage
from nilearn.surface.surface import load_surf_data

if is_plotly_installed():
    import plotly.graph_objects as go


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

    @property
    def _faces(self):
        return np.vstack(
            [self.figure._data[0].get(d) for d in ["i", "j", "k"]]
        ).T

    @property
    def _coords(self):
        return np.vstack(
            [self.figure._data[0].get(d) for d in ["x", "y", "z"]]
        ).T

    def __init__(self, figure=None, output_file=None):
        if not is_plotly_installed():
            raise ImportError(
                "Plotly is required to use `PlotlySurfaceFigure`."
            )
        import plotly.graph_objects as go

        if figure is not None and not isinstance(figure, go.Figure):
            raise TypeError(
                "`PlotlySurfaceFigure` accepts only plotly figure objects."
            )
        super().__init__(figure=figure, output_file=output_file)

    def show(self, renderer="browser"):
        """Show the figure.

        Parameters
        ----------
        renderer : :obj:`str`, default='browser'
            Plotly renderer to be used.

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
        if not is_kaleido_installed():
            raise ImportError(
                "`kaleido` is required to save plotly figures to disk."
            )
        self._check_output_file(output_file=output_file)
        if self.figure is not None:
            self.figure.write_image(self.output_file)

    def add_contours(
        self,
        roi_map,
        levels=None,
        labels=None,
        lines=None,
        elevation=0.1,
        hemi="left",
    ):
        """Draw boundaries around roi.

        Parameters
        ----------
        roi_map : :obj:`str` or :class:`numpy.ndarray` or :obj:`list` of \
                  :class:`numpy.ndarray` or\
                  :obj:`~nilearn.surface.SurfaceImage`
            ROI map to be displayed on the surface
            mesh, can be a file (valid formats are .gii, .mgz, .nii,
            .nii.gz, or FreeSurfer specific files such as .annot or .label),
            or a Numpy array with a value for each vertex of the surf_mesh.
            The value at each vertex is one inside the ROI and zero outside
            the ROI, or an :obj:`int` giving the label number for atlases.

        levels : :obj:`list` of :obj:`int`, or :obj:`None`, default=None
            A :obj:`list` of indices of the regions that are to be outlined.
            Every index needs to correspond to one index in roi_map.
            If :obj:`None`, all regions in roi_map are used.

        labels : :obj:`list` of :obj:`str` or :obj:`None`, default=None
            A :obj:`list` of labels for the individual regions of interest.
            Provide :obj:`None` as list entry to skip showing the label of
            that region. If :obj:`None`, no labels are used.

        lines : :obj:`list` of :obj:`dict` giving the properties of the \
                contours, or :obj:`None`, default=None
            For valid keys, see :attr:`plotly.graph_objects.Scatter3d.line`.
            If length 1, the properties defined in that element will be used
            to draw all requested contours.

        elevations : :obj:`float`, default=0.1
            Controls how high above the face each boundary should be placed.
            0.0 implies directly on boundary, and higher values are farther
            above the face. This is useful for avoiding overlap of surface
            and boundary.

        Warnings
        --------
            Warns when a vertex is isolated; it will not be included in the
            roi contour.

        Notes
        -----
            Regions are traced by connecting the centroids of non-isolated
            faces (triangles).
        """
        if isinstance(roi_map, SurfaceImage):
            assert len(roi_map.shape) == 1 or roi_map.shape[1] == 1
            roi_map = roi_map.data.parts[hemi]

        if levels is None:
            levels = np.unique(roi_map)
        if labels is None:
            labels = [f"Region {i}" for i, _ in enumerate(levels)]
        if lines is None:
            lines = [None] * len(levels)
        elif len(lines) == 1 and len(levels) > 1:
            lines *= len(levels)
        if len(levels) != len(labels):
            raise ValueError(
                "levels and labels need to be either the same length or None."
            )
        if len(levels) != len(lines):
            raise ValueError(
                "levels and lines need to be either the same length or None."
            )

        roi = load_surf_data(roi_map)

        traces = []
        for level, label, line in zip(levels, labels, lines):
            parc_idx = np.where(roi == level)[0]

            # warn when the edge faces exclude vertices in parcellation
            # a vertex is isolated when it is a vertex of 6 faces that
            # each have only on vertex in parcellation
            verts_per_face = np.isin(self._faces, parc_idx).sum(axis=1)
            faces_w_one_v = np.flatnonzero(verts_per_face == 1)
            unique_v, unique_v_counts = np.unique(
                self._faces[faces_w_one_v], return_counts=True
            )
            isolated_v = unique_v[unique_v_counts == 6]
            if any(isolated_v):
                warnings.warn(
                    f"""{label=} contains isolated vertices:
                    {isolated_v.tolist()}. These will not be included in ROI
                    boundary line.""",
                    stacklevel=2,
                )

            sorted_vertices = self._get_sorted_edge_centroids(
                parc_idx=parc_idx, elevation=elevation
            )

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

    def _get_sorted_edge_centroids(self, parc_idx, elevation=0.1):
        """Identify which vertices lie on the outer edge of a parcellation.

        Parameters
        ----------
        parc_idx : :class:`numpy.ndarray`
            Indices of the vertices of the region to be plotted.

        elevation : :obj:`float`
            Controls how high above the face each centroid should be placed.
            0.0 implies directly on boundary, and higher values are farther
            above the face. This is useful for avoiding overlap of surface
            and boundary.

        Returns
        -------
        sorted_vertices : :class:`numpy.ndarray`
             (n_vertices, s) x,y,z coordinates of vertices that trace region
             of interest.

        Notes
        -----
        For each face on the edge of a region
            1. Get a centroid for each face (parallel to the triangle plane)
            2. Find the xyz coordinate that is normal to the triangle face
                (at distance `elevation`)
            3. Arrange the centroids such in a good order for plotting
        """
        # Mask indicating faces whose centroids will compose the boundary.
        edge_faces = self._get_faces_on_edge(parc_idx=parc_idx)

        # gather the centroids of each face
        centroids = []
        segments = []
        vs = []
        idxs = []
        for e, face in zip(edge_faces, self._faces):
            if e:
                t0 = self._coords[face[0]]
                t1 = self._coords[face[1]]
                t2 = self._coords[face[2]]

                # the xyz coordinate is weighted toward the roi boundary (2:1)
                w0 = 2 if face[0] in parc_idx else 1
                w1 = 2 if face[1] in parc_idx else 1
                w2 = 2 if face[2] in parc_idx else 1
                x = np.average((t0[0], t1[0], t2[0]), weights=(w0, w1, w2))
                y = np.average((t0[1], t1[1], t2[1]), weights=(w0, w1, w2))
                z = np.average((t0[2], t1[2], t2[2]), weights=(w0, w1, w2))
                centroids.append(
                    self._project_above_face(
                        np.array((x, y, z)), t0, t1, t2, elevation=elevation
                    )
                )
                segs = [None] * 3
                if face[0] in parc_idx and face[1] in parc_idx:
                    segs[0] = self._transform_coord_to_plane(
                        t0, t0, t1, t2
                    ) + self._transform_coord_to_plane(t1, t0, t1, t2)
                if face[0] in parc_idx and face[2] in parc_idx:
                    segs[1] = self._transform_coord_to_plane(
                        t0, t0, t1, t2
                    ) + self._transform_coord_to_plane(t2, t0, t1, t2)
                if face[1] in parc_idx and face[2] in parc_idx:
                    segs[2] = self._transform_coord_to_plane(
                        t2, t0, t1, t2
                    ) + self._transform_coord_to_plane(t1, t0, t1, t2)
                segments.append(tuple(segs))
                vs.append((t0, t1, t2))
                idxs.append([f for f in face if f in parc_idx])

        centroids = np.array(centroids)

        # Next, sort centroids along boundary
        # Start with the first vertex
        current_vertex = 0
        visited_vertices = {current_vertex}
        last_distance = np.inf
        prev_first = 0

        sorted_vertices = [centroids[0]]

        # Loop over the remaining vertices in order of distance from the
        # current vertex
        for _ in range(1, len(centroids)):
            remaining_vertices = np.array(
                [
                    vertex
                    for vertex in range(len(centroids))
                    if vertex not in visited_vertices
                ]
            )
            remaining_distances = distance_matrix(
                centroids[current_vertex].reshape(1, -1),
                centroids[remaining_vertices],
            )
            # Occasionally, the next closest centroid is one that would
            # cause a loop. This is common when a vertex is a neighbor
            # of only one other vertex in the roi (the loop encircles
            # this corner vertex). So, the next added centroid is one
            # that may be slightly farther away -- if the one that is
            # farther away has fewer vertices within the roi.

            # from the current vertex, there are only at most 5 options
            # that will be good jumps
            n_jumps_remaining = min(5, max(0, len(remaining_vertices) - 1))
            smallest_idx = np.argpartition(
                remaining_distances.squeeze(), n_jumps_remaining
            )[:n_jumps_remaining]
            xy1 = self._transform_coord_to_plane(
                centroids[current_vertex], *vs[current_vertex]
            )
            next_index = -1
            for attempt in np.argsort(remaining_distances[0, smallest_idx]):
                fail = False
                shortest_idx = smallest_idx[attempt]
                xy2 = self._transform_coord_to_plane(
                    centroids[remaining_vertices[shortest_idx]],
                    *vs[current_vertex],
                )
                if all(
                    v not in idxs[current_vertex]
                    for v in idxs[remaining_vertices[shortest_idx]]
                ):
                    # this does not share vertex, so try again
                    fail |= True
                if fail:
                    continue
                # also need to test for whether an edge is shared
                shared = 0
                for v in vs[remaining_vertices[shortest_idx]]:
                    for v2 in vs[current_vertex]:
                        shared += np.all(np.isclose(v, v2))
                if shared < 2:
                    # this does not share and edge, so try again
                    continue
                for e in segments[current_vertex]:
                    if e is not None and self._do_segs_intersect(
                        *xy1, *xy2, *e
                    ):
                        # this one crosses boundary, so try again
                        fail |= True
                if fail:
                    continue
                next_index = shortest_idx

                # if none of those five worked, then just pick the next nearest
                if next_index == -1:
                    next_index = np.argmin(remaining_distances)

            closest_vertex = remaining_vertices[next_index]

            # some regions have multiple, non-isolated vertices
            # this block detects that by checking whether the next
            # vertex is very far away
            if remaining_distances[0, next_index] > last_distance * 3:
                # close the current contour
                # add triple of None, which is parsed by plotly
                # as a signal to start a new closed contour
                sorted_vertices.extend(
                    (centroids[prev_first], np.array([None] * 3))
                )
                # start the new contour
                prev_first = closest_vertex

            visited_vertices.add(closest_vertex)
            sorted_vertices.append(centroids[closest_vertex])

            # Move to the closest vertex and repeat the process
            current_vertex = closest_vertex
            last_distance = remaining_distances[0, next_index]

        # append the first one again to close the outline
        sorted_vertices.append(centroids[prev_first])

        return np.asarray(sorted_vertices)

    def _get_faces_on_edge(self, parc_idx):
        """Identify which faces lie on the outeredge of the parcellation \
        defined by the indices in parc_idx.

        Parameters
        ----------
        parc_idx : numpy.ndarray, indices of the vertices
            of the region to be plotted
        """
        # count how many vertices belong to the given parcellation in each face
        verts_per_face = np.isin(self._faces, parc_idx).sum(axis=1)

        # test if parcellation forms regions
        if np.all(verts_per_face < 2):
            raise ValueError("Vertices in parcellation do not form region.")

        vertices_on_edge = np.intersect1d(
            np.unique(self._faces[verts_per_face == 2]), parc_idx
        )
        faces_outside_edge = np.isin(self._faces, vertices_on_edge).sum(axis=1)

        return np.logical_and(faces_outside_edge > 0, verts_per_face < 3)

    @staticmethod
    def _project_above_face(point, t0, t1, t2, elevation=0.1):
        """Given 3d coordinates `point`, report coordinates that define \
           the closest point that is `elevation` above (normal to) \
           the plane defined by vertices `t0`, `t1`, and `t2`.
        """
        u = t1 - t0
        v = t2 - t0
        # vector normal to plane
        n = np.cross(u, v)
        n /= np.linalg.norm(n)
        p_ = point - t0

        p_normal = np.dot(p_, n) * n
        p_tangent = p_ - p_normal

        closest_point = p_tangent + t0
        return closest_point + elevation * n

    @staticmethod
    def _do_segs_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
        """Check whether line segments intersect.

        Parameters
        ----------
            x1 : :obj:`float` or :obj:`int`
                First coordinate of first segment beginning.
            y1 : :obj:`float` or :obj:`int`
                Second coordinate of first segment beginning.
            x2 : :obj:`float` or :obj:`int`
                First coordinate of first segment end.
            y2 : :obj:`float` or :obj:`int`
                Second coordinate of first segment end.
            x3 : :obj:`float` or :obj:`int`
                First coordinate of second segment beginning.
            y3 : :obj:`float` or :obj:`int`
                Second coordinate of second segment beginning.
            x4 : :obj:`float` or :obj:`int`
                First coordinate of second segment end.
            y4 : :obj:`float` or :obj:`int`
                Second coordinate of second segment end.

        Returns
        -------
            check: :obj:`bool`
                True if segments intersect, otherwise False

        Notes
        -----
            Implements an algorithm described here
            https://en.wikipedia.org/w/index.php?title=Intersection_(geometry)&oldid=1215046212#Two_line_segments
        """
        a1 = x1 - x2
        b1 = x3 - x4
        c1 = x1 - x3
        a2 = y1 - y2
        b2 = y3 - y4
        c2 = y1 - y3
        d = a1 * b2 - a2 * b1
        if np.isclose(d, 0):
            return False
        t = (c1 * b2 - c2 * b1) / d
        u = (c1 * a2 - c2 * a1) / d
        return 0 <= t <= 1 and 0 <= u <= 1

    @staticmethod
    def _transform_coord_to_plane(v, t0, t1, t2):
        """Given 3d point `v`, find closest point on plane defined \
           by vertices `t0`, `t1`, and `t2`.
        """
        A = linalg.orth(np.column_stack((t1 - t0, t2 - t0)))
        normal = np.cross(A[:, 0], A[:, 1])
        normal /= np.linalg.norm(normal)
        B = np.column_stack((A, normal))
        Bp = np.linalg.inv(B)
        P = B @ np.diag((1, 1, 0)) @ Bp
        return tuple(Bp[:2, :] @ (t0 + P @ (v - t0)))
