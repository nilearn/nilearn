from nilearn._utils.helpers import is_kaleido_installed, is_plotly_installed


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
