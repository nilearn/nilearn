"""Backports for matplotlib compatibility across versions"""


def cbar_outline_get_xy(cbar_outline):
    """In the matplotlib versions >= 1.4.0, ColorbarBase.outline is a
    Polygon(Patch) object instead of a Line2D(Line) object. This entails
    different getters and setters.

    Change specifically after commit 48f594c2e2b05839ea394040b06196f39d9fbfba,
    entitled
    "changed colorbar outline from a Line2D object to a Polygon object"
    from August 28th, 2013.

    This function unifies getters and setters of ColorbarBase outline xy
    coordinates."""

    if hasattr(cbar_outline, "get_xy"):
        # loose version >= 1.4.x
        return cbar_outline.get_xy()
    else:
        return cbar_outline.get_xydata()


def cbar_outline_set_xy(cbar_outline, xy):
    """Setter for ColorbarBase.outline xy coordinates.
    See cbar_outline_get_xy for more information.
    """

    if hasattr(cbar_outline, "set_xy"):
        # loose version >= 1.4.x
        return cbar_outline.set_xy(xy)
    else:
        cbar_outline.set_xdata(xy[:, 0])
        cbar_outline.set_ydata(xy[:, 1])
