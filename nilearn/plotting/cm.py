# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Matplotlib colormaps useful for neuroimaging.
"""
import numpy as _np

from matplotlib import cm as _cm
from matplotlib import colors as _colors

################################################################################
# Custom colormaps for two-tailed symmetric statistics
################################################################################

################################################################################
# Helper functions

def _rotate_cmap(cmap, swap_order=('green', 'red', 'blue')):
    """ Utility function to swap the colors of a colormap.
    """
    orig_cdict = cmap._segmentdata.copy()

    cdict = dict()
    cdict['green'] = [(p, c1, c2)
                        for (p, c1, c2) in orig_cdict[swap_order[0]]]
    cdict['blue'] = [(p, c1, c2)
                        for (p, c1, c2) in orig_cdict[swap_order[1]]]
    cdict['red'] = [(p, c1, c2)
                        for (p, c1, c2) in orig_cdict[swap_order[2]]]

    return cdict


def _pigtailed_cmap(cmap, swap_order=('green', 'red', 'blue')):
    """ Utility function to make a new colormap by concatenating a
        colormap with its reverse.
    """
    orig_cdict = cmap._segmentdata.copy()

    cdict = dict()
    cdict['green'] = [(0.5*(1-p), c1, c2)
                        for (p, c1, c2) in reversed(orig_cdict[swap_order[0]])]
    cdict['blue'] = [(0.5*(1-p), c1, c2)
                        for (p, c1, c2) in reversed(orig_cdict[swap_order[1]])]
    cdict['red'] = [(0.5*(1-p), c1, c2)
                        for (p, c1, c2) in reversed(orig_cdict[swap_order[2]])]

    for color in ('red', 'green', 'blue'):
        cdict[color].extend([(0.5*(1+p), c1, c2) 
                                    for (p, c1, c2) in orig_cdict[color]])

    return cdict


def _concat_cmap(cmap1, cmap2):
    """ Utility function to make a new colormap by concatenating two
        colormaps.
    """
    cdict = dict()

    cdict1 = cmap1._segmentdata.copy()
    cdict2 = cmap2._segmentdata.copy()
    if not hasattr(cdict1['red'], '__call__'):
        for c in ['red', 'green', 'blue']:
            cdict[c] = [(0.5*p, c1, c2) for (p, c1, c2) in cdict1[c]]
    else:
        for c in ['red', 'green', 'blue']:
            cdict[c] = []
        ps = _np.linspace(0, 1, 10)
        colors = cmap1(ps)
        for p, (r, g, b, a) in zip(ps, colors):
            cdict['red'].append((.5*p, r, r))
            cdict['green'].append((.5*p, g, g))
            cdict['blue'].append((.5*p, b, b))
    if not hasattr(cdict2['red'], '__call__'):
        for c in ['red', 'green', 'blue']:
            cdict[c].extend([(0.5*(1+p), c1, c2) for (p, c1, c2) in cdict2[c]])
    else:
        ps = _np.linspace(0, 1, 10)
        colors = cmap2(ps)
        for p, (r, g, b, a) in zip(ps, colors):
            cdict['red'].append((.5*(1+p), r, r))
            cdict['green'].append((.5*(1+p), g, g))
            cdict['blue'].append((.5*(1+p), b, b))

    return cdict


def alpha_cmap(color, name='', alpha_min=0.5, alpha_max=1.):
    """ Return a colormap with the given color, and alpha going from
        zero to 1.

        Parameters
        ----------
        color: (r, g, b), or a string
            A triplet of floats ranging from 0 to 1, or a matplotlib
            color string
    """
    red, green, blue = _colors.colorConverter.to_rgb(color)
    if name == '' and hasattr(color, 'startswith'):
        name = color
    cmapspec = [(red, green, blue, 1.),
                (red, green, blue, 1.),
               ]
    cmap = _colors.LinearSegmentedColormap.from_list(
                                '%s_transparent' % name, cmapspec, _cm.LUTSIZE)
    cmap._init()
    cmap._lut[:, -1] = _np.linspace(alpha_min, alpha_max, cmap._lut.shape[0])
    cmap._lut[-1, -1] = 0
    return cmap



################################################################################
# Our colormaps definition
_cmaps_data = dict(
    cold_hot     = _pigtailed_cmap(_cm.hot),
    cold_white_hot = _pigtailed_cmap(_cm.hot_r),
    brown_blue   = _pigtailed_cmap(_cm.bone),
    cyan_copper  = _pigtailed_cmap(_cm.copper),
    cyan_orange  = _pigtailed_cmap(_cm.YlOrBr_r),
    blue_red     = _pigtailed_cmap(_cm.Reds_r),
    brown_cyan   = _pigtailed_cmap(_cm.Blues_r),
    purple_green = _pigtailed_cmap(_cm.Greens_r,
                    swap_order=('red', 'blue', 'green')),
    purple_blue  = _pigtailed_cmap(_cm.Blues_r,
                    swap_order=('red', 'blue', 'green')),
    blue_orange  = _pigtailed_cmap(_cm.Oranges_r,
                    swap_order=('green', 'red', 'blue')),
    black_blue   = _rotate_cmap(_cm.hot),
    black_purple = _rotate_cmap(_cm.hot,
                                    swap_order=('blue', 'red', 'green')),
    black_pink   = _rotate_cmap(_cm.hot,
                            swap_order=('blue', 'green', 'red')),
    black_green  = _rotate_cmap(_cm.hot,
                            swap_order=('red', 'blue', 'green')),
    black_red    = _cm.hot._segmentdata.copy(),
)

if hasattr(_cm, 'ocean'):
    # MPL 0.99 doesn't have Ocean
    _cmaps_data['ocean_hot'] =  _concat_cmap(_cm.ocean, _cm.hot_r)
if hasattr(_cm, 'afmhot'): # or afmhot
    _cmaps_data['hot_white_bone'] = _concat_cmap(_cm.afmhot, _cm.bone_r)
    _cmaps_data['hot_black_bone'] = _concat_cmap(_cm.afmhot_r, _cm.bone)

# Copied from matplotlib 1.2.0 for matplotlib 0.99 compatibility.
_bwr_data = ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0))
_cmaps_data['bwr'] = _colors.LinearSegmentedColormap.from_list(
    'bwr', _bwr_data)._segmentdata.copy()

################################################################################
# Build colormaps and their reverse.
_cmap_d = dict()

for _cmapname in list(_cmaps_data.keys()):  # needed as dict changes within loop
    _cmapname_r = _cmapname + '_r'
    _cmapspec = _cmaps_data[_cmapname]
    _cmaps_data[_cmapname_r] = _cm.revcmap(_cmapspec)
    _cmap_d[_cmapname] = _colors.LinearSegmentedColormap(
                            _cmapname, _cmapspec, _cm.LUTSIZE)
    _cmap_d[_cmapname_r] = _colors.LinearSegmentedColormap(
                            _cmapname_r, _cmaps_data[_cmapname_r],
                            _cm.LUTSIZE)

################################################################################
# A few transparent colormaps
for color, name in (((1, 0, 0), 'red'),
                    ((0, 1, 0), 'green'),
                    ((0, 0, 1), 'blue'),
                    ):
    _cmap_d['%s_transparent' % name] = alpha_cmap(color, name=name)
    _cmap_d['%s_transparent_full_alpha_range' % name] = alpha_cmap(
        color, alpha_min=0,
        alpha_max=1, name=name)


# Save colormaps in the scope of the module
locals().update(_cmap_d)
# Register cmaps in matplotlib too
for k, v in _cmap_d.items():
    _cm.register_cmap(name=k, cmap=v)


################################################################################
# Utility to replace a colormap by another in an interval
################################################################################

def dim_cmap(cmap, factor=.3, to_white=True):
    """ Dim a colormap to white, or to black.
    """
    assert factor >= 0 and factor <=1, ValueError(
            'Dimming factor must be larger than 0 and smaller than 1, %s was passed.' 
                                                        % factor)
    if to_white:
        dimmer = lambda c: 1 - factor*(1-c)
    else:
        dimmer = lambda c: factor*c
    cdict = cmap._segmentdata.copy()
    for c_index, color in enumerate(('red', 'green', 'blue')):
        color_lst = list()
        for value, c1, c2 in cdict[color]:
            color_lst.append((value, dimmer(c1), dimmer(c2)))
        cdict[color] = color_lst

    return _colors.LinearSegmentedColormap(
                                '%s_dimmed' % cmap.name,
                                cdict,
                                _cm.LUTSIZE)


def replace_inside(outer_cmap, inner_cmap, vmin, vmax):
    """ Replace a colormap by another inside a pair of values.
    """
    assert vmin < vmax, ValueError('vmin must be smaller than vmax')
    assert vmin >= 0,    ValueError('vmin must be larger than 0, %s was passed.' 
                                        % vmin)
    assert vmax <= 1,    ValueError('vmax must be smaller than 1, %s was passed.' 
                                        % vmax)
    outer_cdict = outer_cmap._segmentdata.copy()
    inner_cdict = inner_cmap._segmentdata.copy()

    cdict = dict()
    for this_cdict, cmap in [(outer_cdict, outer_cmap),
                             (inner_cdict, inner_cmap)]:
        if hasattr(this_cdict['red'], '__call__'):
            ps = _np.linspace(0, 1, 25)
            colors = cmap(ps)
            this_cdict['red'] = list()
            this_cdict['green'] = list()
            this_cdict['blue'] = list()
            for p, (r, g, b, a) in zip(ps, colors):
                this_cdict['red'].append((p, r, r))
                this_cdict['green'].append((p, g, g))
                this_cdict['blue'].append((p, b, b))


    for c_index, color in enumerate(('red', 'green', 'blue')):
        color_lst = list()

        for value, c1, c2 in outer_cdict[color]:
            if value >= vmin:
                break
            color_lst.append((value, c1, c2))

        color_lst.append((vmin, outer_cmap(vmin)[c_index], 
                                inner_cmap(vmin)[c_index]))

        for value, c1, c2 in inner_cdict[color]:
            if value <= vmin:
                continue
            if value >= vmax:
                break
            color_lst.append((value, c1, c2))

        color_lst.append((vmax, inner_cmap(vmax)[c_index],
                                outer_cmap(vmax)[c_index]))

        for value, c1, c2 in outer_cdict[color]:
            if value <= vmax:
                continue
            color_lst.append((value, c1, c2))

        cdict[color] = color_lst

    return _colors.LinearSegmentedColormap(
                                '%s_inside_%s' % (inner_cmap.name, outer_cmap.name),
                                cdict,
                                _cm.LUTSIZE)


