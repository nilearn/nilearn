"""
Helper functions to download NeuroImaging datasets
"""

from .struct import (fetch_icbm152_2009, load_mni152_template, fetch_oasis_vbm)
from .func import (fetch_haxby_simple, fetch_haxby, fetch_nyu_rest,
                   fetch_adhd, fetch_miyawaki2008,
                   fetch_localizer_contrasts, fetch_abide_pcp,
                   fetch_localizer_calculation_task, fetch_mixed_gambles,
                   fetch_megatrawls_netmats, fetch_cobre, fetch_neurovault)
from .atlas import (fetch_atlas_craddock_2012, fetch_atlas_destrieux_2009,
                    fetch_atlas_harvard_oxford, fetch_atlas_msdl,
                    fetch_atlas_power_2011, fetch_coords_power_2011,
                    fetch_atlas_smith_2009,
                    fetch_atlas_yeo_2011, fetch_atlas_aal,
                    fetch_atlas_basc_multiscale_2015,
                    fetch_coords_dosenbach_2010)

__all__ = ['fetch_icbm152_2009', 'load_mni152_template', 'fetch_oasis_vbm',
           'fetch_haxby_simple', 'fetch_haxby', 'fetch_nyu_rest',
           'fetch_adhd', 'fetch_miyawaki2008', 'fetch_localizer_contrasts',
           'fetch_abide_pcp', 'fetch_localizer_calculation_task',
           'fetch_atlas_craddock_2012', 'fetch_atlas_destrieux_2009',
           'fetch_atlas_harvard_oxford', 'fetch_atlas_msdl',
           'fetch_atlas_power_2011', 'fetch_coords_power_2011',
           'fetch_atlas_smith_2009',
           'fetch_atlas_yeo_2011', 'fetch_mixed_gambles', 'fetch_atlas_aal',
           'fetch_megatrawls_netmats', 'fetch_cobre', 'fetch_neurovault',
           'fetch_atlas_basc_multiscale_2015', 'fetch_coords_dosenbach_2010']
