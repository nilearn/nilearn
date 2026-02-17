"""Field class defining common header fields in tractogram files"""


class Field:
    """Header fields common to multiple streamline file formats.

    In IPython, use `nibabel.streamlines.Field??` to list them.
    """

    NB_STREAMLINES = 'nb_streamlines'
    STEP_SIZE = 'step_size'
    METHOD = 'method'
    NB_SCALARS_PER_POINT = 'nb_scalars_per_point'
    NB_PROPERTIES_PER_STREAMLINE = 'nb_properties_per_streamline'
    NB_POINTS = 'nb_points'
    VOXEL_SIZES = 'voxel_sizes'
    DIMENSIONS = 'dimensions'
    MAGIC_NUMBER = 'magic_number'
    ORIGIN = 'origin'
    VOXEL_TO_RASMM = 'voxel_to_rasmm'
    VOXEL_ORDER = 'voxel_order'
    ENDIANNESS = 'endianness'
