"""
Analysing fMRI data using GLMs.
"""

from nilearn.stats import (contrasts,
                           design_matrix,
                           experimental_paradigm,
                           first_level_model,
                           hemodynamic_models,
                           model,
                           regression,
                           second_level_model,
                           thresholding,
                           utils,
                           )

__all__ = ['contrasts', 'design_matrix', 'experimental_paradigm',
           'first_level_model', 'hemodynamic_models', 'model',
           'regression', 'second_level_model', 'thresholding', 'utils',
           ]

# from nilearn.stats.contrasts import (
#     compute_contrast,
#     compute_fixed_effects,
# )
# from nilearn.stats.design_matrix import (
#     make_first_level_design_matrix,
#     make_second_level_design_matrix,
# )
#
# __all__ = ['compute_contrast',
#            'compute_fixed_effects',
#            'make_first_level_design_matrix',
#            'make_second_level_design_matrix',
#            ]
