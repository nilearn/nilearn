"""
Example of second level design matrix
=====================================

Requires matplotlib

Author : Martin Perez-Guevara: 2016
"""

print(__doc__)

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nistats.design_matrix import (create_second_level_design,
                                   plot_design_matrix)
import pandas as pd

# Experimental paradigm has two conditions and 20 subjects
column_names = ['map_name', 'model_id', 'map_path']
n_subjects = 20
maps_name = ['language'] * n_subjects + ['consonants'] * n_subjects
subject_list = ['sub-%02d' % i for i in range(1, n_subjects + 1)]
maps_model = subject_list * 2
maps_path = [''] * n_subjects * 2
maps_table = pd.DataFrame({'map_name': maps_name,
                           'model_id': maps_model,
                           'effects_map_path': maps_path})
# Specify extra information about the subjects to create additional regressors
extra_info_subjects = pd.DataFrame({'model_id': subject_list,
                                    'age': [i for i in range(15, 35)],
                                    'sex': [0, 1] * 10})

# Create design matrix
design_matrix = create_second_level_design(maps_table, extra_info_subjects)

# plot the results
ax = plot_design_matrix(design_matrix)
ax.set_title('Second level design matrix', fontsize=12)
ax.set_ylabel('maps')
plt.tight_layout()
plt.show()
