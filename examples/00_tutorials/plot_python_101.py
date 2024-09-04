"""
Basic numerics and plotting with Python
=======================================
"""

from nilearn._utils.helpers import is_matplotlib_installed

if not is_matplotlib_installed():
    raise RuntimeError("This script needs the matplotlib library")

import matplotlib.pyplot as plt

# import numpy: the module providing numerical arrays
import numpy as np

# %%
# A simple example of basic Python numerics and how to plot it.


t = np.linspace(1, 10, 2000)

plt.plot(t, np.cos(t))
