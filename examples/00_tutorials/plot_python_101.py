"""
Basic numerics and plotting with Python
=======================================
"""

from nilearn._utils.helpers import check_matplotlib

check_matplotlib()

# import numpy: the module providing numerical arrays
# %%
# A simple example of basic Python numerics and how to plot it.
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(1, 10, 2000)

plt.plot(t, np.cos(t))
