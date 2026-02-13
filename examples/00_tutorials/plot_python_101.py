"""
Basic numerics and plotting with Python
=======================================
"""

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise RuntimeError("This script needs the matplotlib library") from e

# %%
# A simple example of basic Python numerics and how to plot it.

# import numpy: the module providing numerical arrays
import numpy as np

t = np.linspace(1, 10, 2000)

plt.plot(t, np.cos(t))

plt.show()
