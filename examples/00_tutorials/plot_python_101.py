"""
Basic numerics and plotting with Python
========================================

A simple example of basic Python numerics and how to
plot it.
"""

# import numpy: the module providing numerical arrays
import numpy as np

t = np.linspace(1, 10, 2000)

# import matplotlib.pyplot: the module for scientific plotting
import matplotlib.pyplot as plt

plt.plot(t, np.cos(t))
