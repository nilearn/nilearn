"""
Basic numerics and plotting with Python
========================================

"""

# import numpy: the module providing numerical arrays
import numpy as np
t = np.linspace(1, 10, 2000)

# import pylab: the module for scientific plotting
import pylab as pl
pl.plot(t, np.cos(t))
