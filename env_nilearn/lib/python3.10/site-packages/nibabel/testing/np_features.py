"""Look for changes in numpy behavior over versions"""

from functools import cache

import numpy as np


@cache
def memmap_after_ufunc() -> bool:
    """Return True if ufuncs on memmap arrays always return memmap arrays

    This should be True for numpy < 1.12, False otherwise.
    """
    with open(__file__, 'rb') as fobj:
        mm_arr = np.memmap(fobj, mode='r', shape=(10,), dtype=np.uint8)
        return isinstance(mm_arr + 1, np.memmap)
