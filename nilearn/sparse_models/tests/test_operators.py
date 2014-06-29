from ..operators import prox_l1

import itertools
import numpy as np


def test_prox_l1():
    """Tests non-expansivity of proximal operator"""
    x = np.arange(10)[np.newaxis, :]
    tau = .3
    s = prox_l1(x.copy(), tau)
    p = x - s  # projection + shrinkage = id

    # We should have ||s(a) - s(b)||^2 <= ||a - b||^2 - ||p(a) - p(b)||^2
    # for all a and b
    for (a, b), (pa, pb), (sa, sb) in zip(*[itertools.product(z[0], z[0])
                                            for z in [x, p, s]]):
        assert (sa - sb) ** 2 <= (a - b) ** 2 - (pa - pb) ** 2
