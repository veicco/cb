import numpy as np
import pandas as pd
import os
from numpy.testing import assert_array_equal
from cb.base import ContextualBanditBase


np.random.seed(123)


def test_get_actions_one_hot():
    actions_one_hot = ContextualBanditBase()._get_actions_one_hot(
        [10, 11, 12, 13, 14, 15], 12
    )
    assert_array_equal(actions_one_hot, np.array([0, 0, 1, 0, 0, 0]))
