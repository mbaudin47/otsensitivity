import pytest
import numpy as np
from otsensitivity import csiszar


def test_csiszar(ishigami):
    model, sample, data = ishigami

    s = csiszar(sample, data)
    assert s == pytest.approx(np.array([0.253, 0.260, 0.146]), abs=0.1)
