from torchswe import nplike as np
import numpy as num
import pytest

from torchswe.kernels.cunumeric_flux import (
        get_local_speed_kernel,
        get_discontinuous_flux,
        central_scheme_kernel,
)

# needs to instantiate a states class;

def test_dummy():
    assert True
