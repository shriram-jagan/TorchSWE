from torchswe import nplike as np
import numpy as num
import pytest

from torchswe.kernels.cunumeric_reconstruction import (
    _minmod_slope_kernel,
    _fix_face_depth_internal,
    _fix_face_depth_edge,
    _recnstrt_face_velocity,
    _recnstrt_face_conservatives,
)


@pytest.fixture
def get_stencil_params():
    nx = 5
    ny = 8
    ngh = 2
    xbg = ngh
    xed = nx + ngh
    ybg = ngh
    yed = ny + ngh

    return nx, ny, ngh, xbg, xed, ybg, yed


# this could be a class with all of these variables
@pytest.fixture
def get_U_either_side_incl_x(get_stencil_params):
    """ Computes the following quantities for a given resolution:
    xpU = states.face.x.plus.p
    xmU = states.face.x.minus.p
    U   = states.p
    """
    nx, ny, ngh, xbg, xed, ybg, yed = get_stencil_params

    shape_xmU = (3, ny, nx + 1)
    shape_U = (3, ny + 2 * ngh, nx + 2 * ngh)
    shape_xpU = (3, ny, nx + 1)
    ntotal_xmU = int(np.prod(shape_xmU))
    ntotal_U = int(np.prod(shape_U))
    ntotal_xpU = int(np.prod(shape_xpU))

    xmU = np.arange(ntotal_xmU).reshape(shape_xmU).astype(float)
    U = np.arange(ntotal_U).reshape(shape_U).astype(float)
    xpU = np.arange(ntotal_xpU).reshape(shape_xpU).astype(float)

    return xmU, U, xpU


@pytest.fixture
def get_UQ_plus_x(get_stencil_params):
    """ Computes the following quantities for a given resolution:

    xpU = states.face.x.plus.p
    xpQ = states.face.x.plus.q
    """
    nx, ny, ngh, xbg, xed, ybg, yed = get_stencil_params

    shape_xpU = (3, ny, nx + 1)
    shape_xpQ = (3, ny, nx + 1)
    ntotal_xpU = int(np.prod(shape_xpU))
    ntotal_xmU = int(np.prod(shape_xmU))

    xpU = np.arange(ntotal_xpU).reshape(shape_xpU).astype(float)
    xpQ = np.arange(ntotal_xmU).reshape(shape_xpQ).astype(float)

    return xpU, xpQ


@pytest.fixture
def get_UQ_plus_y(get_stencil_params):
    """ Computes the following quantities for a given resolution:

    ypU = states.face.y.plus.p
    ypQ = states.face.y.plus.q
    """
    nx, ny, ngh, xbg, xed, ybg, yed = get_stencil_params

    shape_ypU = (3, ny + 1, nx)
    shape_ypQ = (3, ny + 1, nx)
    ntotal_ypU = int(np.prod(shape_ypU))
    ntotal_ymU = int(np.prod(shape_ymU))

    ypU = np.arange(ntotal_ypU).reshape(shape_ypU).astype(float)
    ypQ = np.arange(ntotal_ymU).reshape(shape_ypQ).astype(float)

    return xpU, xpQ


@pytest.fixture
def get_U_either_side_incl_in_y(get_stencil_params):
    """ Computes the following quantities for a given resolution:

    ypU = states.face.y.plus.p
    ymU = states.face.y.minus.p
    U   = states.p
    """
    nx, ny, ngh, xbg, xed, ybg, yed = get_stencil_params

    shape_ypU = (3, ny + 1, nx)
    shape_U = (3, ny + 2 * ngh, nx + 2 * ngh)
    shape_ymU = (3, ny + 1, nx)
    ntotal_ypU = int(np.prod(shape_ypU))
    ntotal_U = int(np.prod(shape_U))
    ntotal_ymU = int(np.prod(shape_ymU))

    ypU = np.arange(ntotal_ypU).reshape(shape_ypU).astype(float)
    U = np.arange(ntotal_U).reshape(shape_U).astype(float)
    ymU = np.arange(ntotal_ymU).reshape(shape_ymU).astype(float)

    return ypU, U, ymU


class TestPositiveCases:

    # slopes for w, hu, and hv in x and y
    def test_minmod_slope_kernel(self,):

        (nx, ny) = (4, 5)
        ngh = 2

        xbg = ngh
        xed = nx + ngh
        ybg = ngh
        yed = ny + ngh

        theta = 1.0

        shape = (3, ny + 2 * ngh, nx + 2 * ngh)
        ntotal = int(np.prod(shape))

        Q = np.arange(ntotal).reshape(shape)

        slpx = _minmod_slope_kernel(
            s1=Q[:, ybg:yed, xbg - 2 : xed],
            s2=Q[:, ybg:yed, xbg - 1 : xed + 1],
            s3=Q[:, ybg:yed, xbg : xed + 2],
            theta=theta,
        )

        slpy = _minmod_slope_kernel(
            s1=Q[:, ybg - 2 : yed, xbg:xed],
            s2=Q[:, ybg - 1 : yed + 1, xbg:xed],
            s3=Q[:, ybg : yed + 2, xbg:xed],
            theta=theta,
        )

        assert np.all(slpx == 0.5)
        assert np.all(slpy == 4.0)

    def test_fix_face_depth_internal(self, get_U_either_side_incl_x):
        (nx, ny) = (5, 8)
        ngh = 2

        xbg = ngh
        xed = nx + ngh
        ybg = ngh
        yed = ny + ngh

        tol = 2.0
        xmU, U, xpU = get_U_either_side_incl_x

        # find the indices that will get updated
        indices = np.logical_or(
            xpU[0, :, :nx] < tol, U[0, ybg:yed, xbg:xed] < tol, xmU[0, :, 1:] < tol
        )

        old_xmU = xmU[0, :, 1:][indices]
        old_xpU = xpU[0, :, :nx][indices]
        _fix_face_depth_internal(
            xpU[0, :, :nx].copy(),
            U[0, ybg:yed, xbg:xed],
            xmU[0, :, 1:].copy(),
            tol,
            xpU[0, :, :nx],
            xmU[0, :, 1:],
        )
        new_xmU = xmU[0, :, 1:][indices]
        new_xpU = xpU[0, :, :nx][indices]

        # right now we only check if the element has been updated or not
        # eventually, we need to check the values as well
        assert not np.all(old_xmU == new_xmU)
        assert not np.all(old_xpU == new_xpU)

        # do for ymU and ypU

    def test_fix_face_depth_edge(self, get_U_either_side_incl_x):
        (nx, ny) = (5, 8)
        ngh = 2

        xbg = ngh
        xed = nx + ngh
        ybg = ngh
        yed = ny + ngh

        tol = 20
        xmU, U, xpU = get_U_either_side_incl_x

        indices = np.logical_or(xmU[0, :, 0] < tol, U[0, ybg:yed, xbg - 1] < tol)
        old_xmU = xmU[0, :, 0][indices]
        _fix_face_depth_edge(xmU[0, :, 0], U[0, ybg:yed, xbg - 1], tol, xmU[0, :, 0])
        new_xmU = xmU[0, :, 0][indices]

        assert not np.all(old_xmU == new_xmU)

        indices = np.logical_or(xpU[0, :, nx] < tol, U[0, ybg:yed, xbg - 1] < tol)
        old_xpU = xpU[0, :, nx][indices]
        _fix_face_depth_edge(
            xpU[0, :, nx].copy(), U[0, ybg:yed, xed], tol, xpU[0, :, nx]
        )
        new_xpU = xpU[0, :, nx][indices]

        # right now we only check if the element has been updated or not
        # eventually, we need to check the values as well
        assert not np.all(old_xpU == new_xpU)

        # do for ymU and ypU

    def test_recnstrt_face_velocity(self,):
        # _recnstrt_face_velocity(xpQ, drytol, xpU)
        # assert np.all(xpU[1:3, xpQ <= drytol] == 0.0)
        assert True

    def test_recnstrt_face_conservatives(self,):
        assert True
