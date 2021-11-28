# vim:fenc=utf-8
# vim:ft=pyrex
cimport _checker
cimport cython
import cupy


ctypedef fused OutflowBC:
    OutflowWH
    OutflowOther


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class OutflowCommon:

    # depth
    cdef object hbci  # depth at the inner side of the boundary cell faces
    cdef object hbco  # depth at the outer side of the boundary cell faces
    cdef object hother  # depth at the inner side of the another face of the 1st internal cell


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class OutflowWH(OutflowCommon):

    # conservatives
    cdef object wc0  # w at the cell centers of the 1st internal cell layer
    cdef object wbci  # w at the inner side of the boundary cell faces
    cdef object wbco  # w at the outer side of the boundary cell faces
    cdef object wother  # w at the inner side of the another face of the 1st internal cell

    # depth
    cdef object hc0  # depth at the cell centers of the 1st internal cell layer

    # topography elevation
    cdef object bbc  # topo elevations at the boundary cell faces
    cdef object bother  # topo elevation at cell faces between the 1st and 2nd internal cell layers

    # tolerance
    cdef double tol  # depths under this tolerance are considered dry cells

    def __call__(self):
        _outflow_bc_w_h_kernel(
            self.wc0, self.hc0, self.bbc, self.bother, self.tol,
            self.wbci, self.wbco, self.wother, self.hbci, self.hbco, self.hother
        )


@cython.auto_pickle(False)  # meaningless to pickle a BC instance as everything is a memoryview
cdef class OutflowOther(OutflowCommon):

    # conservatives
    cdef object qbci  # q at the inner side of the boundary cell faces
    cdef object qbco  # q at the outer side of the boundary cell faces
    cdef object qother  # q at the inner side of the another face of the 1st internal cell

    # velocities
    cdef object uc0  # u or v at the cell centers of the 1st internal cell layer
    cdef object ubci  # u or v at the inner side of the boundary cell faces
    cdef object ubco  # u or v at the outer side of the boundary cell faces
    cdef object uother  # u or v at the inner side of the another face of the 1st internal cell

    # tolerance
    cdef double drytol  # depths under this values are considered wet but still cells

    def __call__(self):
        _outflow_bc_kernel(
            self.uc0, self.hbci, self.hbco, self.hother, self.drytol,
            self.qbci, self.qbco, self.qother, self.ubci, self.ubco, self.uother
        )


cdef _outflow_bc_w_h_kernel = cupy.ElementwiseKernel(
    "T wc0, T hc0, T bbc, T bother, T tol",
    "T wbci, T wbco, T wother, T hbci, T hbco, T hother",
    """
        if (hc0 < tol) {
            hbci = 0.0;
            hbco = 0.0;
            hother = 0.0;
            wbci = bbc;
            wbco = bbc;
            wother = bother;
            continue;
        }

        hbci = wc0 - bbc;
        hother = wc0 - bother;

        // fix negative depth
        if (hbci < tol) {
            hbci = 0.0;
            hother = hc0 * 2.0;
        } else if (hother < tol) {
            hbci = hc0 * 2.0;
            hother = 0.0;
        }

        hbco = hbci;

        // reconstruct to eliminate rounding error-edffect in further calculations
        wbci = hbci + bbc;
        wbco = hbco + bbc;
        wother = hother + bother;
    """,
    "_outflow_bc_w_h_kernel"
)


cdef _outflow_bc_kernel = cupy.ElementwiseKernel(
    "T uc0, T hbci, T hbco, T hother, T drytol",
    "T qbci, T qbco, T qother, T ubci, T ubco, T uother",
    """
        if (hbco < drytol) {
            ubco = 0.0;
            qbco = 0.0;
        } else {
            ubco = uc0;
            qbco = hbco * ubco;
        }

        if (hbci < drytol) {
            ubci = 0.0;
            qbci = 0.0;
        } else {
            ubci = uc0;
            qbci = hbci * ubci;
        }

        if (hother < drytol) {
            uother = 0.0;
            qother = 0.0;
        } else {
            uother = uc0;
            qother = hother * uother;
        }
    """,
    "_outflow_bc_kernel"
)


cdef void _outflow_bc_set_west(
    OutflowBC bc,
    object Q, object xmQ, object xpQ,
    object U, object xmU, object xpU,
    object Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp,
    const double tol, const double drytol
) except *:

    bc.hbci = xpU[0, :, 0]
    bc.hbco = xmU[0, :, 0]
    bc.hother = xmU[0, :, 1]

    if OutflowBC is OutflowWH:
        bc.wc0 = Q[0, ngh:Q.shape[1]-ngh, ngh]
        bc.wbci = xpQ[0, :, 0]
        bc.wbco = xmQ[0, :, 0]
        bc.wother = xmQ[0, :, 1]

        bc.hc0 = U[0, ngh:U.shape[1]-ngh, ngh]

        bc.bbc = Bx[:, 0]
        bc.bother = Bx[:, 1]

        bc.tol = tol
    else:
        bc.qbci = xpQ[comp, :, 0]
        bc.qbco = xmQ[comp, :, 0]
        bc.qother = xmQ[comp, :, 1]

        bc.uc0 = U[comp, ngh:Q.shape[1]-ngh, ngh]
        bc.ubci = xpU[comp, :, 0]
        bc.ubco = xmU[comp, :, 0]
        bc.uother = xmU[comp, :, 1]

        bc.drytol = drytol


cdef void _outflow_bc_set_east(
    OutflowBC bc,
    object Q, object xmQ, object xpQ,
    object U, object xmU, object xpU,
    object Bx,
    const Py_ssize_t ngh, const Py_ssize_t comp,
    const double tol, const double drytol
) except *:

    bc.hbci = xmU[0, :, xmU.shape[2]-1]
    bc.hbco = xpU[0, :, xpU.shape[2]-1]
    bc.hother = xpU[0, :, xpU.shape[2]-2]

    if OutflowBC is OutflowWH:
        bc.wc0 = Q[0, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
        bc.wbci = xmQ[0, :, xmQ.shape[2]-1]
        bc.wbco = xpQ[0, :, xpQ.shape[2]-1]
        bc.wother = xpQ[0, :, xpQ.shape[2]-2]

        bc.hc0 = U[0, ngh:U.shape[1]-ngh, U.shape[2]-ngh-1]

        bc.bbc = Bx[:, Bx.shape[1]-1]
        bc.bother = Bx[:, Bx.shape[1]-2]

        bc.tol = tol
    else:
        bc.qbci = xmQ[comp, :, xmQ.shape[2]-1]
        bc.qbco = xpQ[comp, :, xpQ.shape[2]-1]
        bc.qother = xpQ[comp, :, xpQ.shape[2]-2]

        bc.uc0 = U[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
        bc.ubci = xmU[comp, :, xmU.shape[2]-1]
        bc.ubco = xpU[comp, :, xpU.shape[2]-1]
        bc.uother = xpU[comp, :, xpU.shape[2]-2]

        bc.drytol = drytol


cdef void _outflow_bc_set_south(
    OutflowBC bc,
    object Q, object ymQ, object ypQ,
    object U, object ymU, object ypU,
    object By,
    const Py_ssize_t ngh, const Py_ssize_t comp,
    const double tol, const double drytol
) except *:

    bc.hbci = ypU[0, 0, :]
    bc.hbco = ymU[0, 0, :]
    bc.hother = ymU[0, 1, :]

    if OutflowBC is OutflowWH:
        bc.wc0 = Q[0, ngh, ngh:Q.shape[2]-ngh]
        bc.wbci = ypQ[0, 0, :]
        bc.wbco = ymQ[0, 0, :]
        bc.wother = ymQ[0, 1, :]

        bc.hc0 = U[0, ngh, ngh:U.shape[2]-ngh]

        bc.bbc = By[0, :]
        bc.bother = By[1, :]

        bc.tol = tol
    else:
        bc.qbci = ypQ[comp, 0, :]
        bc.qbco = ymQ[comp, 0, :]
        bc.qother = ymQ[comp, 1, :]

        bc.uc0 = U[comp, ngh, ngh:Q.shape[2]-ngh]
        bc.ubci = ypU[comp, 0, :]
        bc.ubco = ymU[comp, 0, :]
        bc.uother = ymU[comp, 1, :]

        bc.drytol = drytol


cdef void _outflow_bc_set_north(
    OutflowBC bc,
    object Q, object ymQ, object ypQ,
    object U, object ymU, object ypU,
    object By,
    const Py_ssize_t ngh, const Py_ssize_t comp,
    const double tol, const double drytol
) except *:

    bc.hbci = ymU[0, ymU.shape[1]-1, :]
    bc.hbco = ypU[0, ypU.shape[1]-1, :]
    bc.hother = ypU[0, ypU.shape[1]-2, :]

    if OutflowBC is OutflowWH:
        bc.wc0 = Q[0, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
        bc.wbci = ymQ[0, ymQ.shape[1]-1, :]
        bc.wbco = ypQ[0, ypQ.shape[1]-1, :]
        bc.wother = ypQ[0, ypQ.shape[1]-2, :]

        bc.hc0 = U[0, U.shape[1]-ngh-1, ngh:U.shape[2]-ngh]

        bc.bbc = By[By.shape[0]-1, :]
        bc.bother = By[By.shape[0]-2, :]

        bc.tol = tol
    else:
        bc.qbci = ymQ[comp, ymQ.shape[1]-1, :]
        bc.qbco = ypQ[comp, ypQ.shape[1]-1, :]
        bc.qother = ypQ[comp, ypQ.shape[1]-2, :]

        bc.uc0 = U[comp, Q.shape[1]-ngh-1, ngh:Q.shape[2]-ngh]
        bc.ubci = ymU[comp, ymU.shape[1]-1, :]
        bc.ubco = ypU[comp, ypU.shape[1]-1, :]
        bc.uother = ypU[comp, ypU.shape[1]-2, :]

        bc.drytol = drytol


cdef void _outflow_bc_factory(
    OutflowBC bc,
    object Q, object xmQ, object xpQ, object ymQ, object ypQ,
    object U, object xmU, object xpU, object ymU, object ypU,
    object Bx, object By,
    const Py_ssize_t ngh, const unsigned comp, const unsigned ornt,
    const double tol, const double drytol,
):

    # runtime check for the shapes
    _checker.shape_checker(Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By, ngh, comp, ornt)

    if ornt == 0:  # west
        _outflow_bc_set_west[OutflowBC](bc, Q, xmQ, xpQ, U, xmU, xpU, Bx, ngh, comp, tol, drytol)
    elif ornt == 1:  # east
        _outflow_bc_set_east[OutflowBC](bc, Q, xmQ, xpQ, U, xmU, xpU, Bx, ngh, comp, tol, drytol)
    elif ornt == 2:  # south
        _outflow_bc_set_south[OutflowBC](bc, Q, ymQ, ypQ, U, ymU, ypU, By, ngh, comp, tol, drytol)
    elif ornt == 3:  # north
        _outflow_bc_set_north[OutflowBC](bc, Q, ymQ, ypQ, U, ymU, ypU, By, ngh, comp, tol, drytol)
    else:
        raise ValueError(f"orientation id {ornt} not accepted.")


def outflow_bc_factory(ornt, comp, states, topo, tol, drytol, *args, **kwargs):
    """Factory to create a outflow (constant extrapolation) boundary condition callable object.
    """

    # aliases
    cdef object Q = states.Q
    cdef object xmQ = states.face.x.minus.Q
    cdef object xpQ = states.face.x.plus.Q
    cdef object ymQ = states.face.y.minus.Q
    cdef object ypQ = states.face.y.plus.Q

    cdef object U = states.U
    cdef object xmU = states.face.x.minus.U
    cdef object xpU = states.face.x.plus.U
    cdef object ymU = states.face.y.minus.U
    cdef object ypU = states.face.y.plus.U

    cdef object Bx = topo.xfcenters
    cdef object By = topo.yfcenters

    cdef Py_ssize_t ngh = states.domain.nhalo

    if isinstance(ornt, str):
        if ornt in ["w", "west"]:
            ornt = 0
        elif ornt in ["e", "east"]:
            ornt = 1
        elif ornt in ["s", "south"]:
            ornt = 2
        elif ornt in ["n", "north"]:
            ornt = 3

    if comp == 0:
        bc = OutflowWH()
        _outflow_bc_factory[OutflowWH](
            bc, Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
            ngh, comp, ornt, tol, drytol
        )
    elif comp == 1 or comp == 2:
        bc = OutflowOther()
        _outflow_bc_factory[OutflowOther](
            bc, Q, xmQ, xpQ, ymQ, ypQ, U, xmU, xpU, ymU, ypU, Bx, By,
            ngh, comp, ornt, tol, drytol
        )
    else:
        raise ValueError(f"Unrecognized component: {comp}")

    return bc
