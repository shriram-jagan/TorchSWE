# vim:fenc=utf-8
# vim:ft=pyrex

"""Linear reconstruction.
"""
from torchswe import nplike as _nplike

def reconstruct_cell_centers(qa, qb, qc, ua, ub, uc, b, drytol, tol):

    ua = qa - b

    ub = qb/ua
    uc = qc/ua

    if ua < tol:
        ua = 0.0
        ub = 0.0
        uc = 0.0
        qa = b
        qb = 0.0
        qc = 0.0
    if ua < drytol:
        ub = 0.0
        uc = 0.0
        qb = 0.0
        qc = 0.0

    return qa, qb, qc, ua, ub, uc


def reconstruct_face_velocity(ub, uc, ua, drytol, qb, qc):

    ub = qb/ua 
    uc = qc/ua
    if ua <= drytol:
        ub = 0.0
        uc = 0.0

    return ub, uc


def reconstruct_face_conservatives(qa, qb, qc, b, ua, ub, uc):
    """For internal use."""

    qa = ua + b
    qb = ua * ub
    qc = ua * uc

    return qa, qb, qc


vfunc_reconstruct_face_velocity       = _nplike.vectorize(reconstruct_face_velocity, cache=True)
vfunc_reconstruct_face_conservatives  = _nplike.vectorize(reconstruct_face_conservatives, cache=True)
vfunc_reconstruct_cell_centers        = _nplike.vectorize(reconstruct_cell_centers, cache=True)


# TODO: This kernel will be vectorized once we have support for multiple return statements
def _minmod_slope_kernel(s1, s2, s3, theta):
    """For internal use."""
    denominator = s3 - s2;

    with _nplike.errstate(divide="ignore", invalid="ignore"):
        slp = (s2 - s1) / denominator;

    slp[s3 == s2] = 0.0

    slp = _nplike.maximum(
        _nplike.minimum(
            _nplike.minimum(
                slp * theta,
                (slp + 1.0) / 2.0
            ),
            theta
        ),
        0.
    )

    slp *= denominator;
    slp /= 2.0;

    return slp


def _fix_face_depth_internal(hl, hc, hr, tol, nhl, nhr):
    """For internal use."""

    ids = hc < tol
    nhl[ids] = 0.0;
    nhr[ids] = 0.0;

    ids = hl < tol
    nhl[ids] = 0.0;
    _nplike.putmask(nhr, ids, hc*2.0) 

    ids = hr < tol
    _nplike.putmask(nhl, ids, hc*2.0)
    nhr[ids] = 0.0;


def _fix_face_depth_edge(h, hc, tol, nh):
    """For internal use."""

    nh[_nplike.logical_or(hc < tol, h < tol)] = 0.0
    _nplike.putmask(nh, h > hc*2.0, hc*2.0)


def reconstruct(states, runtime, config):
    """Reconstructs quantities at cell interfaces and centers.

    The following quantities in `states` are updated in this function:
        1. non-conservative quantities defined at cell centers (states.U)
        2. discontinuous non-conservative quantities defined at cell interfaces
        3. discontinuous conservative quantities defined at cell interfaces

    Arguments
    ---------
    states : torchswe.utils.data.States
    runtime : torchswe.utils.misc.DummyDict
    config : torchswe.utils.config.Config

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Returning it just for coding style. The values are actually
        updated in-place.
    """

    # aliases to save object look-up time in Python's underlying dictionary
    Q = states.q
    U = states.p
    slpx = states.slpx
    slpy = states.slpy
    xmQ = states.face.x.minus.q
    xmU = states.face.x.minus.p
    xpQ = states.face.x.plus.q
    xpU = states.face.x.plus.p
    ymQ = states.face.y.minus.q
    ymU = states.face.y.minus.p
    ypQ = states.face.y.plus.q
    ypU = states.face.y.plus.p
    xfcenters = runtime.topo.xf
    yfcenters = runtime.topo.yf

    ny = states.domain.y.n
    nx = states.domain.x.n
    ngh = states.domain.nhalo
    xbg = ngh
    xed = nx + ngh
    ybg = ngh
    yed = ny + ngh

    theta = config.params.theta
    drytol = config.params.drytol
    tol = runtime.tol

    # slopes for w, hu, and hv in x and y
    slpx = _minmod_slope_kernel(Q[:, ybg:yed, xbg-2:xed], Q[:, ybg:yed, xbg-1:xed+1], Q[:, ybg:yed, xbg:xed+2], theta)
    slpy = _minmod_slope_kernel(Q[:, ybg-2:yed, xbg:xed], Q[:, ybg-1:yed+1, xbg:xed], Q[:, ybg:yed+2, xbg:xed], theta)

    # extrapolate discontinuous w, hu, and hv
    _nplike.add(Q[:, ybg:yed, xbg-1:xed], slpx[:, :, :nx+1], out=xmQ)
    _nplike.subtract(Q[:, ybg:yed, xbg:xed+1], slpx[:, :, 1:], out=xpQ)
    _nplike.add(Q[:, ybg-1:yed, xbg:xed], slpy[:, :ny+1, :], out=ymQ)
    _nplike.subtract(Q[:, ybg:yed+1, xbg:xed], slpy[:, 1:, :], out=ypQ)

    # calculate depth at cell faces
    _nplike.subtract(xmQ[0], xfcenters, out=xmU[0])
    _nplike.subtract(xpQ[0], xfcenters, out=xpU[0])
    _nplike.subtract(ymQ[0], yfcenters, out=ymU[0])
    _nplike.subtract(ypQ[0], yfcenters, out=ypU[0])

    # the fixes for negative depths in x- and y- directions are commented out now
    # since the depths are positive and there is very little work done in the kernels
    # when the depths are positive (if condition not satisfied). This results
    # in sporadic usages in perf runs. 
    # This will be uncommented once the runtime can go farther ahead and schedule few more tasks

    # fix negative depths in x direction
    if 0:
        _fix_face_depth_internal(xpU[0, :, :nx].copy(), U[0, ybg:yed, xbg:xed], xmU[0, :, 1:].copy(), tol, xpU[0, :, :nx], xmU[0, :, 1:])
        _fix_face_depth_edge(xmU[0, :, 0].copy(), U[0, ybg:yed, xbg-1], tol, xmU[0, :, 0])
        _fix_face_depth_edge(xpU[0, :, nx].copy(), U[0, ybg:yed, xed], tol, xpU[0, :, nx])

    # fix negative depths in y direction
    if 0:
        _fix_face_depth_internal(ypU[0, :ny, :].copy(), U[0, ybg:yed, xbg:xed], ymU[0, 1:, :].copy(), tol, ypU[0, :ny, :], ymU[0, 1:, :])
        _fix_face_depth_edge(ymU[0, 0, :].copy(), U[0, ybg-1, xbg:xed], tol, ymU[0, 0, :])
        _fix_face_depth_edge(ypU[0, ny, :].copy(), U[0, yed, xbg:xed], tol, ypU[0, ny, :])

    # reconstruct velocity at cell faces in x and y directions
    xpU[1], xpU[2] = vfunc_reconstruct_face_velocity(xpU[1], xpU[2], xpU[0], drytol, xpQ[1], xpQ[2])
    xmU[1], xmU[2] = vfunc_reconstruct_face_velocity(xmU[1], xmU[2], xmU[0], drytol, xmQ[1], xmQ[2])
    ypU[1], ypU[2] = vfunc_reconstruct_face_velocity(ypU[1], ypU[2], ypU[0], drytol, ypQ[1], ypQ[2])
    ymU[1], ymU[2] = vfunc_reconstruct_face_velocity(ymU[1], ymU[2], ymU[0], drytol, ymQ[1], ymQ[2])

    # reconstruct conservative quantities at cell faces
    xmQ[0], xmQ[1], xmQ[2] = vfunc_reconstruct_face_conservatives(xmQ[0], xmQ[1], xmQ[2], xfcenters, xmU[0], xmU[1], xmU[2])
    xpQ[0], xpQ[1], xpQ[2] = vfunc_reconstruct_face_conservatives(xpQ[0], xpQ[1], xpQ[2], xfcenters, xpU[0], xpU[1], xpU[2])
    ymQ[0], ymQ[1], ymQ[2] = vfunc_reconstruct_face_conservatives(ymQ[0], ymQ[1], ymQ[2], yfcenters, ymU[0], ymU[1], ymU[2])
    ypQ[0], ypQ[1], ypQ[2] = vfunc_reconstruct_face_conservatives(ypQ[0], ypQ[1], ypQ[2], yfcenters, ypU[0], ypU[1], ypU[2])

    return states


def reconstruct_cell_centers(states, runtime, config):
    """Calculate cell-centered non-conservatives.

    `states.U` will be updated in this function, and `states.Q` may be changed, too.

    Arguments
    ---------
    states : torchswe.utils.data.States
    runtime : torchswe.utils.misc.DummyDict
    config : torchswe.utils.config.Config

    Returns
    -------
    states : torchswe.utils.data.States
    """

    tol = runtime.tol
    drytol = config.params.drytol
    c = runtime.topo.c

    states.q[0], states.q[1], states.q[2], states.p[0], states.p[1], states.p[2] = vfunc_reconstruct_cell_centers(
            states.q[0], 
            states.q[1], 
            states.q[2], 
            states.p[0], 
            states.p[1], 
            states.p[2], 
            c, 
            drytol, 
            tol,
        )

    return states
