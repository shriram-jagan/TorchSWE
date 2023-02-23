# vim:fenc=utf-8
# vim:ft=pyrex
from torchswe import nplike as _nplike


def get_discontinuous_flux(states):
    """Calculting the discontinuous fluxes on the both sides at cell faces.

    Arguments
    ---------
    states : torchswe.utils.data.States
    gravity : float

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.
    """

    x = states.face.x
    xm = x.minus
    xp = x.plus

    y = states.face.y
    ym = y.minus
    yp = y.plus

    gravity2_x = states.gravity2_x
    gravity2_y = states.gravity2_y

    # face normal to x-direction: [hu, hu^2 + g(h^2)/2, huv]
    xm.f[0] = xm.q[1]
    xm.f[1] = xm.q[1] * xm.p[1] + gravity2_x * xm.p[0] * xm.p[0]
    xm.f[2] = xm.q[1] * xm.p[2]
    xp.f[0] = xp.q[1]
    xp.f[1] = xp.q[1] * xp.p[1] + gravity2_x * xp.p[0] * xp.p[0]
    xp.f[2] = xp.q[1] * xp.p[2]

    # face normal to y-direction: [hv, huv, hv^2+g(h^2)/2]
    ym.f[0] = ym.q[2]
    ym.f[1] = ym.q[2] * ym.p[1]
    ym.f[2] = ym.q[2] * ym.p[2] + gravity2_y * ym.p[0] * ym.p[0]
    yp.f[0] = yp.q[2]
    yp.f[1] = yp.q[2] * yp.p[1]
    yp.f[2] = yp.q[2] * yp.p[2] + gravity2_y * yp.p[0] * yp.p[0]

    return states


def central_scheme_kernel(ma, pa, mf, pf, mq, pq, flux):
    """For internal use"""

    flux = (pa * mf - ma * pf + pa * ma * (pq - mq)) / (pa - ma);

    cond = (pa == ma)  # should we deal with small rounding err here???
    flux[:, cond] = 0.0


def central_scheme(states):
    """A central scheme to calculate numerical flux at interfaces.

    Arguments
    ---------
    states : torchswe.utils.data.States

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Updated in-place. Returning it just for coding style.
    """

    x = states.face.x
    xm = x.minus
    xp = x.plus

    y = states.face.y
    ym = y.minus
    yp = y.plus

    central_scheme_kernel(xm.a, xp.a, xm.f, xp.f, xm.q, xp.q, x.cf)
    central_scheme_kernel(ym.a, yp.a, ym.f, yp.f, ym.q, yp.q, y.cf)

    return states


def get_local_speed_kernel(hp, hm, up, um, g, ap, am, zeros):
    """For internal use to mimic CuPy and Cython kernels."""
    ghp = _nplike.sqrt(g * hp);
    ghm = _nplike.sqrt(g * hm);
    ap = _nplike.maximum(_nplike.maximum(up+ghp, um+ghm), zeros);
    am = _nplike.minimum(_nplike.minimum(up-ghp, um-ghm), zeros);


def get_local_speed(states):
    """Calculate local speeds on the two sides of cell faces.

    Arguments
    ---------
    states : torchswe.utils.data.States
    gravity : float
        Gravity in m / s^2.

    Returns
    -------
    states : torchswe.utils.data.States
        The same object as the input. Changed inplace. Returning it just for coding style.
    """

    # faces normal to x- and y-directions

    get_local_speed_kernel(
            states.face.x.plus.p[0],
            states.face.x.minus.p[0],
            states.face.x.plus.p[1], 
            states.face.x.minus.p[1], 
            states.gravity_x,
            states.face.x.plus.a, 
            states.face.x.minus.a,
            _nplike.zeros(states.x_plus_p0_shape),
            )

    get_local_speed_kernel(
            states.face.y.plus.p[0], 
            states.face.y.minus.p[0], 
            states.face.y.plus.p[2], 
            states.face.y.minus.p[2], 
            states.gravity_y,
            states.face.y.plus.a, 
            states.face.y.minus.a,
            _nplike.zeros(states.y_plus_p0_shape),
            )

    return states
