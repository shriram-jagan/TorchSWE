# vim:fenc=utf-8
# vim:ft=pyrex
from torchswe import nplike as _nplike

def get_dis_flux_kernel1(xmf0, xmf1, xmf2, grav2,  xmq1, xmp0, xmp1, xmp2):

    # face normal to x-direction: [hu, hu^2 + g(h^2)/2, huv]
    xmf0 = xmq1
    xmf1 = xmq1 * xmp1 + grav2 * xmp0 * xmp0
    xmf2 = xmq1 * xmp2

def get_dis_flux_kernel2(xpf0, xpf1, xpf2, grav2, xpq1, xpp0, xpp1, xpp2):
    xpf0 = xpq1
    xpf1 = xpq1 * xpp1 + grav2 * xpp0 * xpp0
    xpf2 = xpq1 * xpp2


def get_dis_flux_kernel3(ypf0,ypf1,ypf2,grav2, ypq2, ypp0, ypp1, ypp2) :
    ypf0 = ypq2
    ypf1 = ypq2 * ypp1
    ypf2 = ypq2 * ypp2 + grav2 * ypp0 * ypp0


def get_dis_flux_kernel4(ymf0, ymf1, ymf2, grav2, ymq2, ymp0, ymp1, ymp2):
    ymf0 = ymq2
    ymf1 = ymq2 * ymp1
    ymf2 = ymq2 * ymp2 + grav2 * ymp0 * ymp0

vecfunc1 = _nplike.vectorize(get_dis_flux_kernel1, otypes=(float,float,float), cache=True)
vecfunc2 = _nplike.vectorize(get_dis_flux_kernel2, otypes=(float,float,float),cache=True)
vecfunc3 = _nplike.vectorize(get_dis_flux_kernel3,otypes=(float,float,float), cache=True)
vecfunc4 = _nplike.vectorize(get_dis_flux_kernel4, otypes=(float,float,float),cache=True)


def get_discontinuous_flux(states, gravity):
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
    a = _nplike.empty(xm.f[0].shape)
    a.fill(gravity/2.0)

    vecfunc1(xm.f[0], xm.f[1], xm.f[2],a,
        xm.q[1],xm.p[0], xm.p[1], xm.p[2])
    vecfunc2(xp.f[0], xp.f[1], xp.f[2],a,
        xp.q[1], xp.p[0], xp.p[1],xp.p[2])

    a2 = _nplike.empty(yp.f[0].shape)
    a2.fill(gravity/2.0)
    vecfunc3(yp.f[0], yp.f[1], yp.f[2],a2, yp.q[2],
        yp.p[0], yp.p[1], yp.p[2])
    vecfunc4( ym.f[0], ym.f[1], ym.f[2], a2,ym.q[2],
        ym.p[0], ym.p[1], ym.p[2])

    return states

    #grav2 = gravity/2.0

    ## face normal to x-direction: [hu, hu^2 + g(h^2)/2, huv]
    #xm.f[0] = xm.q[1]
    #xm.f[1] = xm.q[1] * xm.p[1] + grav2 * xm.p[0] * xm.p[0]
    #xm.f[2] = xm.q[1] * xm.p[2]
    #xp.f[0] = xp.q[1]
    #xp.f[1] = xp.q[1] * xp.p[1] + grav2 * xp.p[0] * xp.p[0]
    #xp.f[2] = xp.q[1] * xp.p[2]

    ## face normal to y-direction: [hv, huv, hv^2+g(h^2)/2]
    #ym.f[0] = ym.q[2]
    #ym.f[1] = ym.q[2] * ym.p[1]
    #ym.f[2] = ym.q[2] * ym.p[2] + grav2 * ym.p[0] * ym.p[0]
    #yp.f[0] = yp.q[2]
    #yp.f[1] = yp.q[2] * yp.p[1]
    #yp.f[2] = yp.q[2] * yp.p[2] + grav2 * yp.p[0] * yp.p[0]

    #return states

def central_scheme_kernel(ma, pa, mf, pf, mq, pq, flux):
    """For internal use"""
    denominator = pa - ma
    coeff = pa * ma

    with _nplike.errstate(divide="ignore", invalid="ignore"):
        flux = (pa * mf - ma * pf + coeff * (pq - mq)) / denominator;

    zero_ji = (denominator == 0.)  # should we deal with small rounding err here???
    if zero_ji[0].size > 0:
        flux[:, zero_ji] = 0.


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


def get_local_speed_kernel(hp, hm, up, um, g, ap, am):
    """For internal use to mimic CuPy and Cython kernels."""
    ghp = _nplike.sqrt(g * hp);
    ghm = _nplike.sqrt(g * hm);
    ap = _nplike.maximum(_nplike.maximum(up+ghp, um+ghm), 0.0);
    am = _nplike.minimum(_nplike.minimum(up-ghp, um-ghm), 0.0);


def get_local_speed(states, gravity):
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
            gravity,
            states.face.x.plus.a, 
            states.face.x.minus.a,
            )

    get_local_speed_kernel(
            states.face.y.plus.p[0], 
            states.face.y.minus.p[0], 
            states.face.y.plus.p[2], 
            states.face.y.minus.p[2], 
            gravity,
            states.face.y.plus.a, 
            states.face.y.minus.a,
            )

    return states
