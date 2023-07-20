# vim:fenc=utf-8
# vim:ft=pyrex
from torchswe import nplike as _nplike

def func_xmf0_xmf2(xmf0, xmf2,xmq1, xmp2):
    xmf0 = xmq1
    xmf2 = xmq1 * xmp2

    return xmf0, xmf2

def func_xmf1( xmf1, grav2,  xmq1, xmp0, xmp1):
    xmf1 = xmq1 * xmp1 + grav2 * xmp0 * xmp0

    return xmf1

def func_xpf0_xpf2(xpf0, xpf2, xpq1, xpp2):
    xpf0 = xpq1
    xpf2 = xpq1 * xpp2

    return xpf0, xpf2

def func_xpf1(xpf1, grav2, xpq1, xpp0, xpp1):
    xpf1 = xpq1 * xpp1 + grav2 * xpp0 * xpp0

    return xpf1

def func_ypf0_ypf1(ypf0,ypf1,ypq2, ypp1) :
    ypf0 = ypq2
    ypf1 = ypq2 * ypp1

    return ypf0, ypf1

def func_ypf2(ypf2,grav2, ypq2, ypp0, ypp2) :
    ypf2 = ypq2 * ypp2 + grav2 * ypp0 * ypp0

    return ypf2

def func_ymf0_ymf1(ymf0, ymf1, ymq2, ymp1):
    ymf0 = ymq2
    ymf1 = ymq2 * ymp1

    return ymf0, ymf1

def func_ymf2(ymf2, grav2, ymq2, ymp0, ymp2):
    ymf2 = ymq2 * ymp2 + grav2 * ymp0 * ymp0

    return ymf2


def central_scheme_kernel(flux, ma, pa, mf, pf, mq, pq):
    diff = pa - ma
    if diff == 0.0:
        flux = 0.0
    else:
        flux = (pa * mf - ma * pf + pa * ma * (pq - mq)) / diff;
    return flux


def get_local_speed_kernel(ap, am, hp, hm, up, um, g):

    ghp = (g * hp) ** 0.5
    ghm = (g * hm) ** 0.5

    ap = max(max(up + ghp, um + ghm), 0.0)
    am = min(min(up - ghp, um - ghp), 0.0)

    return ap, am


vfunc_xmf0_xmf2 = _nplike.vectorize(func_xmf0_xmf2, otypes=(float,float), cache=True)
vfunc_xmf1      = _nplike.vectorize(func_xmf1, otypes=(float,),cache=True)
vfunc_xpf0_xpf2 = _nplike.vectorize(func_xpf0_xpf2,otypes=(float,float), cache=True)
vfunc_xpf1      = _nplike.vectorize(func_xpf1, otypes=(float,),cache=True)

vfunc_ypf0_ypf1 = _nplike.vectorize(func_ypf0_ypf1, otypes=(float,float), cache=True)
vfunc_xpf1      = _nplike.vectorize(func_ypf2, otypes=(float,),cache=True)
vfunc_ymf0_ymf1 = _nplike.vectorize(func_ymf0_ymf1,otypes=(float,float), cache=True)
vfunc_ymf2      = _nplike.vectorize(func_ymf2, otypes=(float,),cache=True)

vfunc_central_scheme_kernel = _nplike.vectorize(central_scheme_kernel, otypes=(float,), cache=True)
vfunc_get_local_speed_kernel = _nplike.vectorize(get_local_speed_kernel, cache=True)


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
    grav2 = gravity/2.0

    xm.f[0], xm.f[2] = vfunc_xmf0_xmf2(xm.f[0], xm.f[2],xm.q[1], xm.p[2])
    xm.f[1]          = vfunc_xmf1(xm.f[1], grav2, xm.q[1], xm.p[0], xm.p[1])
    xp.f[0], xp.f[2] = vfunc_xpf0_xpf2(xp.f[0], xp.f[2], xp.q[1], xp.p[2])
    xp.f[1]          = vfunc_xpf1(xp.f[1], grav2, xp.q[1], xp.p[0], xp.p[1])

    yp.f[0], yp.f[1] = vfunc_ypf0_ypf1(yp.f[0], yp.f[1], yp.q[2],yp.p[1])
    yp.f[2]          = vfunc_xpf1(yp.f[2], grav2, yp.q[2], yp.p[0], yp.p[2])
    ym.f[0], ym.f[1] = vfunc_ymf0_ymf1(ym.f[0], ym.f[1], ym.q[2], ym.p[1])
    ym.f[2]          = vfunc_ymf2(ym.f[2], grav2, ym.q[2], ym.p[0], ym.p[2])

    return states


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

    x.cf[0] = vfunc_central_scheme_kernel(x.cf[0], xm.a, xp.a, xm.f[0], xp.f[0], xm.q[0], xp.q[0])
    x.cf[1] = vfunc_central_scheme_kernel(x.cf[1], xm.a, xp.a, xm.f[1], xp.f[1], xm.q[1], xp.q[1])
    x.cf[2] = vfunc_central_scheme_kernel(x.cf[2], xm.a, xp.a, xm.f[2], xp.f[2], xm.q[2], xp.q[2])

    y.cf[0] = vfunc_central_scheme_kernel(y.cf[0], ym.a, yp.a, ym.f[0], yp.f[0], ym.q[0], yp.q[0])
    y.cf[1] = vfunc_central_scheme_kernel(y.cf[1], ym.a, yp.a, ym.f[1], yp.f[1], ym.q[1], yp.q[1])
    y.cf[2] = vfunc_central_scheme_kernel(y.cf[2], ym.a, yp.a, ym.f[2], yp.f[2], ym.q[2], yp.q[2])

    return states


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
    states.face.x.plus.a, states.face.x.minus.a = vfunc_get_local_speed_kernel(
            states.face.x.plus.a, 
            states.face.x.minus.a,
            states.face.x.plus.p[0],
            states.face.x.minus.p[0],
            states.face.x.plus.p[1], 
            states.face.x.minus.p[1], 
            gravity)

    states.face.y.plus.a, states.face.y.minus.a = vfunc_get_local_speed_kernel(
            states.face.y.plus.a, 
            states.face.y.minus.a, 
            states.face.y.plus.p[0], 
            states.face.y.minus.p[0], 
            states.face.y.plus.p[2], 
            states.face.y.minus.p[2], 
            gravity)

    return states
