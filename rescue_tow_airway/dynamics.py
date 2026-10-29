"""Model M1: nonlinear 2.5D strip-theory dynamics of the towed victim.

Generalized coordinates ``q = (zeta, theta, phi, chi)``:

* ``zeta``  heave of the center of gravity (m, up positive);
* ``theta`` pitch (rad, head up positive);
* ``phi``   roll of the torso about the body axis (rad, toward the
  victim's left positive);
* ``chi``   roll of the head relative to the torso about the neck axis.

Surge is prescribed by the tow speed and sway and yaw are held by the
rescuer. For each cross-section primitive the model evaluates

* buoyancy and the Froude-Krylov inertia force,
  ``rho (g + a_w) A_sub dxi``, where ``A_sub`` is the exact area of the
  rotated ellipse below the local free surface, itself a straight line
  with the local elevation and cross-axis slope;
* added mass ``a' = c_a rho pi b^2 / 2``, radiation damping and
  quadratic drag, all acting on the velocity of the section relative to
  the vertical particle velocity of the water, and all scaled by the
  submerged fraction of the section;
* gravity at the primitive center.

Generalized forces follow from the vertical-velocity Jacobian of each
point; the mass matrix includes the three translational Jacobians of
every primitive plus section roll inertias. Velocity-quadratic
(Coriolis) terms are neglected, which is consistent with the small
angular rates of a floating body.
"""

import numpy as np
from scipy.optimize import fsolve

from .body import FIN_AREA, tow_pitch_moment
from .waves import G, RHO

NQ = 4


def _ellipse_below_line(yc, zc, b, c, alpha, eta, s):
    """Area and centroid of a rotated ellipse below the line z = eta + s y.

    The ellipse is mapped affinely to the unit disk, where the region
    below the line is a circular segment with closed-form area and
    centroid; both map back exactly.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    rn_y = -s * ca + sa
    rn_z = s * sa + ca
    my, mz = b * rn_y, c * rn_z
    mn = np.sqrt(my * my + mz * mz)
    d = (eta - (zc - s * yc)) / mn
    dc = np.clip(d, -1.0, 1.0)
    root = np.sqrt(1.0 - dc * dc)
    acap = np.arccos(dc) - dc * root
    asub = np.pi - acap
    gfac = -(2.0 / 3.0) * root ** 3 / np.maximum(asub, 1e-12)
    uy, uz = my / mn * gfac, mz / mn * gfac
    oy = ca * b * uy - sa * c * uz
    oz = sa * b * uy + ca * c * uz
    return asub * b * c, yc + oy, zc + oz, asub / np.pi


def kinematics(q, body):
    """Positions of primitive centers in the inertial (towed) frame.

    Parameters
    ----------
    q : ndarray (R, 4)
    body : Body

    Returns
    -------
    dict of ndarray (R, P)
        ``x2`` along-axis lever, ``yg`` lateral position, ``zg`` vertical
        position, ``z1`` vertical offset after roll, ``alpha`` section
        roll angle, plus pivot positions (R, 1).
    """
    zeta, th, ph, chi = (q[:, i:i + 1] for i in range(NQ))
    head = body.head[None, :]
    py, pz = body.pivot
    ch, sh = np.cos(chi * head), np.sin(chi * head)
    y = py + ch * (body.y0 - py) - sh * (body.z0 - pz)
    z = pz + sh * (body.y0 - py) + ch * (body.z0 - pz)
    cp, sp = np.cos(ph), np.sin(ph)
    y1 = y * cp - z * sp
    z1 = y * sp + z * cp
    ct, st = np.cos(th), np.sin(th)
    x2 = body.xi * ct - z1 * st
    z2 = body.xi * st + z1 * ct
    ypiv = py * cp - pz * sp
    zpiv = py * sp + pz * cp
    return dict(x2=x2, yg=y1, zg=zeta + z2, z1=z1, ct=ct, st=st,
                alpha=ph + chi * head, ypiv=ypiv, zpiv=zpiv)


def point_height(q, body, xi, y0, z0, is_head):
    """Global lateral and vertical position of one body-fixed point.

    Returns
    -------
    yg, zg : ndarray (R,)
    """
    zeta, th, ph, chi = (q[:, i] for i in range(NQ))
    py, pz = body.pivot
    if is_head:
        ch, sh = np.cos(chi), np.sin(chi)
        y = py + ch * (y0 - py) - sh * (z0 - pz)
        z = pz + sh * (y0 - py) + ch * (z0 - pz)
    else:
        y, z = np.full_like(zeta, y0), np.full_like(zeta, z0)
    cp, sp = np.cos(ph), np.sin(ph)
    y1 = y * cp - z * sp
    z1 = y * sp + z * cp
    zg = zeta + xi * np.sin(th) + z1 * np.cos(th)
    return y1, zg


def generalized_forces(q, qd, eta, w, acc, slope, body, speed=0.0,
                       lin_drag=None, lin_roll=None, still=False):
    """Generalized forces and mass matrix of model M1.

    Parameters
    ----------
    q, qd : ndarray (R, 4)
        Coordinates and velocities.
    eta, w, acc, slope : ndarray (R, J)
        Surface kinematics at the strips.
    body : Body
    speed : float
        Tow speed through the water (m/s), sets the tow pitch moment.
    lin_drag, lin_roll : ndarray (P,) or None
        Equivalent linear coefficients that replace the quadratic heave
        drag and quadratic roll damping (used by the linear model).
    still : bool
        Ignore the water acceleration in the buoyancy (static checks).

    Returns
    -------
    Q : ndarray (R, 4)
        Generalized forces (including tow and neck terms).
    M : ndarray (R, 4, 4)
        Total mass matrix, rigid body plus added mass.
    """
    hy = body.hydro
    kin = kinematics(q, body)
    st_idx = body.strip
    e_p = eta[:, st_idx]
    w_p = w[:, st_idx]
    a_p = acc[:, st_idx]
    s_p = slope[:, st_idx]
    ct = kin["ct"]
    alpha = kin["alpha"]
    area, yb, zb, frac = _ellipse_below_line(kin["yg"], kin["zg"], body.b,
                                             body.c, alpha, e_p, s_p)
    head = body.head[None, :]
    dxi = body.dxi[None, :]

    # Vertical Jacobians at the primitive center and at the centroid.
    jc = np.stack([np.ones_like(kin["x2"]), kin["x2"], kin["yg"] * ct,
                   head * (kin["yg"] - kin["ypiv"]) * ct], -1)
    jb = np.stack([np.ones_like(kin["x2"]), kin["x2"], yb * ct,
                   head * (yb - kin["ypiv"]) * ct], -1)

    # Buoyancy and Froude-Krylov inertia.
    gacc = G if still else G + a_p
    fb = RHO * gacc * area * dxi * body.wet[None, :]

    # Relative vertical velocity and section hydrodynamics.
    wb = np.einsum("rpk,rk->rp", jc, qd)
    wr = wb - w_p
    beff = np.sqrt((body.b * np.cos(alpha)) ** 2
                   + (body.c * np.sin(alpha)) ** 2)
    wetf = frac * body.wet[None, :]
    am = hy.ca * RHO * 0.5 * np.pi * beff ** 2 * wetf
    brad = hy.beta_r * RHO * 2 * beff * np.sqrt(2 * G * beff) * wetf
    if lin_drag is None:
        drag = 0.5 * RHO * hy.cd * 2 * beff * np.abs(wr) * wr * wetf
    else:
        drag = lin_drag[None, :-1] * wr * wetf
    fh = (-brad * wr - drag + am * a_p) * dxi
    fg = -body.mass[None, :] * G

    Q = ((fg + fh)[:, None, :] @ jc)[:, 0] + (fb[:, None, :] @ jb)[:, 0]

    # Roll damping of the body sections (about their own axes).
    isbody = (body.kind == 0)[None, :]
    om = qd[:, 2:3] + head * qd[:, 3:4]
    b44 = hy.beta44 * RHO * beff ** 4 * np.sqrt(G / beff)
    if lin_roll is None:
        q44 = hy.c44 * RHO * beff ** 5 * np.abs(om) * om
    else:
        q44 = lin_roll[None, :] * om
    mr = -(b44 * om + q44) * wetf * dxi * isbody
    Q[:, 2] += mr.sum(1)
    Q[:, 3] += (mr * head).sum(1)

    # Fin blades: drag on the relative heave velocity at the feet.
    fs = body.fin_strip
    wfin = qd[:, 0] + body.fin_xi * ct[:, 0] * qd[:, 1] - w[:, fs]
    if lin_drag is None:
        ffin = -0.5 * RHO * hy.cd_fin * FIN_AREA * np.abs(wfin) * wfin
    else:
        ffin = -lin_drag[-1] * wfin
    Q[:, 0] += ffin
    Q[:, 1] += ffin * body.fin_xi * ct[:, 0]

    # Neck spring and damper; tow couple.
    Q[:, 3] -= body.rescue.neck_k * q[:, 3] + body.rescue.neck_c * qd[:, 3]
    Q[:, 1] += tow_pitch_moment(body, speed)

    # Mass matrix: translational Jacobians plus section roll inertia.
    zero = np.zeros_like(kin["x2"])
    z1 = kin["z1"]
    zp = kin["zpiv"]
    jy = np.stack([zero, zero, -z1, head * -(z1 - zp)], -1)
    jx = np.stack([zero, -(kin["zg"] - q[:, :1]), -kin["st"] * kin["yg"],
                   head * -kin["st"] * (kin["yg"] - kin["ypiv"])], -1)
    sm = np.sqrt(body.mass)[None, :, None]
    isec = body.mass * (body.b ** 2 + body.c ** 2) / 4.0
    jw = np.stack([zero, zero, np.ones_like(zero), head + zero], -1)
    stack = np.concatenate([jc * sm, jy * sm, jx * sm,
                            jw * np.sqrt(isec)[None, :, None],
                            jc * np.sqrt(am * dxi)[:, :, None]], 1)
    M = np.swapaxes(stack, 1, 2) @ stack
    return Q, M


def airway_clearance(q, eta_a, slope_a, body):
    """Height of the airway above the local water surface (m).

    The water level at the airway accounts for the cross-axis slope at
    the airway's lateral position. The awash margin is subtracted.
    """
    xa, ya, za = body.airway
    yg, zg = point_height(q, body, xa, ya, za, True)
    return zg - (eta_a + slope_a * yg) - body.rescue.airway_margin


def equilibrium(body, speed=0.0):
    """Static floating position in still water with the tow couple.

    Returns
    -------
    ndarray (4,)
        Equilibrium coordinates (roll and head roll are zero by symmetry).
    """
    nj = body.strip_xi.size
    zero = np.zeros((1, nj))

    def res(x):
        q = np.array([[x[0], x[1], 0.0, 0.0]])
        Q, _ = generalized_forces(q, np.zeros((1, NQ)), zero, zero, zero,
                                  zero, body, speed, still=True)
        return [Q[0, 0] / 1000.0, Q[0, 1] / 100.0]

    best = None
    for z0 in (-0.2, -0.1, -0.3, 0.0, -0.45):
        for t0 in (0.3, 0.1, 0.5, 0.0, 0.8):
            sol, info, ier, msg = fsolve(res, [z0, t0], full_output=True)
            err = np.abs(res(sol)).max()
            if err < 1e-6 and abs(sol[1]) < 1.3:
                return np.array([sol[0], sol[1], 0.0, 0.0])
            if best is None or err < best[1]:
                best = (sol, err)
    raise RuntimeError("no floating equilibrium (residual %.3g); the diver "
                       "is negatively buoyant" % best[1])


def simulate(body, fields, dt, speed=0.0, q0=None, ramp=10.0):
    """Integrate M1 with the classical fourth-order Runge-Kutta scheme.

    Parameters
    ----------
    body : Body
    fields : tuple of ndarray (R, J, 2N+1)
        ``eta, w, acc, slope`` sampled every ``dt / 2``.
    dt : float
        Time step (s).
    speed : float
        Tow speed through the water (m/s).
    q0 : ndarray (4,), optional
        Initial coordinates; defaults to the still-water equilibrium.
    ramp : float
        Duration of the smooth start of the wave forcing (s).

    Returns
    -------
    dict
        ``t`` (N+1,), ``q`` (R, N+1, 4), ``qd`` (R, N+1, 4),
        ``clear`` airway clearance (R, N+1), ``water`` water level at
        the airway (R, N+1).
    """
    eta, w, acc, slope = fields
    nr, nj, nt = eta.shape
    nsteps = (nt - 1) // 2
    th = np.arange(nt) * 0.5 * dt
    env = np.where(th < ramp, np.sin(0.5 * np.pi * th / ramp) ** 2, 1.0) \
        if ramp > 0 else np.ones(nt)
    if q0 is None:
        q0 = equilibrium(body, speed)
    q = np.tile(q0, (nr, 1)).astype(float)
    qd = np.zeros((nr, NQ))
    qs = np.empty((nr, nsteps + 1, NQ))
    qds = np.empty((nr, nsteps + 1, NQ))
    qs[:, 0], qds[:, 0] = q, qd

    def rhs(qq, vv, m):
        e = env[m]
        Q, M = generalized_forces(qq, vv, e * eta[:, :, m], e * w[:, :, m],
                                  e * acc[:, :, m], e * slope[:, :, m],
                                  body, speed)
        return vv, np.linalg.solve(M, Q[..., None])[..., 0]

    for n in range(nsteps):
        m0, m1, m2 = 2 * n, 2 * n + 1, 2 * n + 2
        k1q, k1v = rhs(q, qd, m0)
        k2q, k2v = rhs(q + 0.5 * dt * k1q, qd + 0.5 * dt * k1v, m1)
        k3q, k3v = rhs(q + 0.5 * dt * k2q, qd + 0.5 * dt * k2v, m1)
        k4q, k4v = rhs(q + dt * k3q, qd + dt * k3v, m2)
        q = q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
        qd = qd + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        qs[:, n + 1], qds[:, n + 1] = q, qd

    idx = np.arange(0, nt, 2)
    ea = env[idx][None, :] * eta[:, body.airway_strip, idx]
    sa = env[idx][None, :] * slope[:, body.airway_strip, idx]
    clear = np.empty((nr, nsteps + 1))
    for n in range(nsteps + 1):
        clear[:, n] = airway_clearance(qs[:, n], ea[:, n], sa[:, n], body)
    return dict(t=idx * 0.5 * dt, q=qs, qd=qds, clear=clear, water=ea)
