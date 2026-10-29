"""Drawing of the victim and the rescuer for the animations.

The victim is drawn from the same stations, cylinder, bladder, neck
pivot, nod angle and airway point as the hydrodynamic model, so the
picture shows the body exactly where the simulation places it. The
rescuer is not simulated; it is drawn in the do-si-do position beside
the victim's head, riding the local water surface, with a flutter kick.
"""

import numpy as np

from .body import (AIRWAY, CYLINDER, NECK_PIVOT, STATIONS)

# Palette ------------------------------------------------------------------
SKIN = "#c8977a"
SKIN_DARK = "#9c6f56"
SUIT = "#16191d"
SUIT_PANEL = "#2b6f8f"
SUIT_EDGE = "#060708"
SLEEVE = "#22272d"
BCD = "#2f3439"
BCD_TRIM = "#f2c230"
TANK = "#b9c2c9"
TANK_DARK = "#7d8992"
TANK_BAND = "#f2c230"
FIN = "#f2c230"
FIN_DARK = "#b58a12"
HOSE = "#111315"
R_SUIT = "#1b2433"
R_ACCENT = "#d9472b"
R_BCD = "#7a2a26"
MASK_GLASS = "#9fd8e8"


def _nod(pts, nod):
    """Rotate neck-frame points of the head by the nod angle."""
    px, pz = NECK_PIVOT
    cn, sn = np.cos(nod), np.sin(nod)
    rx, rz = pts[:, 0] - px, pts[:, 1] - pz
    return np.stack([px + rx * cn + rz * sn, pz - rx * sn + rz * cn], 1)


def _smooth(pts, n=6, closed=True):
    """Chaikin corner cutting for organic outlines."""
    p = np.asarray(pts, float)
    for _ in range(n):
        if closed:
            q = np.roll(p, -1, 0)
        else:
            q = p[1:]
            p = p[:-1]
        a = 0.75 * p + 0.25 * q
        b = 0.25 * p + 0.75 * q
        p = np.empty((2 * a.shape[0], 2))
        p[0::2], p[1::2] = a, b
        if not closed:
            p = np.vstack([p, q[-1:]])
    return p


def _station(xi, j):
    return np.interp(xi, STATIONS[:, 0], STATIONS[:, j])


HEAD_PROFILE = np.array([
    (0.000, 0.055), (0.030, 0.058), (0.055, 0.064), (0.070, 0.086),
    (0.088, 0.094), (0.100, 0.093), (0.110, 0.100), (0.120, 0.104),
    (0.135, 0.128), (0.150, 0.110), (0.162, 0.100), (0.176, 0.096),
    (0.192, 0.106), (0.215, 0.098), (0.243, 0.072), (0.268, 0.022),
    (0.266, -0.040), (0.240, -0.082), (0.180, -0.104), (0.110, -0.090),
    (0.050, -0.066), (0.000, -0.070)])

FACE_PATCH = np.array([
    (0.068, 0.086), (0.088, 0.094), (0.100, 0.093), (0.110, 0.100),
    (0.120, 0.104), (0.135, 0.128), (0.150, 0.110), (0.162, 0.100),
    (0.176, 0.096), (0.192, 0.106), (0.212, 0.099), (0.205, 0.066),
    (0.160, 0.050), (0.105, 0.052), (0.075, 0.064)])


def victim_shapes(body):
    """Static outline polygons of the victim in the CG frame (xi, z').

    Returns
    -------
    list of dict
        Keys ``pts`` (N, 2), ``fc``, ``ec``, ``lw``, ``z`` (draw order),
        ``alpha``.
    """
    nod = body.rescue.nod
    xg, zg = body.cg_shift
    shapes = []

    def add(pts, fc, ec="none", lw=0.0, z=0, alpha=1.0, head=False,
            smooth=True, closed=True):
        p = np.asarray(pts, float)
        if smooth:
            p = _smooth(p, 4, closed)
        if head:
            p = _nod(p, nod)
        p = p - np.array([xg, zg])
        shapes.append(dict(pts=p, fc=fc, ec=ec, lw=lw, z=z, alpha=alpha,
                           closed=closed))

    # Fins first (behind the legs).
    for dz, col in ((0.012, FIN_DARK), (-0.006, FIN)):
        blade = [(-1.70, 0.012 + dz), (-1.86, 0.03 + dz), (-2.20, 0.0 + dz),
                 (-2.42, -0.045 + dz), (-2.44, -0.075 + dz),
                 (-2.20, -0.055 + dz), (-1.86, -0.045 + dz),
                 (-1.70, -0.040 + dz)]
        add(blade, col, "#3a2c05", 0.4, 1)

    # Body silhouette: feet to shoulders, front edge then back edge.
    xs = np.linspace(-1.80, 0.045, 140)
    c = _station(xs, 2)
    zc = _station(xs, 3)
    bump_f = (0.016 * np.exp(-((xs + 0.16) / 0.10) ** 2)
              + 0.010 * np.exp(-((xs + 1.22) / 0.04) ** 2)
              - 0.012 * np.exp(-((xs + 0.55) / 0.08) ** 2))
    bump_b = (0.022 * np.exp(-((xs + 0.70) / 0.09) ** 2)
              + 0.014 * np.exp(-((xs + 1.40) / 0.09) ** 2)
              + 0.010 * np.exp(-((xs + 1.74) / 0.03) ** 2))
    top = zc + c + bump_f
    bot = zc - c - bump_b
    outline = np.vstack([np.stack([xs, top], 1),
                         np.stack([xs[::-1], bot[::-1]], 1)])
    add(outline, SUIT, SUIT_EDGE, 0.6, 3, smooth=False)
    # Side panel stripe along the suit.
    stripe = np.vstack([np.stack([xs[30:120], zc[30:120] + 0.10 * c[30:120]],
                                 1),
                        np.stack([xs[119:29:-1],
                                  zc[119:29:-1] - 0.05 * c[119:29:-1]], 1)])
    add(stripe, SUIT_PANEL, "none", 0, 4, alpha=0.85, smooth=False)

    # Cylinder under the back, valve and first stage at the head end.
    r = CYLINDER["radius"]
    zt = -(0.115 + r)
    x0, x1 = CYLINDER["xi0"], CYLINDER["xi1"]
    th = np.linspace(-np.pi / 2, np.pi / 2, 20)
    cyl = np.vstack([np.stack([x1 - 0.04 + 0.04 * np.cos(th),
                               zt + r * np.sin(th)], 1),
                     np.stack([x0 + 0.05 - 0.05 * np.cos(th[::-1]),
                               zt + r * np.sin(th[::-1])], 1)])
    add(cyl, TANK, "#59636b", 0.6, 2, smooth=False)
    add([(x0 + 0.02, zt - 0.2 * r), (x1 - 0.03, zt - 0.2 * r),
         (x1 - 0.03, zt - 0.8 * r), (x0 + 0.03, zt - 0.8 * r)],
        TANK_DARK, "none", 0, 2, alpha=0.6, smooth=False)
    add([(x1 - 0.16, zt + r), (x1 - 0.11, zt + r), (x1 - 0.11, zt - r),
         (x1 - 0.16, zt - r)], TANK_BAND, "none", 0, 2, smooth=False)
    add([(x1, zt + 0.025), (x1 + 0.06, zt + 0.025), (x1 + 0.06, zt - 0.025),
         (x1, zt - 0.025)], "#3d4348", "#111", 0.4, 2, smooth=False)
    add([(x1 + 0.03, zt + 0.06), (x1 + 0.09, zt + 0.06),
         (x1 + 0.09, zt - 0.02), (x1 + 0.03, zt - 0.02)], "#2a2e32",
        "#111", 0.4, 2)

    # BCD jacket around the torso, puffed by the bladder volume.
    puff = 0.012 + 0.03 * min(body.rescue.bcd_liters, 20.0) / 12.0
    xb = np.linspace(-0.52, -0.01, 40)
    cb = _station(xb, 2)
    zb = _station(xb, 3)
    env = np.sin(np.pi * (xb + 0.52) / 0.51) ** 0.6
    if body.rescue.bcd_style == "jacket":
        btop = zb + cb + 0.012 + puff * env
        bbot = zb - cb - 0.020
    else:
        btop = zb + cb + 0.010
        bbot = zb - cb - 0.025 - 1.6 * puff * env
    bcd = np.vstack([np.stack([xb, btop], 1),
                     np.stack([xb[::-1], bbot[::-1]], 1)])
    add(bcd, BCD, "#0c0e10", 0.6, 5, smooth=False)
    add(np.stack([xb, btop - 0.006], 1), "none", BCD_TRIM, 1.0, 6,
        smooth=False, closed=False)
    add([(-0.30, btop[17] - 0.004), (-0.27, btop[19] - 0.004),
         (-0.27, zb[19] - cb[19]), (-0.30, zb[17] - cb[17])], "#0b0c0d",
        "none", 0, 6, smooth=False)

    # Near arm resting along the body.
    arm = [(-0.02, 0.055), (-0.10, 0.060), (-0.30, 0.040), (-0.33, 0.000),
           (-0.58, 0.020), (-0.66, 0.010), (-0.68, -0.025), (-0.60, -0.035),
           (-0.34, -0.040), (-0.30, -0.030), (-0.10, -0.030), (0.00, -0.015)]
    add(arm, SLEEVE, SUIT_EDGE, 0.5, 7)
    add([(-0.66, 0.014), (-0.74, 0.010), (-0.76, -0.012), (-0.68, -0.026)],
        "#101214", "none", 0, 7)

    # Inflator hose over the shoulder.
    hose = [(-0.04, 0.16), (0.02, 0.22), (0.00, 0.26), (-0.06, 0.24),
            (-0.14, 0.17)]
    add(hose, "none", HOSE, 2.6, 8, closed=False)

    # Head: hood, face, closed eye, lips.
    add(HEAD_PROFILE, SUIT, SUIT_EDGE, 0.6, 9, head=True)
    add(FACE_PATCH, SKIN, SKIN_DARK, 0.5, 10, head=True)
    add([(0.168, 0.094), (0.176, 0.097), (0.184, 0.094)], "none", "#3b2a22",
        0.9, 11, head=True, closed=False)
    add([(0.097, 0.094), (0.104, 0.097), (0.111, 0.099)], "none", "#6e3d33",
        0.9, 11, head=True, closed=False)
    return shapes


def airway_point(body):
    """Airway position in the CG frame (xi, z')."""
    p = _nod(np.array([AIRWAY]), body.rescue.nod)[0]
    return p - np.array(body.cg_shift)


def to_screen(pts, q):
    """Project CG-frame (xi, z') points into the side view for pose q.

    The roll angle does not change the side projection beyond a small
    foreshortening of the vertical offsets, ``z' cos(phi)``, which is
    applied so that a rolling body visibly flattens.
    """
    zeta, th = q[0], q[1]
    cphi = np.cos(q[2])
    xi, z = pts[:, 0], pts[:, 1] * (0.55 + 0.45 * abs(cphi))
    ct, st = np.cos(th), np.sin(th)
    return np.stack([xi * ct - z * st, zeta + xi * st + z * ct], 1)


def rescuer_shapes(t, kick_period=1.1):
    """Outline polygons of the rescuer in a frame centered on its head.

    The rescuer faces +x, lies at about 35 degrees below horizontal from
    the head toward the fins, and kicks with a flutter kick.

    Returns
    -------
    list of dict
        As for :func:`victim_shapes`, plus a ``hand`` anchor at the
        shoulder from which the arm is drawn.
    """
    shapes = []

    def add(pts, fc, ec="none", lw=0.0, z=0, alpha=1.0, smooth=True,
            closed=True):
        p = np.asarray(pts, float)
        if smooth:
            p = _smooth(p, 4, closed)
        shapes.append(dict(pts=p, fc=fc, ec=ec, lw=lw, z=z, alpha=alpha,
                           closed=closed))

    ang = np.radians(-145.0)
    u = np.array([np.cos(ang), np.sin(ang)])     # head -> feet
    nrm = np.array([-u[1], u[0]])                 # toward the back
    hip = np.array([-0.02, -0.12]) + 0.62 * u
    phase = 2 * np.pi * t / kick_period
    for side, sgn, col in ((0, 1.0, "#0f141c"), (1, -1.0, R_SUIT)):
        swing = sgn * 0.32 * np.sin(phase)
        knee = hip + 0.42 * (np.cos(swing * 0.6) * u
                             + np.sin(swing * 0.6) * nrm)
        ankle = knee + 0.42 * (np.cos(swing) * u + np.sin(swing) * nrm)
        dk = knee - hip
        dk = dk / np.linalg.norm(dk)
        da = ankle - knee
        da = da / np.linalg.norm(da)
        nk = np.array([-dk[1], dk[0]])
        na = np.array([-da[1], da[0]])
        leg = [hip + 0.075 * nk, knee + 0.055 * nk, ankle + 0.04 * na,
               ankle - 0.04 * na, knee - 0.055 * nk, hip - 0.075 * nk]
        add(leg, col, "#05070a", 0.5, 1 + side)
        flex = 0.35 * np.cos(phase + 0.6) * sgn
        tip_dir = np.cos(flex) * da + np.sin(flex) * na
        blade = [ankle + 0.05 * na, ankle + 0.62 * tip_dir + 0.07 * na,
                 ankle + 0.64 * tip_dir - 0.06 * na, ankle - 0.05 * na]
        add(blade, "#33506b" if side else "#22384d", "#0b1520", 0.4,
            1 + side, smooth=False)
        add([ankle + 0.30 * tip_dir + 0.02 * na,
             ankle + 0.58 * tip_dir + 0.04 * na,
             ankle + 0.58 * tip_dir + 0.0 * na,
             ankle + 0.30 * tip_dir - 0.01 * na], R_ACCENT, "none", 0,
            1 + side, alpha=0.8, smooth=False)

    # Torso with BCD and cylinder on the back.
    sh = np.array([-0.02, -0.12])
    torso = [sh + 0.13 * nrm, hip + 0.12 * nrm, hip - 0.13 * nrm,
             sh - 0.12 * nrm]
    add(torso, R_SUIT, "#05070a", 0.6, 3)
    tank = [sh + 0.10 * nrm + 0.06 * u, hip + 0.10 * nrm - 0.04 * u,
            hip + 0.27 * nrm - 0.04 * u, sh + 0.27 * nrm + 0.06 * u]
    add(tank, TANK, "#59636b", 0.6, 2)
    bcd = [sh + 0.13 * nrm + 0.02 * u, sh + 0.13 * nrm + 0.36 * u,
           sh - 0.13 * nrm + 0.36 * u, sh - 0.12 * nrm + 0.02 * u]
    add(bcd, R_BCD, "#3a0f08", 0.6, 4, alpha=0.95)

    # Head with hood, mask and snorkel.
    th = np.linspace(0, 2 * np.pi, 40)
    add(np.stack([0.10 * np.cos(th), 0.118 * np.sin(th)], 1), "#0e1218",
        "#05070a", 0.6, 5, smooth=False)
    add([(0.055, -0.035), (0.125, -0.030), (0.128, 0.055), (0.060, 0.060)],
        "#141414", "#000", 0.6, 6)
    add([(0.070, -0.022), (0.118, -0.018), (0.120, 0.045), (0.072, 0.048)],
        MASK_GLASS, "none", 0, 7, alpha=0.9)
    add([(0.082, 0.030), (0.095, 0.040), (0.100, 0.028)], "none", "white",
        1.2, 8, closed=False)
    add([(-0.07, 0.02), (-0.08, 0.30), (-0.05, 0.34), (-0.02, 0.33)],
        "none", "#f08a24", 3.0, 4, closed=False)
    return shapes, sh


def front_view_shapes(body, ysign=1.0):
    """Cross-section outlines seen from the crown, for the airway inset.

    Returns head and chest sections in the (y', z') plane about the
    neck pivot and the body axis; the caller applies roll.
    """
    th = np.linspace(0, 2 * np.pi, 80)
    xh = 0.17
    bh, ch = _station(xh, 1), _station(xh, 2)
    head = np.stack([bh * np.cos(th), ch * np.sin(th)], 1)
    xc = -0.15
    bc, cc = _station(xc, 1), _station(xc, 2)
    chest = np.stack([bc * np.cos(th), cc * np.sin(th)], 1)
    nose = np.array([(-0.018, ch - 0.01), (0.0, ch + 0.028),
                     (0.018, ch - 0.01)])
    return dict(head=head, chest=chest, nose=nose,
                eyes=[(-0.032, ch - 0.012), (0.032, ch - 0.012)],
                cyl_r=CYLINDER["radius"], cyl_z=-(cc + CYLINDER["radius"]))
