"""Idealized unconscious diver: geometry, mass, and rescue configuration.

The victim floats supine and is towed head first. The body axis points
from the feet toward the top of the head (body coordinate ``xi``), ``y'``
points to the victim's left, and ``z'`` points out of the chest (up when
supine). Every part of the diver is a stack of elliptical cross-section
primitives of length ``dxi``:

* the body itself (tissue plus exposure suit), with stations that taper
  from the head through the chest and hips to the feet;
* the scuba cylinder (with valve, first stage, and harness hardware)
  under the back;
* the BCD bladder, either a jacket style (two lobes at the sides of the
  torso) or a back-inflation wing (two lobes behind the back, beside
  the cylinder).

The head pivots at the neck. A static nod angle tilts the head back to
hold the airway open; the head can roll about the neck axis relative to
the torso against a weak spring when the head is unsupported.

All geometric values are idealized round numbers chosen to reproduce a
plausible adult in a 5 mm wetsuit with an aluminum 80 cylinder. None is
a measurement of a particular person.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from .waves import RHO

# Body stations: (xi, half-breadth b, half-thickness c, center z').
# xi = 0 is the base of the neck; positive toward the top of the head.
STATIONS = np.array([
    (-1.80, 0.050, 0.040, -0.030),   # feet in booties
    (-1.62, 0.070, 0.050, -0.030),   # ankles
    (-1.22, 0.120, 0.060, -0.030),   # knees
    (-0.80, 0.165, 0.080, -0.020),   # upper thighs
    (-0.62, 0.175, 0.100, -0.010),   # hips
    (-0.40, 0.155, 0.105, -0.010),   # waist
    (-0.12, 0.175, 0.120, 0.000),    # chest
    (0.000, 0.150, 0.100, 0.000),    # shoulders
    (0.040, 0.060, 0.060, -0.010),   # neck
    (0.080, 0.075, 0.090, 0.000),    # jaw
    (0.170, 0.080, 0.100, 0.000),    # mid head
    (0.270, 0.055, 0.065, 0.000),    # crown
])

NECK_PIVOT = (0.040, -0.030)          # (xi, z') of the cervical pivot
AIRWAY = (0.120, 0.105)               # (xi, z') of the mouth and nose
CYLINDER = dict(xi0=-0.58, xi1=0.02, radius=0.092, net_kg=-2.5)
BCD_SPAN = (-0.50, -0.02)             # bladder extent along the body
FIN_XI = -2.05                        # center of the fin blades
FIN_AREA = 2 * 0.20 * 0.45            # two fin blades (m^2)


@dataclass
class Hydro:
    """Hydrodynamic coefficients of the strip model.

    Parameters
    ----------
    ca : float
        Heave added-mass coefficient relative to ``rho pi b^2 / 2``.
    beta_r : float
        Radiation damping per unit length, ``beta_r rho 2b sqrt(2 g b)``.
    cd : float
        Quadratic heave drag coefficient on the breadth ``2b``.
    beta44 : float
        Linear roll damping, ``beta44 rho b^4 sqrt(g / b)`` per length.
    c44 : float
        Quadratic roll damping, ``c44 rho b^5`` per length.
    cd_fin : float
        Drag coefficient of the fin blades.
    """

    ca: float = 1.0
    beta_r: float = 0.20
    cd: float = 1.0
    beta44: float = 0.08
    c44: float = 0.30
    cd_fin: float = 1.9


@dataclass
class Rescue:
    """The rescue configuration that the rescuer controls.

    Parameters
    ----------
    bcd_liters : float
        Air volume in the victim's BCD (liters).
    bcd_style : str
        ``"jacket"`` (side lobes) or ``"wing"`` (back lobes).
    nod : float
        Head tilt-back angle that opens the airway (rad).
    neck_k : float
        Rotational stiffness resisting head roll (N m / rad). About 2 for
        an unsupported, unconscious head; 40 or more when the rescuer's
        hand cradles the head.
    neck_c : float
        Rotational damping of the neck (N m s / rad).
    tow : str
        ``"valve"`` (cylinder valve, low) or ``"collar"`` (BCD shoulder,
        near the body axis).
    weights_kg : float
        Lead still worn at the hips (kg). Rescue protocol removes it.
    body_density : float
        Mean density of tissue plus suit (kg/m^3).
    airway_margin : float
        Clearance below which the airway is counted as awash (m).
    """

    bcd_liters: float = 12.0
    bcd_style: str = "jacket"
    nod: float = 0.25
    neck_k: float = 2.0
    neck_c: float = 0.6
    tow: str = "collar"
    weights_kg: float = 0.0
    body_density: float = 1005.0
    airway_margin: float = 0.02


@dataclass
class Body:
    """Discretized victim: flat arrays over primitives and strips."""

    xi: np.ndarray        # along-axis coordinate of each primitive (CG frame)
    y0: np.ndarray        # lateral offset of the primitive center
    z0: np.ndarray        # vertical offset of the primitive center
    b: np.ndarray         # semi-axis along y'
    c: np.ndarray         # semi-axis along z'
    dxi: np.ndarray       # primitive length
    mass: np.ndarray      # primitive mass
    head: np.ndarray      # 1 for head primitives
    wet: np.ndarray       # 1 if the primitive carries hydrodynamic loads
    kind: np.ndarray      # 0 body, 1 cylinder, 2 bladder, 3 lead
    strip: np.ndarray     # index into the strip coordinate array
    strip_xi: np.ndarray  # along-axis coordinate of each strip (CG frame)
    pivot: tuple          # (y, z) of the neck pivot in the CG frame
    airway: tuple         # (xi, y, z) of the airway in the CG frame
    airway_strip: int
    fin_strip: int
    fin_xi: float
    cg_shift: tuple       # (xi, z) of the CG in neck coordinates
    rescue: Rescue = field(default_factory=Rescue)
    hydro: Hydro = field(default_factory=Hydro)

    @property
    def total_mass(self):
        """Total mass of the diver and equipment (kg)."""
        return float(self.mass.sum())

    @property
    def displaced_kg(self):
        """Mass of seawater displaced when fully submerged (kg)."""
        return float(RHO * (np.pi * self.b * self.c * self.dxi).sum())


def _interp_stations(xi):
    out = []
    for j in range(1, 4):
        out.append(np.interp(xi, STATIONS[:, 0], STATIONS[:, j]))
    return out


def tow_drag_coefficient():
    """Quadratic tow resistance of the victim, D = k U^2 (N s^2 m^-2)."""
    return 110.0


def build_body(rescue=None, hydro=None, dxi=0.06):
    """Assemble the primitive representation of a towed, supine diver.

    Parameters
    ----------
    rescue : Rescue, optional
        Rescue configuration.
    hydro : Hydro, optional
        Hydrodynamic coefficients.
    dxi : float
        Target strip length (m).

    Returns
    -------
    Body
    """
    rescue = rescue or Rescue()
    hydro = hydro or Hydro()
    lo, hi = STATIONS[0, 0], STATIONS[-1, 0]
    n = int(round((hi - lo) / dxi))
    edges = np.linspace(lo, hi, n + 1)
    xs = 0.5 * (edges[1:] + edges[:-1])
    ds = np.diff(edges)
    bb, cc, zz = _interp_stations(xs)

    prims = []

    def add(xi, y, z, b, c, d, rho, head, wet, kind):
        prims.append((xi, y, z, b, c, d, rho * np.pi * b * c * d, head,
                      wet, kind))

    # Body. Head primitives are rotated by the nod angle about the pivot.
    px, pz = NECK_PIVOT
    cn, sn = np.cos(rescue.nod), np.sin(rescue.nod)
    for x, d, b, c, z in zip(xs, ds, bb, cc, zz):
        head = x > px
        if head:
            rx, rz = x - px, z - pz
            x, z = px + rx * cn + rz * sn, pz - rx * sn + rz * cn
        add(x, 0.0, z, b, c, d, rescue.body_density, int(head), 1, 0)

    # Cylinder under the back.
    cyl = CYLINDER
    sel = (xs > cyl["xi0"]) & (xs < cyl["xi1"])
    vol = np.pi * cyl["radius"] ** 2 * ds[sel].sum()
    rho_cyl = (RHO * vol - cyl["net_kg"]) / vol
    for x, d in zip(xs[sel], ds[sel]):
        bx = np.interp(x, STATIONS[:, 0], STATIONS[:, 2])
        add(x, 0.0, -(bx + cyl["radius"]), cyl["radius"], cyl["radius"], d,
            rho_cyl, 0, 1, 1)

    # BCD bladder, split into two lobes.
    sel = (xs > BCD_SPAN[0]) & (xs < BCD_SPAN[1])
    span = ds[sel].sum()
    area = 1e-3 * max(rescue.bcd_liters, 1e-3) / (2.0 * span)
    r = np.sqrt(area / np.pi)
    for x, d in zip(xs[sel], ds[sel]):
        bx = np.interp(x, STATIONS[:, 0], STATIONS[:, 1])
        cx = np.interp(x, STATIONS[:, 0], STATIONS[:, 2])
        for side in (-1.0, 1.0):
            if rescue.bcd_style == "jacket":
                y, z = side * (bx + 0.7 * r), 0.25 * cx
            elif rescue.bcd_style == "wing":
                y, z = side * (0.6 * CYLINDER["radius"] + r), -(cx + 0.8 * r)
            else:
                raise ValueError("bcd_style must be 'jacket' or 'wing'")
            add(x, y, z, r, r, d, 0.0, 0, 1, 2)

    # Lead at the hips (normally removed during a rescue).
    if rescue.weights_kg > 0.0:
        sel = (xs > -0.72) & (xs < -0.52)
        vol = rescue.weights_kg / 11340.0
        rl = np.sqrt(vol / (2 * np.pi * ds[sel].sum()))
        for x, d in zip(xs[sel], ds[sel]):
            for side in (-1.0, 1.0):
                add(x, side * 0.12, -0.05, rl, rl, d, 11340.0, 0, 1, 3)

    arr = np.array(prims, dtype=float)
    xi, y0, z0, b, c, dd, m, head, wet, kind = arr.T

    # Fin blades (mass is small, folded into the feet).
    m_fins = 2.0
    # Center of gravity, used as the origin of the body frame.
    mt = m.sum() + m_fins
    xg = (np.sum(m * xi) + m_fins * FIN_XI) / mt
    zg = np.sum(m * z0) / mt
    xi = xi - xg
    z0 = z0 - zg

    # Strips: one per distinct along-axis coordinate.
    keys = np.round(xi, 6)
    strip_xi, strip = np.unique(keys, return_inverse=True)
    ax, az = AIRWAY
    rx, rz = ax - px, az - pz
    ax2, az2 = px + rx * cn + rz * sn, pz - rx * sn + rz * cn
    airway = (ax2 - xg, 0.0, az2 - zg)
    strip_xi = np.append(strip_xi, [airway[0], FIN_XI - xg])
    airway_strip = strip_xi.size - 2
    fin_strip = strip_xi.size - 1

    # Fold the fin mass into the most distal body primitive.
    m = m.copy()
    m[np.argmin(xi)] += m_fins

    return Body(xi=xi, y0=y0, z0=z0, b=b, c=c, dxi=dd, mass=m,
                head=head.astype(int), wet=wet.astype(int),
                kind=kind.astype(int), strip=strip.astype(int),
                strip_xi=strip_xi, pivot=(0.0, pz - zg), airway=airway,
                airway_strip=airway_strip, fin_strip=fin_strip,
                fin_xi=FIN_XI - xg, cg_shift=(xg, zg), rescue=rescue,
                hydro=hydro)


def tow_pitch_moment(body, speed):
    """Pitch moment (N m, head-up positive) of the steady tow force.

    The tow force balances the resistance ``k U^2``, which acts near the
    body axis. A tow point below the axis (the cylinder valve, under the
    supine victim's head end) produces a head-up couple; a tow point at
    the BCD shoulder strap produces almost none.
    """
    drag = tow_drag_coefficient() * speed ** 2
    if body.rescue.tow == "valve":
        z_tow = -(0.10 + 2 * CYLINDER["radius"]) - body.cg_shift[1]
    elif body.rescue.tow == "collar":
        z_tow = 0.04 - body.cg_shift[1]
    else:
        raise ValueError("tow must be 'valve' or 'collar'")
    z_drag = -body.cg_shift[1]
    return -(z_tow - z_drag) * drag


def with_rescue(body_or_none=None, **kw):
    """Convenience: build a body with a modified rescue configuration."""
    base = body_or_none.rescue if body_or_none is not None else Rescue()
    hydro = body_or_none.hydro if body_or_none is not None else Hydro()
    return build_body(replace(base, **kw), hydro)
