"""Floating equilibrium, natural modes and still-water roll stability.

* Equilibrium heave, pitch and airway clearance against BCD volume for
  both tow points (jacket BCD).
* Natural periods and damping ratios of the linearized body.
* Release from a 0.15 rad roll in still water for jacket and wing BCDs,
  showing whether the victim self-rights or rolls face down.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.body import Rescue, build_body  # noqa: E402
from rescue_tow_airway.dynamics import equilibrium, simulate  # noqa: E402
from rescue_tow_airway.dynamics import airway_clearance  # noqa: E402
from rescue_tow_airway.io import save_cache  # noqa: E402
from rescue_tow_airway.linear import LinearModel  # noqa: E402

VOLUMES = np.arange(7.0, 24.01, 1.0)


def main():
    out = {}
    for tow in ("collar", "valve"):
        for u in (0.0, 0.5):
            z, t, c = [], [], []
            for v in VOLUMES:
                b = build_body(Rescue(bcd_liters=v, tow=tow))
                try:
                    q0 = equilibrium(b, u)
                    z.append(q0[0])
                    t.append(q0[1])
                    c.append(airway_clearance(q0[None], np.zeros(1),
                                              np.zeros(1), b)[0])
                except RuntimeError:
                    z.append(np.nan)
                    t.append(np.nan)
                    c.append(np.nan)
            key = "%s_%02d" % (tow, int(10 * u))
            out["heave_" + key] = np.array(z)
            out["pitch_" + key] = np.array(t)
            out["clear_" + key] = np.array(c)

    # Natural modes of the default configuration.
    lm = LinearModel(build_body(), 0.5)
    wn, zeta, lam = lm.eigen()
    out["eig_wn"], out["eig_zeta"] = wn, zeta

    # Roll release in still water.
    dt, nt = 0.04, 1500
    t = np.arange(nt + 1) * dt
    cases = [("jacket", 12.0, 2.0), ("jacket", 12.0, 40.0),
             ("wing", 12.0, 2.0), ("wing", 18.0, 2.0)]
    rolls, heads, clears = [], [], []
    for style, vol, kn in cases:
        b = build_body(Rescue(bcd_liters=vol, bcd_style=style, neck_k=kn))
        try:
            q0 = equilibrium(b, 0.0)
        except RuntimeError:
            q0 = np.array([-0.20, 0.35, 0.0, 0.0])
        q0 = q0 + np.array([0.0, 0.0, 0.15, 0.0])
        zf = np.zeros((1, b.strip_xi.size, 2 * nt + 1))
        res = simulate(b, (zf, zf, zf, zf), dt, 0.0, q0, ramp=0.0)
        rolls.append(res["q"][0, :, 2])
        heads.append(res["q"][0, :, 3])
        clears.append(res["clear"][0])
        print(style, vol, kn, "final roll %.3f rad, clearance %.3f m"
              % (rolls[-1][-1], clears[-1][-1]))
    out["release_t"] = t
    out["release_roll"] = np.array(rolls)
    out["release_head"] = np.array(heads)
    out["release_clear"] = np.array(clears)
    out["release_cases"] = np.array(["%s %g L, k=%g" % c for c in cases])
    out["volumes"] = VOLUMES
    save_cache("statics", **out)
    print("natural periods (s):", np.round(2 * np.pi / wn, 3))
    print("damping ratios:", np.round(zeta, 3))


if __name__ == "__main__":
    main()
