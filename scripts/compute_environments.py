"""Beach exit (E5) and boat hull (E6).

E5: a 0.8 m, 9 s deep-water swell shoaled to decreasing depth, towed
toward shore (with the waves) at 0.5 m/s.

E6: the wind chop reflected from a vertical hull, victim held still at
distance ``d`` from the hull, either with the head toward the hull
(``normal``) or lying alongside it (``parallel``). The surface variance
in front of a reflecting wall is closed form,

    var(eta)(d) = sum 1/2 a^2 (1 + R^2 + 2 R cos(2 k_x d)),

with nodes and antinodes that the directional spread smears out with
distance.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.body import build_body  # noqa: E402
from rescue_tow_airway.diagnostics import events  # noqa: E402
from rescue_tow_airway.environments import e5_beach, e6_hull  # noqa: E402
from rescue_tow_airway.io import save_cache  # noqa: E402
from rescue_tow_airway.linear import LinearModel  # noqa: E402
from rescue_tow_airway.montecarlo import run_m1  # noqa: E402
from rescue_tow_airway.breathing import evaluate  # noqa: E402

DEPTHS_M0 = np.array([30.0, 20.0, 15.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0,
                      3.5, 3.0, 2.5, 2.0])
DEPTHS_M1 = np.array([20.0, 8.0, 5.0, 3.5, 2.5])
DIST_M0 = np.arange(0.2, 8.001, 0.05)
DIST_M1 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])
ORIENT = (("normal", 0.0), ("parallel", 90.0))


def m1_summary(res):
    ev = events(res["clear"], res["dt"], res["first"])
    br = evaluate(res["clear"], res["water"], res["dt"], "fixed",
                  first=res["first"], rng=np.random.default_rng(1))
    return [ev["p_awash"], ev["rate"], br["fail"],
            res["clear"][:, res["first"]:].std()]


def main():
    body = build_body()
    out = {}
    # ---- E5 beach exit ----------------------------------------------
    lm = LinearModel(body, 0.5)
    rows = []
    for h in DEPTHS_M0:
        sea = e5_beach(h)
        st = lm.response(sea.spectral_grid(), 0.0)
        hs = 4.0 * np.sqrt(0.5 * sea.spectral_grid()["a2"].sum())
        rows.append([hs, st.p_awash, st.rate, st.p_breath, st.sigma])
    out["e5_depth_m0"] = DEPTHS_M0
    out["e5_m0"] = np.array(rows)
    rows = []
    for i, h in enumerate(DEPTHS_M1):
        res = run_m1(body, e5_beach(h), 0.5, 0.0, n_real=16, duration=240.0,
                     seed=700 + i)
        rows.append(m1_summary(res))
        print("E5 h=%.1f  M1 P %.4f fail %.4f" % (h, rows[-1][0],
                                                  rows[-1][2]), flush=True)
    out["e5_depth_m1"] = DEPTHS_M1
    out["e5_m1"] = np.array(rows)

    # ---- E6 boat hull -------------------------------------------------
    sea = e6_hull()
    grid = sea.spectral_grid()
    lm0 = LinearModel(body, 0.0)
    kx = grid["kx"][:, 0]
    refl = np.abs(grid["coef"][:, 1])
    var = np.array([np.sum(0.5 * grid["a2"]
                           * (1 + refl ** 2 + 2 * refl * np.cos(2 * kx * d)))
                    for d in DIST_M0])
    out["e6_var_eta"] = var
    out["e6_var_open"] = np.sum(0.5 * grid["a2"])
    out["e6_dist_m0"] = DIST_M0
    for name, h in ORIENT:
        rows = []
        for d in DIST_M0:
            st = lm0.response(grid, np.radians(h), origin=(-d, 0.0))
            rows.append([st.p_awash, st.rate, st.p_breath, st.sigma,
                         st.sig_roll])
        out["e6_m0_" + name] = np.array(rows)
        rows = []
        for i, d in enumerate(DIST_M1):
            res = run_m1(body, sea, 0.0, np.radians(h), n_real=16,
                         duration=240.0, seed=800 + i, origin=(-d, 0.0))
            rows.append(m1_summary(res))
            print("E6 %s d=%.1f  M1 P %.4f fail %.4f" % (name, d, rows[-1][0],
                                                         rows[-1][2]),
                  flush=True)
        out["e6_m1_" + name] = np.array(rows)
    out["e6_dist_m1"] = DIST_M1
    save_cache("environments", **out)


if __name__ == "__main__":
    main()
