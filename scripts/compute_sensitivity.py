"""Global (Sobol) sensitivity of the airway risk to uncertain inputs.

The hydrodynamic coefficients of a floating human are poorly known, so
the conclusions are tested across wide ranges. M0 is evaluated in the
wind chop (E2) at 0.5 m/s, and three outputs are analyzed: the breath
failure probability towing into the waves (180 deg), across them
(90 deg), and the log ratio of the two, which carries the heading
recommendation. Inputs are sampled uniformly (the neck stiffness
uniformly in its logarithm).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.body import Hydro, Rescue, build_body  # noqa: E402
from rescue_tow_airway.diagnostics import sobol_jansen  # noqa: E402
from rescue_tow_airway.environments import e2_chop  # noqa: E402
from rescue_tow_airway.io import save_cache  # noqa: E402
from rescue_tow_airway.linear import LinearModel  # noqa: E402

NAMES = ["c_a", "beta_r", "C_D", "beta_44", "c_44", "rho_body", "k_neck"]
BOUNDS = [(0.5, 1.5), (0.05, 0.40), (0.5, 2.0), (0.02, 0.20), (0.1, 1.0),
          (995.0, 1015.0), (0.5, 20.0)]
LOG = [False] * 6 + [True]
N_BASE = 256


def model(x):
    sea = e2_chop()
    grid = sea.spectral_grid(n_omega=100, n_theta=36)
    out = np.empty((x.shape[0], 3))
    for i, row in enumerate(x):
        hy = Hydro(ca=row[0], beta_r=row[1], cd=row[2], beta44=row[3],
                   c44=row[4])
        body = build_body(Rescue(body_density=row[5], neck_k=row[6]), hy)
        try:
            lm = LinearModel(body, 0.5)
            p180 = lm.response(grid, np.pi).p_breath
            p90 = lm.response(grid, 0.5 * np.pi).p_breath
        except RuntimeError:
            p180 = p90 = np.nan
        out[i] = [p180, p90, np.log10(max(p180, 1e-6) / max(p90, 1e-6))]
    return out


def main():
    s1, st = sobol_jansen(model, BOUNDS, n=N_BASE, seed=11, log=LOG)
    rng = np.random.default_rng(12)
    u = rng.random((400, len(BOUNDS)))
    lo = np.array([np.log(b[0]) if g else b[0] for b, g in zip(BOUNDS, LOG)])
    hi = np.array([np.log(b[1]) if g else b[1] for b, g in zip(BOUNDS, LOG)])
    x = lo + u * (hi - lo)
    x[:, -1] = np.exp(x[:, -1])
    y = model(x)
    save_cache("sensitivity", names=np.array(NAMES), s1=s1, st=st,
               sample_x=x, sample_y=y)
    for j, n in enumerate(NAMES):
        print("%-9s S1 %s  ST %s" % (n, np.round(s1[j], 3),
                                     np.round(st[j], 3)))
    print("fraction of samples where 90 deg beats 180 deg: %.3f"
          % np.mean(y[:, 2] > 0))


if __name__ == "__main__":
    main()
