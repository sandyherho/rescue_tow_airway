"""Monte Carlo (M1) airway risk on a coarse speed-heading grid.

M1 checks the closed-form maps at three speeds and three headings (four
for the asymmetric mixed sea). Where M0 gives negligible risk at every
speed and heading (E1, E4+), one tow condition is simulated.

Usage: ``python compute_riskmaps_m1.py [E1 E2 ...]`` (default: all).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.body import build_body  # noqa: E402
from rescue_tow_airway.breathing import evaluate  # noqa: E402
from rescue_tow_airway.diagnostics import events  # noqa: E402
from rescue_tow_airway.environments import suite  # noqa: E402
from rescue_tow_airway.io import has_cache, save_cache  # noqa: E402
from rescue_tow_airway.montecarlo import run_m1  # noqa: E402

SPEEDS = np.array([0.0, 0.5, 1.0])
SYM_HEADINGS = np.array([0.0, 90.0, 180.0])
FULL_HEADINGS = np.array([0.0, 90.0, 180.0, 270.0])
SPOT = {"E1", "E4+"}       # M0 risk is negligible everywhere: one check
N_REAL = 12
DURATION = 200.0


def summarize(res):
    """Airway statistics of one Monte Carlo set."""
    ev = events(res["clear"], res["dt"], res["first"])
    br = evaluate(res["clear"], res["water"], res["dt"], "fixed",
                  first=res["first"], rng=np.random.default_rng(1))
    c = res["clear"][:, res["first"]:]
    q = res["q"][:, res["first"]:]
    face = np.abs(q[..., 2] + q[..., 3]).max(1)
    return dict(p_awash=ev["p_awash"], rate=ev["rate"],
                duration=ev["wet"].mean() if ev["wet"].size else 0.0,
                p_breath=br["fail"], sigma=c.std(), mean=c.mean(),
                roll_max=face.max(), roll_std=q[..., 2].std())


def main(tags):
    body = build_body()
    for sea in suite():
        if tags and sea.name not in tags:
            continue
        tag = sea.name.replace("-", "m").replace("+", "p")
        if has_cache("m1_map_" + tag):
            continue
        heads = FULL_HEADINGS if sea.name == "E3" else SYM_HEADINGS
        speeds = np.array([0.5]) if sea.name in SPOT else SPEEDS
        if sea.name in SPOT:
            heads = np.array([180.0])
        keys = ("p_awash", "rate", "duration", "p_breath", "sigma", "mean",
                "roll_max", "roll_std")
        out = {k: np.empty((speeds.size, heads.size)) for k in keys}
        for i, u in enumerate(speeds):
            for j, h in enumerate(heads):
                res = run_m1(body, sea, u, np.radians(h), n_real=N_REAL,
                             duration=DURATION, seed=1000 + 10 * i + j,
                             keep_states=True, batch=16)
                s = summarize(res)
                for k in keys:
                    out[k][i, j] = s[k]
                print(sea.name, u, h, "P %.4f  P_breath %.4f  roll %.2f"
                      % (s["p_awash"], s["p_breath"], s["roll_max"]),
                      flush=True)
        save_cache("m1_map_" + tag, speeds=speeds, headings=heads, **out)


if __name__ == "__main__":
    main(sys.argv[1:])
