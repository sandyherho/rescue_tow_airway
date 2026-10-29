"""Closed-form (M0) airway risk over tow speed and heading.

For every open-water environment, the linear model is built at each tow
speed and the Rice statistics are evaluated on a fine heading grid.
Heading is the angle between the tow direction and the mean direction
of wave travel: 0 deg tows with the waves, 180 deg tows into them.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.body import build_body  # noqa: E402
from rescue_tow_airway.environments import suite  # noqa: E402
from rescue_tow_airway.io import save_cache  # noqa: E402
from rescue_tow_airway.linear import LinearModel  # noqa: E402

SPEEDS = np.round(np.arange(0.0, 1.0001, 0.1), 2)
HEADINGS = np.arange(0.0, 360.0, 7.5)
FIELDS = ("p_awash", "rate", "duration", "p_breath", "sigma", "sigma_dot",
          "sig_roll", "sig_head", "sig_heave", "sig_pitch", "tz_enc",
          "mean")


def main():
    body = build_body()
    models = {u: LinearModel(body, u) for u in SPEEDS}
    for sea in suite():
        grid = sea.spectral_grid()
        out = {f: np.empty((SPEEDS.size, HEADINGS.size)) for f in FIELDS}
        for i, u in enumerate(SPEEDS):
            for j, h in enumerate(HEADINGS):
                st = models[u].response(grid, np.radians(h),
                                        current=sea.current)
                for f in FIELDS:
                    out[f][i, j] = getattr(st, f)
        tag = sea.name.replace("-", "m").replace("+", "p")
        save_cache("m0_map_" + tag, speeds=SPEEDS, headings=HEADINGS, **out)
        k = np.unravel_index(np.argmin(out["p_breath"]), out["p_breath"].shape)
        print("%-4s best heading %5.1f deg at %.1f m/s, P_breath %.4f; "
              "worst %.4f" % (sea.name, HEADINGS[k[1]], SPEEDS[k[0]],
                              out["p_breath"][k], out["p_breath"].max()))


if __name__ == "__main__":
    main()
