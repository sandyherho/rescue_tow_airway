"""BCD volume, tow point and head support in the wind chop (E2).

M0 sweeps the BCD volume for both tow points, with the head unsupported
(neck stiffness 2 N m/rad) and cradled by the rescuer's hand (40 N m/rad).
M1 checks a subset at 0.5 m/s into the waves and across them.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.body import Rescue, build_body  # noqa: E402
from rescue_tow_airway.breathing import evaluate  # noqa: E402
from rescue_tow_airway.diagnostics import events  # noqa: E402
from rescue_tow_airway.environments import e2_chop  # noqa: E402
from rescue_tow_airway.io import save_cache  # noqa: E402
from rescue_tow_airway.linear import LinearModel  # noqa: E402
from rescue_tow_airway.montecarlo import run_m1  # noqa: E402

VOL_M0 = np.arange(8.0, 24.01, 1.0)
VOL_M1 = np.array([8.0, 10.0, 12.0, 16.0, 20.0])
HEADINGS = np.array([0.0, 90.0, 180.0])
SPEED = 0.5


def main():
    sea = e2_chop()
    grid = sea.spectral_grid()
    out = dict(vol_m0=VOL_M0, vol_m1=VOL_M1, headings=HEADINGS)
    for tow in ("collar", "valve"):
        for kn in (2.0, 40.0):
            arr = np.full((VOL_M0.size, HEADINGS.size, 4), np.nan)
            for i, v in enumerate(VOL_M0):
                body = build_body(Rescue(bcd_liters=v, tow=tow, neck_k=kn))
                try:
                    lm = LinearModel(body, SPEED)
                except RuntimeError:
                    continue
                for j, h in enumerate(HEADINGS):
                    st = lm.response(grid, np.radians(h))
                    arr[i, j] = [st.p_awash, st.p_breath, st.mean, st.sigma]
            out["m0_%s_k%02d" % (tow, kn)] = arr
            print(tow, kn, "P_breath at 180 deg:",
                  np.round(arr[:, 2, 1], 3), flush=True)
    for kn in (2.0, 40.0):
        arr = np.full((VOL_M1.size, 2, 3), np.nan)
        for i, v in enumerate(VOL_M1):
            body = build_body(Rescue(bcd_liters=v, neck_k=kn))
            for j, h in enumerate((90.0, 180.0)):
                res = run_m1(body, sea, SPEED, np.radians(h), n_real=16,
                             duration=240.0, seed=900 + 10 * i + j)
                ev = events(res["clear"], res["dt"], res["first"])
                br = evaluate(res["clear"], res["water"], res["dt"],
                              "fixed", first=res["first"],
                              rng=np.random.default_rng(1))
                arr[i, j] = [ev["p_awash"], br["fail"],
                             res["clear"][:, res["first"]:].mean()]
                print("M1 V=%g k=%g head %g: P %.4f fail %.4f"
                      % (v, kn, h, arr[i, j, 0], arr[i, j, 1]), flush=True)
        out["m1_k%02d" % kn] = arr
    save_cache("buoyancy", **out)


if __name__ == "__main__":
    main()
