"""Rescue-breath timing, clustering and extremes of awash events (M1).

Two scenarios per environment, 32 tows of 300 s each:

* ``tow``  towing at 0.5 m/s into the waves (heading 180 deg);
* ``hold`` stopped (0 m/s) with the head toward the waves.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.body import build_body  # noqa: E402
from rescue_tow_airway.breathing import STRATEGIES, evaluate  # noqa: E402
from rescue_tow_airway.diagnostics import (events, fano, gev_fit,  # noqa
                                           timing_information)
from rescue_tow_airway.environments import (e1_swell, e2_chop,  # noqa: E402
                                            e3_mixed, e4_current,
                                            e7_confused)
from rescue_tow_airway.io import save_cache  # noqa: E402
from rescue_tow_airway.montecarlo import run_m1  # noqa: E402

SCENARIOS = (("tow", 0.5, 180.0), ("hold", 0.0, 180.0))
WINDOWS = np.array([10.0, 20.0, 30.0, 60.0])


def main():
    body = build_body()
    seas = [e1_swell(), e2_chop(), e3_mixed(), e7_confused(),
            e4_current(-1)]
    for k, sea in enumerate(seas):
        for name, u, h in SCENARIOS:
            tag = sea.name.replace("-", "m") + "_" + name
            res = run_m1(body, sea, u, np.radians(h), n_real=32,
                         duration=300.0, seed=500 + 10 * k + len(name))
            dt, first = res["dt"], res["first"]
            ev = events(res["clear"], dt, first)
            out = dict(p_awash=ev["p_awash"], rate=ev["rate"],
                       wet=ev["wet"], dry=ev["dry"], longest=ev["longest"],
                       fano=np.array([fano(res["clear"], dt, w, first)
                                      for w in WINDOWS]),
                       windows=WINDOWS,
                       info=timing_information(ev["dry"]))
            if ev["longest"].max() > 0 and np.sum(ev["longest"] > 0) > 8:
                out["gev"] = np.array(gev_fit(ev["longest"]))
            else:
                out["gev"] = np.full(3, np.nan)
            for s in STRATEGIES:
                r = evaluate(res["clear"], res["water"], dt, s, first=first,
                             rng=np.random.default_rng(3))
                out["fail_" + s] = r["fail"]
                out["interval_" + s] = r["interval"]
                out["wait_" + s] = r["wait"]
                out["nbreath_" + s] = r["n"]
            n0 = first
            n1 = first + int(round(90.0 / dt))
            out["sample_t"] = res["t"][n0:n1] - res["t"][n0]
            out["sample_clear"] = res["clear"][0, n0:n1]
            out["sample_water"] = res["water"][0, n0:n1]
            save_cache("breath_" + tag, **out)
            print("%-4s %-4s P %.3f rate %.3f/s Fano30 %.2f info %.2f bits "
                  "fail fixed %.3f clear %.3f crest %.3f oracle %.3f"
                  % (sea.name, name, out["p_awash"], out["rate"],
                     out["fano"][2], out["info"], out["fail_fixed"],
                     out["fail_clear"], out["fail_crest"],
                     out["fail_oracle"]), flush=True)


if __name__ == "__main__":
    main()
