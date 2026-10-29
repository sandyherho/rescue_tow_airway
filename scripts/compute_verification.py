"""Verification of M1 against its own linearization M0.

1. Amplitude ladder: the E2 spectrum scaled to Hs = 0.05 to 0.8 m,
   long-crested along each of three tow headings, at 0.5 m/s. At small
   amplitude M1 must reproduce M0; the departure at larger amplitude
   measures the nonlinearity of the airway exposure.
2. Time-step convergence of M1 at the design sea state.
3. The M0 airway transfer function against encounter frequency.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.body import build_body  # noqa: E402
from rescue_tow_airway.diagnostics import events  # noqa: E402
from rescue_tow_airway.io import save_cache  # noqa: E402
from rescue_tow_airway.linear import LinearModel  # noqa: E402
from rescue_tow_airway.montecarlo import run_m1  # noqa: E402
from rescue_tow_airway.waves import SeaState, WaveSystem  # noqa: E402

HS = np.array([0.05, 0.15, 0.3, 0.45, 0.6, 0.8])
HEADINGS = np.array([0.0, 90.0, 180.0])
SPEED = 0.5


def m1_stats(res):
    c = res["clear"][:, res["first"]:]
    ev = events(res["clear"], res["dt"], res["first"])
    q = res["q"][:, res["first"]:]
    sd = np.diff(c, axis=1).std() / res["dt"]
    return [c.mean(), c.std(), sd, ev["p_awash"], ev["rate"],
            q[..., 0].std(), q[..., 1].std(), q[..., 2].std(),
            q[..., 3].std()]


def main():
    body = build_body()
    lm = LinearModel(body, SPEED)
    m0 = np.empty((HS.size, HEADINGS.size, 9))
    m1 = np.empty((HS.size, HEADINGS.size, 9))
    for i, hs in enumerate(HS):
        for j, h in enumerate(HEADINGS):
            sea = SeaState([WaveSystem(hs, 3.5, 3.3, 0.0, 10.0)])
            st = lm.response(sea.spectral_grid(), np.radians(h))
            m0[i, j] = [st.mean, st.sigma, st.sigma_dot, st.p_awash,
                        st.rate, st.sig_heave, st.sig_pitch, st.sig_roll,
                        st.sig_head]
            res = run_m1(body, sea, SPEED, np.radians(h), n_real=16,
                         duration=200.0, seed=7 + i, keep_states=True)
            m1[i, j] = m1_stats(res)
            print("Hs %.2f head %3.0f  sigma_r M0 %.4f M1 %.4f"
                  "  P M0 %.4f M1 %.4f" % (hs, h, m0[i, j, 1], m1[i, j, 1],
                                           m0[i, j, 3], m1[i, j, 3]),
                  flush=True)

    # Time-step convergence (same realizations).
    sea = SeaState([WaveSystem(0.6, 3.5, 3.3, 0.0, 10.0)])
    dts = np.array([0.08, 0.04, 0.02])
    conv = []
    ref_series = []
    for dt in dts:
        res = run_m1(body, sea, SPEED, np.pi, n_real=4, duration=120.0,
                     dt=dt, seed=99, keep_states=False)
        step = int(round(0.08 / dt))
        series = res["clear"][:, ::step]
        ref_series.append(series)
        ev = events(res["clear"], res["dt"], res["first"])
        conv.append([res["clear"][:, res["first"]:].std(), ev["p_awash"]])
        print("dt %.2f  sigma %.5f  P %.4f" % (dt, conv[-1][0], conv[-1][1]))
    ref = ref_series[-1]
    err = [np.sqrt(np.mean((s - ref) ** 2)) for s in ref_series]

    # Transfer function of the airway clearance (unit amplitude waves).
    tw = np.linspace(0.6, 4.5, 300)
    grid = dict(a2=np.full(tw.size, 1e-6), omega=tw, sigma=tw[:, None],
                coef=np.ones((tw.size, 1), complex))
    trans = {}
    for h in (0.0, 90.0, 180.0):
        k = tw ** 2 / 9.81
        grid["kx"] = k[:, None]
        grid["ky"] = np.zeros_like(k)[:, None]
        _, tr = lm.response(grid, np.radians(h), return_transfer=True)
        trans["tf_we_%03d" % h] = tr["we"]
        trans["tf_r_%03d" % h] = np.abs(tr["rh"])
        trans["tf_heave_%03d" % h] = np.abs(tr["qh"][:, 0])
        trans["tf_pitch_%03d" % h] = np.abs(tr["qh"][:, 1])
        trans["tf_roll_%03d" % h] = np.abs(tr["qh"][:, 2])
    save_cache("verification", hs=HS, headings=HEADINGS, m0=m0, m1=m1,
               dts=dts, conv=np.array(conv), conv_err=np.array(err),
               tf_omega=tw, **trans)


if __name__ == "__main__":
    main()
