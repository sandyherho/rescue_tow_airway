"""Figure 4: heading dependence at 0.5 m/s and the encounter period.

(a) Breath failure probability against tow heading at 0.5 m/s, M0
curves and M1 markers, for the open-water environments.
(b) Mean encounter period of the airway clearance (M0) against heading.
(c) Breath failure probability against tow speed for the best and worst
headings in the wind chop.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.io import (has_cache, load_cache,  # noqa: E402
                                  save_figure, write_csv)
from rescue_tow_airway.plotting import (ENV_COLORS, legend_below,  # noqa
                                        line, panel_label, style)

ENVS = [("E2", "E2"), ("E7", "E7"), ("E3", "E3"), ("E4m", "E4-"),
        ("E1", "E1")]


def main():
    style()
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    fig.subplots_adjust(wspace=0.42)
    rows = []
    for tag, name in ENVS:
        m = load_cache("m0_map_" + tag)
        i = int(np.argmin(np.abs(m["speeds"] - 0.5)))
        h = m["headings"]
        sel = h <= 180.0 if tag != "E3" else np.ones_like(h, bool)
        col = ENV_COLORS[name]
        axs[0].semilogy(h[sel], np.maximum(m["p_breath"][i, sel], 1e-4),
                        color=col, label=name)
        axs[1].plot(h[sel], m["tz_enc"][i, sel], color=col, label=name)
        if has_cache("m1_map_" + tag):
            m1 = load_cache("m1_map_" + tag)
            k = int(np.argmin(np.abs(m1["speeds"] - 0.5)))
            hh = m1["headings"]
            s1 = hh <= 180.0 if tag != "E3" else np.ones_like(hh, bool)
            axs[0].semilogy(hh[s1], np.maximum(m1["p_breath"][k, s1], 1e-4),
                            "o", color=col, ms=3.5)
        for hj, pj, tj in zip(h[sel], m["p_breath"][i, sel],
                              m["tz_enc"][i, sel]):
            rows.append([name, hj, pj, tj])
    axs[0].set_ylim(1e-4, 1)
    axs[0].set_xlim(0, 180)
    axs[0].set_xticks([0, 45, 90, 135, 180])
    axs[0].set_xlabel("heading (deg; 0 with waves, 180 into)")
    axs[0].set_ylabel("P(breath meets water)")
    panel_label(axs[0], "a", x=-0.25)
    axs[1].set_xlim(0, 180)
    axs[1].set_xticks([0, 45, 90, 135, 180])
    axs[1].set_xlabel("heading (deg)")
    axs[1].set_ylabel("mean period of airway clearance (s)")
    panel_label(axs[1], "b", x=-0.22)
    m = load_cache("m0_map_E2")
    for hd, ls in ((180.0, "-"), (0.0, "--"), (90.0, ":")):
        j = int(np.argmin(np.abs(m["headings"] - hd)))
        axs[2].semilogy(m["speeds"], np.maximum(m["p_breath"][:, j], 1e-4),
                        ls, color=ENV_COLORS["E2"], label="%g°" % hd)
    if has_cache("m1_map_E2"):
        m1 = load_cache("m1_map_E2")
        for hd, mk in ((180.0, "o"), (0.0, "s"), (90.0, "^")):
            j = int(np.argmin(np.abs(m1["headings"] - hd)))
            axs[2].semilogy(m1["speeds"], np.maximum(m1["p_breath"][:, j],
                                                     1e-4), mk,
                            color=ENV_COLORS["E2"], ms=3.5, mfc="white")
    axs[2].set_ylim(1e-4, 1)
    axs[2].set_xlabel("tow speed (m/s)")
    axs[2].set_ylabel("P(breath meets water)")
    panel_label(axs[2], "c", x=-0.25)
    c2 = ENV_COLORS["E2"]
    legend_below(fig, [
        ("(a), (b)", [line(ENV_COLORS[n]) for _, n in ENVS]
         + [line("k"), line("k", "none", "o")],
         [n for _, n in ENVS] + ["M0", "M1"]),
        ("(c) E2", [(line(c2, ls), line(c2, "none", mk, "white"))
                    for ls, mk in (("-", "o"), ("--", "s"), (":", "^"))],
         ["into the waves (180\u00b0)", "with the waves (0\u00b0)",
          "across (90\u00b0)"])])
    save_figure(fig, "fig04_heading")
    write_csv("fig04_heading_m0", ["env", "heading", "p_breath",
                                   "t_encounter"], rows)


if __name__ == "__main__":
    main()
