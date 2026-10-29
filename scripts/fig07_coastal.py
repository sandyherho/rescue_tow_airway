"""Figure 7: beach exit (E5) and waiting beside a boat hull (E6)."""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.body import build_body  # noqa: E402
from rescue_tow_airway.io import load_cache, save_figure, write_csv  # noqa
from rescue_tow_airway.plotting import (C, legend_below, line,  # noqa: E402
                                        panel_label, style)


def main():
    style()
    e = load_cache("environments")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    fig.subplots_adjust(wspace=0.62)

    ax = axs[0]
    d0, m0 = e["e5_depth_m0"], e["e5_m0"]
    ax.plot(d0, 100 * m0[:, 4], color=C["grey"], label="M0")
    ax.plot(e["e5_depth_m1"], 100 * e["e5_m1"][:, 3], "o",
            color=C["grey"], ms=3.5, label="M1")
    ax.invert_xaxis()
    ax.set_ylim(0, None)
    ax.set_xlabel("local depth (m)")
    ax.set_ylabel("std of airway clearance (cm)")
    pmax = max(m0[:, 3].max(), e["e5_m1"][:, 2].max())
    ax.text(0.04, 0.12, "P(breath meets water) < %.0e\nat every depth"
            % max(pmax, 1e-4), transform=ax.transAxes, fontsize=6)
    for dd, hh in zip(d0[::3], m0[::3, 0]):
        ax.annotate("Hs %.2f" % hh, (dd, 100 * np.interp(dd, d0[::-1],
                                                         m0[::-1, 4])),
                    (-4, 5), textcoords="offset points", fontsize=5.5,
                    ha="center", color=C["blue"])
    panel_label(ax, "a", x=-0.28)

    ax = axs[1]
    dist = e["e6_dist_m0"]
    ratio = e["e6_var_eta"] / e["e6_var_open"]
    ax.plot(dist, ratio, color=C["blue"])
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set_xlabel("distance from hull (m)")
    ax.set_ylabel("surface variance / open water")
    panel_label(ax, "b", x=-0.28)

    ax = axs[2]
    # With the head toward the hull the airway lies ahead of the center of
    # gravity; keep only distances that leave it at least 0.15 m off the
    # hull.
    xa = build_body().airway[0]
    for name, col in (("normal", C["red"]), ("parallel", C["purple"])):
        ok0 = dist - (xa if name == "normal" else 0.0) >= 0.15
        d1 = e["e6_dist_m1"]
        ok1 = d1 - (xa if name == "normal" else 0.0) >= 0.15
        ax.semilogy(dist[ok0], np.maximum(e["e6_m0_" + name][ok0, 2], 1e-5),
                    color=col, label="%s" % (
                        "head toward hull" if name == "normal" else
                        "alongside hull"))
        ax.semilogy(d1[ok1], np.maximum(e["e6_m1_" + name][ok1, 2], 1e-5),
                    "o", color=col, ms=3.5)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("distance of the CG from the hull (m)")
    ax.set_ylabel("P(breath meets water)")
    pal = max(e["e6_m0_parallel"][:, 2].max(), e["e6_m1_parallel"][:, 2].max())
    ax.text(0.97, 0.06, "alongside: below %.0e\nat every distance"
            % max(pal, 1e-4), transform=ax.transAxes, ha="right",
            fontsize=6, color=C["purple"])
    panel_label(ax, "c", x=-0.28)
    legend_below(fig, [
        ("(a), (c)", [line("k"), line("k", "none", "o")], ["M0", "M1"]),
        ("(c)", [line(C["red"]), line(C["purple"])],
         ["head toward the hull", "alongside the hull"])])
    save_figure(fig, "fig07_coastal")
    write_csv("fig07_e5_m0", ["depth", "hs", "p_awash", "rate", "p_breath",
                              "sigma"], np.c_[d0, m0])
    write_csv("fig07_e6_m0", ["distance", "var_ratio", "p_breath_normal",
                              "p_breath_parallel"],
              np.c_[dist, ratio, e["e6_m0_normal"][:, 2],
                    e["e6_m0_parallel"][:, 2]])


if __name__ == "__main__":
    main()
