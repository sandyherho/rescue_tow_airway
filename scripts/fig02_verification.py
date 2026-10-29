"""Figure 2: M1 against its own linearization M0, and convergence."""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.io import load_cache, save_figure, write_csv  # noqa
from rescue_tow_airway.plotting import (C, legend_below, line,  # noqa: E402
                                        panel_label, style)

HCOL = {0: C["blue"], 1: C["green"], 2: C["red"]}
HLAB = {0: "with waves (0°)", 1: "across (90°)",
        2: "into waves (180°)"}


def main():
    style()
    v = load_cache("verification")
    hs, m0, m1 = v["hs"], v["m0"], v["m1"]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    ax = axs[0, 0]
    for j in range(3):
        ax.plot(hs, 100 * m0[:, j, 1], "-", color=HCOL[j], label=HLAB[j])
        ax.plot(hs, 100 * m1[:, j, 1], "o", color=HCOL[j], ms=3.5)
    ax.set_xlabel("significant wave height Hs (m)")
    ax.set_ylabel("std of airway clearance (cm)")
    panel_label(ax, "a")

    ax = axs[0, 1]
    for j in range(3):
        sel = (m1[:, j, 3] > 0) | (m0[:, j, 3] > 1e-5)
        ax.semilogy(hs[sel], np.maximum(m0[sel, j, 3], 1e-5), "-",
                    color=HCOL[j])
        ax.semilogy(hs[sel], np.maximum(m1[sel, j, 3], 1e-5), "o",
                    color=HCOL[j], ms=3.5)
    ax.set_ylim(1e-4, 0.5)
    ax.set_xlabel("significant wave height Hs (m)")
    ax.set_ylabel("fraction of time awash")
    panel_label(ax, "b")

    ax = axs[1, 0]
    for j, h in enumerate((0, 90, 180)):
        we = v["tf_we_%03d" % h]
        ax.plot(np.abs(we), v["tf_r_%03d" % h], color=HCOL[j])
    ax.set_xlabel("encounter frequency |ωe| (rad/s)")
    ax.set_ylabel("|r| per unit wave amplitude")
    ax.set_xlim(0, 5)
    panel_label(ax, "c")

    ax = axs[1, 1]
    dts = v["dts"]
    err = v["conv_err"]
    ax.loglog(dts[:-1], 100 * err[:-1], "o-", color=C["purple"],
              label="M1 against dt = 0.02 s")
    ref = 100 * err[0] * (dts[:-1] / dts[0]) ** 4
    ax.loglog(dts[:-1], ref, "k:", lw=0.8, label="slope 4")
    ax.set_xticks(dts[:-1])
    ax.set_xticklabels(["%.2f" % d for d in dts[:-1]])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("time step (s)")
    ax.set_ylabel("rms clearance difference (cm)")
    panel_label(ax, "d")
    h_d, l_d = ax.get_legend_handles_labels()
    legend_below(fig, [
        ("(a)\u2013(c)", [line(HCOL[j]) for j in range(3)]
         + [line("k"), line("k", "none", "o")],
         [HLAB[j] for j in range(3)] + ["M0 (linear)", "M1 (nonlinear)"]),
        ("(d)", h_d, l_d)])
    save_figure(fig, "fig02_verification")

    rows = []
    for i, h_ in enumerate(hs):
        for j, hd in enumerate(v["headings"]):
            rows.append([h_, hd] + list(m0[i, j]) + list(m1[i, j]))
    names = ["mean", "sigma", "sigma_dot", "p_awash", "rate", "s_heave",
             "s_pitch", "s_roll", "s_head"]
    write_csv("fig02_ladder", ["hs", "heading"] + ["m0_" + n for n in names]
              + ["m1_" + n for n in names], rows)


if __name__ == "__main__":
    main()
