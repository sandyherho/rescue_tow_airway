"""Figure 5: BCD air, tow point and head support in the wind chop."""

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


def main():
    style()
    b = load_cache("buoyancy")
    v0, v1 = b["vol_m0"], b["vol_m1"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    fig.subplots_adjust(wspace=0.42)
    rows = []
    for k, (hd, j) in enumerate(((180.0, 2), (90.0, 1))):
        ax = axs[k]
        for tow, col in (("collar", C["blue"]), ("valve", C["orange"])):
            for kn, ls in ((2, "-"), (40, "--")):
                arr = b["m0_%s_k%02d" % (tow, kn)]
                ax.semilogy(v0, np.maximum(arr[:, j, 1], 1e-5), ls,
                            color=col, label="%s, %s head" %
                            (tow, "loose" if kn == 2 else "held"))
                for vv, row in zip(v0, arr[:, j]):
                    rows.append([hd, tow, kn, vv] + list(row))
        jj = 1 if hd == 180.0 else 0
        for kn, mk in ((2, "o"), (40, "s")):
            arr = b["m1_k%02d" % kn]
            ax.semilogy(v1, np.maximum(arr[:, jj, 1], 1e-5), mk,
                        color=C["blue"], ms=3.5,
                        mfc="white" if kn == 40 else C["blue"])
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("BCD air (L)")
        ax.set_ylabel("P(breath meets water)")
        panel_label(ax, "ab"[k], x=-0.25)
    ax = axs[2]
    for tow, col in (("collar", C["blue"]), ("valve", C["orange"])):
        arr = b["m0_%s_k02" % tow]
        ax.plot(v0, 100 * arr[:, 2, 2], color=col, label=tow)
        ax.plot(v0, 100 * arr[:, 2, 3], ":", color=col)
    ax.set_xlabel("BCD air (L)")
    ax.set_ylabel("airway clearance (cm)")
    panel_label(ax, "c", x=-0.25)
    legend_below(fig, [
        ("(a)\u2013(c)", [line(C["blue"]), line(C["orange"])],
         ["collar tow", "valve tow"]),
        ("(a), (b)", [line("k", "-"), line("k", "--"),
                      line(C["blue"], "none", "o"),
                      line(C["blue"], "none", "s", "white")],
         ["M0, head loose", "M0, head held", "M1, head loose",
          "M1, head held"]),
        ("(c)", [line("k", "-"), line("k", ":")],
         ["mean (M0)", "standard deviation (M0)"])])
    save_figure(fig, "fig05_buoyancy")
    write_csv("fig05_buoyancy_m0", ["heading", "tow", "neck_k", "bcd_L",
                                    "p_awash", "p_breath", "mean", "sigma"],
              rows)


if __name__ == "__main__":
    main()
