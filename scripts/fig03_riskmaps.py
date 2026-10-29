"""Figure 3: breath failure risk over tow speed and heading.

Polar maps of the probability that a rescue breath at a random moment
meets water (M0, closed form), with the M1 Monte Carlo estimates as
colored markers on the same scale. The mean wave direction points up:
a tow toward the top travels with the waves, toward the bottom into
them. Radius is tow speed.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.io import (has_cache, load_cache,  # noqa: E402
                                  save_figure, write_csv)
from rescue_tow_airway.plotting import panel_label, style  # noqa: E402

ENVS = [("E1", "E1 open-ocean swell"), ("E2", "E2 wind chop"),
        ("E3", "E3 swell crossed by wind sea"),
        ("E7", "E7 confused sea"), ("E4m", "E4 opposing current"),
        ("E4p", "E4 following current")]
NORM = LogNorm(1e-3, 1.0)
CMAP = "magma_r"


def floor(p):
    return np.clip(p, 1.01e-3, 1.0)


def main():
    style()
    fig = plt.figure(figsize=(7.0, 5.2))
    rows = []
    for k, (tag, _) in enumerate(ENVS):
        ax = fig.add_subplot(2, 3, k + 1, projection="polar")
        m = load_cache("m0_map_" + tag)
        th = np.radians(np.r_[m["headings"], 360.0])
        p = m["p_breath"]
        p = np.c_[p, p[:, :1]]
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        pcm = ax.pcolormesh(th, m["speeds"], floor(p), norm=NORM, cmap=CMAP,
                            shading="gouraud", rasterized=True)
        if has_cache("m1_map_" + tag):
            m1 = load_cache("m1_map_" + tag)
            hh = m1["headings"]
            uu = m1["speeds"]
            pts_th, pts_r, pts_c = [], [], []
            for i, u in enumerate(uu):
                for j, h in enumerate(hh):
                    for hm in ({h, (360 - h) % 360} if tag != "E3" else {h}):
                        pts_th.append(np.radians(hm))
                        pts_r.append(u)
                        pts_c.append(m1["p_breath"][i, j])
                    rows.append([tag, u, h, m1["p_breath"][i, j],
                                 m1["p_awash"][i, j]])
            ax.scatter(pts_th, pts_r, c=floor(np.array(pts_c)), norm=NORM,
                       cmap=CMAP, s=16, edgecolors="white", lw=0.5,
                       zorder=5)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["0.5", "1 m/s"], fontsize=5.5)
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["with", "across", "into", ""], fontsize=6)
        ax.tick_params(pad=0)
        ax.grid(lw=0.3, alpha=0.5)
        panel_label(ax, "abcdef"[k], x=-0.12, y=1.06)
    cax = fig.add_axes([0.25, 0.04, 0.5, 0.018])
    cb = fig.colorbar(pcm, cax=cax, orientation="horizontal")
    cb.set_label("probability that a rescue breath meets water "
                 "(shading M0, dots M1)")
    fig.subplots_adjust(hspace=0.45, wspace=0.35, bottom=0.14, top=0.92)
    save_figure(fig, "fig03_riskmaps")
    write_csv("fig03_m1_points", ["env", "speed", "heading", "p_breath",
                                  "p_awash"], rows)


if __name__ == "__main__":
    main()
