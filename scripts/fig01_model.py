"""Figure 1: the model diver, its floating equilibrium and roll stability."""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, Polygon  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.body import Rescue, build_body  # noqa: E402
from rescue_tow_airway.diver_art import (airway_point, to_screen,  # noqa
                                         victim_shapes)
from rescue_tow_airway.dynamics import equilibrium, kinematics  # noqa
from rescue_tow_airway.io import load_cache, save_figure, write_csv  # noqa
from rescue_tow_airway.plotting import (C, legend_below, line,  # noqa: E402
                                        panel_label, style)


def main():
    style()
    st = load_cache("statics")
    body = build_body()
    q0 = equilibrium(body, 0.5)
    fig = plt.figure(figsize=(7.0, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.45,
                          wspace=0.75)

    # (a) side view: drawing plus model primitives at equilibrium.
    ax = fig.add_subplot(gs[0, :2])
    ax.axhspan(-1.5, 0, color="#dcecf3", zorder=0)
    ax.axhline(0, color=C["blue"], lw=0.8)
    for s in sorted(victim_shapes(body), key=lambda d: d["z"]):
        p = to_screen(s["pts"], q0)
        ax.add_patch(Polygon(p, closed=s["closed"], fc=s["fc"], ec=s["ec"],
                             lw=s["lw"], alpha=0.25 * s["alpha"], zorder=2))
    kin = kinematics(q0[None], body)
    xs = kin["x2"][0]
    zs = kin["zg"][0]
    cols = {0: "#444444", 1: C["grey"], 2: C["orange"]}
    for i in range(body.xi.size):
        if body.kind[i] == 2 and body.y0[i] < 0:
            continue
        ax.add_patch(Ellipse((xs[i], zs[i]), body.dxi[i],
                             2 * body.c[i], angle=np.degrees(q0[1]),
                             fc="none", ec=cols[body.kind[i]], lw=0.5,
                             zorder=3))
    a = to_screen(airway_point(body)[None], q0)[0]
    ax.plot(*a, "o", color=C["red"], ms=4, zorder=5)
    ax.annotate("airway", a, (a[0] - 0.55, a[1] + 0.2), fontsize=7,
                color=C["red"], arrowprops=dict(arrowstyle="-", lw=0.5,
                                                color=C["red"]))
    ax.plot(0, q0[0], "+", color="k", ms=6, zorder=5)
    ax.annotate("CG", (0, q0[0]), (0.15, q0[0] - 0.28), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.annotate("", (1.05, 0.25), (0.55, 0.25),
                arrowprops=dict(arrowstyle="-|>", lw=0.8))
    ax.text(0.8, 0.29, "tow", ha="center", fontsize=7)
    ax.text(-2.55, 0.05, "still water", fontsize=7, color=C["blue"])
    ax.set_xlim(-2.6, 1.15)
    ax.set_ylim(-1.0, 0.45)
    ax.set_aspect("equal")
    ax.set_xlabel("along tow (m)")
    ax.set_ylabel("height (m)")
    panel_label(ax, "a", x=-0.08)

    # (b) cross-sections: jacket versus wing.
    ax = fig.add_subplot(gs[0, 2])
    for k, (style_, dx) in enumerate((("jacket", -0.33), ("wing", 0.33))):
        b = build_body(Rescue(bcd_style=style_))
        i0 = np.argmin(np.abs(b.xi - (-0.25 - b.cg_shift[0])))
        x0 = b.xi[i0]
        sel = np.isclose(b.xi, x0)
        for i in np.flatnonzero(sel):
            fc = {0: "#d0d0d0", 1: "#9aa4ab", 2: C["orange"]}[b.kind[i]]
            ax.add_patch(Ellipse((dx + b.y0[i], b.z0[i]), 2 * b.b[i],
                                 2 * b.c[i], fc=fc, ec="k", lw=0.5))
        ax.text(dx, 0.25, style_, ha="center", fontsize=7.5)
    ax.annotate("chest up", (0.0, 0.17), ha="center", fontsize=6.5,
                color=C["grey"])
    ax.set_xlim(-0.66, 0.66)
    ax.set_ylim(-0.45, 0.32)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_label(ax, "b", x=-0.16)

    # (c) equilibrium pitch and clearance against BCD volume.
    ax = fig.add_subplot(gs[1, 0])
    v = st["volumes"]
    for tow, ls in (("collar", "-"), ("valve", "--")):
        ax.plot(v, np.degrees(st["pitch_%s_05" % tow]), ls, color=C["blue"],
                label="pitch, %s" % tow)
    ax.set_xlabel("BCD air (L)")
    ax.set_ylabel("trim, head up (deg)", color=C["blue"])
    ax2 = ax.twinx()
    for tow, ls in (("collar", "-"), ("valve", "--")):
        ax2.plot(v, 100 * st["clear_%s_05" % tow], ls, color=C["red"])
    ax2.set_ylabel("airway clearance (cm)", color=C["red"])
    ax2.spines["right"].set_visible(True)
    panel_label(ax, "c")

    # (d) natural modes.
    ax = fig.add_subplot(gs[1, 1])
    wn, zeta = st["eig_wn"], st["eig_zeta"]
    ax.scatter(2 * np.pi / wn, zeta, color=C["purple"], s=18)
    for k, (w, z) in enumerate(zip(wn, zeta)):
        off = (4, -9) if k % 2 == 0 else (4, 5)
        ax.annotate("%.2f s" % (2 * np.pi / w), (2 * np.pi / w, z),
                    off, textcoords="offset points", fontsize=6)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.12, 14)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("natural period (s)")
    ax.set_ylabel("damping ratio")
    panel_label(ax, "d")

    # (e) roll release in still water.
    ax = fig.add_subplot(gs[1, 2])
    t = st["release_t"]
    cl = [C["blue"], C["sky"], C["red"], C["orange"]]
    for k, lab in enumerate(st["release_cases"]):
        ax.plot(t, np.degrees(st["release_roll"][k] + st["release_head"][k]),
                color=cl[k], ls=("-", "--", "-", "-")[k],
                label=str(lab).replace(", k=", ", k "))
    ax.axhline(180, color="k", lw=0.5, ls=":")
    ax.text(29, 186, "face down", ha="right", fontsize=6.5,
            va="bottom")
    ax.set_xlim(0, 30)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("face roll (deg)")
    panel_label(ax, "e")
    h_e, l_e = ax.get_legend_handles_labels()
    legend_below(fig, [
        ("(c)", [line(C["blue"]), line(C["red"]), line("k", "-"),
                 line("k", "--")],
         ["trim (left axis)", "airway clearance (right axis)",
          "collar tow", "valve tow"]),
        ("(e)", h_e, l_e)])

    save_figure(fig, "fig01_model")
    rows = [[vv] + [st["%s_%s_05" % (q, tow)][i] for q in ("heave", "pitch",
                                                           "clear")
                    for tow in ("collar", "valve")]
            for i, vv in enumerate(v)]
    write_csv("fig01_equilibrium", ["bcd_L", "heave_collar", "heave_valve",
                                    "pitch_collar", "pitch_valve",
                                    "clear_collar", "clear_valve"], rows)


if __name__ == "__main__":
    main()
