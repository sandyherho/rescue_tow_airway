"""Animation 4: waiting beside the boat, at an antinode and at a node.

The wind chop (E6) reflects from a vertical hull. Two stopped victims,
head toward the hull, wait in the same sea: one about 1 m from the
hull, near the antinode where the surface rises and falls most, and one
at the first node of the closed-form surface variance, where the surface
rises least but tilts most. The dashed lines are the closed-form band of
plus and minus two standard deviations of the surface,
2 sqrt(sum 1/2 a^2 (1 + R^2 + 2 R cos 2 k_x d)), the envelope that the
moving surface fills.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.anim_style import (BAD, BG, DIM, OK, TEXT,  # noqa
                                          SideScene, save_animation)
from rescue_tow_airway.body import build_body  # noqa: E402
from rescue_tow_airway.diver_art import (airway_point, to_screen,  # noqa
                                         victim_shapes)
from rescue_tow_airway.environments import e6_hull  # noqa: E402
from rescue_tow_airway.io import path  # noqa: E402
from rescue_tow_airway.montecarlo import single_run  # noqa: E402
from rescue_tow_airway.waves import field_at_points  # noqa: E402

SEED = 31
T_SHOW, SPIN = 24.0, 20.0
D_NEAR = 1.0


def node_distance(sea):
    """First minimum of the closed-form surface variance beyond 2.5 m."""
    g = sea.spectral_grid()
    kx = g["kx"][:, 0]
    r = np.abs(g["coef"][:, 1])
    d = np.arange(2.5, 8.0, 0.01)
    var = [np.sum(0.5 * g["a2"] * (1 + r ** 2 + 2 * r * np.cos(2 * kx * x)))
           for x in d]
    return float(d[int(np.argmin(var))]), g


def main():
    body = build_body()
    sea = e6_hull()
    d_far, grid = node_distance(sea)
    runs = [single_run(body, sea, 0.0, 0.0, SEED, duration=SPIN + T_SHOW + 1,
                       origin=(-d, 0.0)) for d in (D_NEAR, d_far)]
    waves = runs[0]["waves"]
    dt = runs[0]["dt"]
    t = runs[0]["t"]
    n0 = int(round(SPIN / dt))
    nf = int(round(T_SHOW / dt))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.0, 0.215, 1.0, 0.785])
    xa = -D_NEAR
    xw = (-8.6 - xa, 1.3 - xa)
    yh = (xw[1] - xw[0]) * (0.785 * 7.2) / 12.8
    scene = SideScene(ax, body, xlim=xw, ylim=(1.5 - yh, 1.5), seed=7,
                      rescuer=False)
    # Second victim.
    vs = sorted(victim_shapes(body), key=lambda s: s["z"])
    vp = []
    for s in vs:
        p = Polygon(s["pts"], closed=s["closed"], fc=s["fc"], ec=s["ec"],
                    lw=s["lw"], alpha=s["alpha"], zorder=40 + s["z"])
        ax.add_patch(p)
        vp.append(p)
    dot2 = ax.scatter([], [], s=26, lw=0.6, edgecolors="white", zorder=80)
    glow2 = ax.scatter([], [], s=420, alpha=0.18, lw=0, zorder=79)
    off = (-d_far) - xa
    # Hull: topsides, antifouling, ladder.
    hx = -xa
    ax.add_patch(Polygon([(hx, -3.5), (hx + 2.0, -3.5), (hx + 2.0, 1.6),
                          (hx - 0.05, 1.6), (hx - 0.12, 1.25), (hx, 1.1)],
                         fc="#e9eef1", ec="#9aa7ae", lw=1.0, zorder=75))
    ax.add_patch(Polygon([(hx, -3.5), (hx + 2.0, -3.5), (hx + 2.0, 0.08),
                          (hx, 0.08)], fc="#7a2b22", ec="none", zorder=76))
    ax.add_patch(Polygon([(hx, 0.08), (hx + 2.0, 0.08), (hx + 2.0, 0.16),
                          (hx, 0.16)], fc="#1d3b57", ec="none", zorder=76))
    for yy in np.arange(-0.9, 1.2, 0.3):
        ax.plot([hx - 0.32, hx - 0.06], [yy, yy], color="#c9d3d8", lw=2.2,
                zorder=77, solid_capstyle="round")
    for xx in (hx - 0.32, hx - 0.06):
        ax.plot([xx, xx], [-1.0, 1.45], color="#c9d3d8", lw=2.6, zorder=77)
    # Closed-form envelope of the surface.
    xs = scene.x
    kx = grid["kx"][:, 0]
    r = np.abs(grid["coef"][:, 1])
    dist = np.maximum(hx - xs, 0.0)
    sd = np.sqrt([np.sum(0.5 * grid["a2"] * (1 + r ** 2 + 2 * r
                                             * np.cos(2 * kx * dd)))
                  for dd in dist])
    ax.plot(xs, 2 * sd, "--", color="#ffe2a8", lw=0.9, alpha=0.8, zorder=71)
    ax.plot(xs, -2 * sd, "--", color="#ffe2a8", lw=0.9, alpha=0.8,
            zorder=71)
    ax.text(xs[40], 2 * sd[40] + 0.05, "±2σ band of the surface "
            "(closed form)", color="#ffe2a8", fontsize=9, zorder=90)
    for lab, x in (("%.1f m from the hull" % D_NEAR, 0.0),
                   ("%.1f m (first node of elevation)" % d_far, off)):
        ax.text(x - 0.9, -1.55, lab, color=TEXT, fontsize=11,
                ha="center", zorder=90)

    hud = fig.add_axes([0.06, 0.065, 0.90, 0.125], facecolor="#06131d")
    for s in hud.spines.values():
        s.set_color("#2a4656")
    hud.tick_params(colors=DIM, labelsize=8)
    hud.set_xlim(-20, 0)
    hud.set_ylim(-0.35, 0.40)
    hud.axhline(0, color=DIM, lw=0.6, ls="--")
    hud.set_ylabel("airway\nclearance (m)", color=DIM, fontsize=8)
    hud.set_xlabel("time before now (s)", color=DIM, fontsize=8,
                   labelpad=1)
    tr = [hud.plot([], [], color=c, lw=1.0)[0] for c in ("#ff9a7a",
                                                         "#9fe8ff")]
    fig.patches.append(matplotlib.patches.FancyBboxPatch(
        (0.012, 0.835), 0.6, 0.15, boxstyle="round,pad=0.006",
        transform=fig.transFigure, fc="#02070d", ec="none", alpha=0.55))
    fig.text(0.02, 0.965, "Where to wait at the boat", color=TEXT,
             fontsize=15, fontweight="bold", va="top")
    fig.text(0.02, 0.925, "Wind chop reflected from a vertical hull (R = 0.9)"
             " | two stopped victims in the same sea | model M1",
             color=DIM, fontsize=10, va="top")
    stats = fig.text(0.02, 0.885, "", color=TEXT, fontsize=10, va="top",
                     family="monospace")
    stats2 = fig.text(0.02, 0.85, "", color=TEXT, fontsize=10, va="top",
                      family="monospace")
    fig.text(0.02, 0.235, "trace:", color=DIM, fontsize=9)
    fig.text(0.06, 0.235, "— %.1f m" % D_NEAR, color="#ff9a7a",
             fontsize=9)
    fig.text(0.11, 0.235, "— %.1f m" % d_far, color="#9fe8ff",
             fontsize=9)
    apt = airway_point(body)[None]

    def frame(i):
        n = n0 + i
        tn = t[n]
        eta = field_at_points(waves, xa + xs, np.zeros_like(xs), tn)
        qa = runs[0]["q"][0, n]
        ca = runs[0]["clear"][0, n]
        scene.update(tn, eta, qa, ca, None, 0.0)
        qb = runs[1]["q"][0, n]
        cb = runs[1]["clear"][0, n]
        for p, s in zip(vp, vs):
            p.set_xy(to_screen(s["pts"], qb) + np.array([off, 0.0]))
        a = to_screen(apt, qb)[0] + np.array([off, 0.0])
        col = OK if cb >= 0 else BAD
        dot2.set_offsets([a])
        dot2.set_color(col)
        glow2.set_offsets([a])
        glow2.set_color(col)
        lo = max(n0 - int(20 / dt), 0)
        for k, run in enumerate(runs):
            tr[k].set_data(t[lo:n + 1] - tn, run["clear"][0, lo:n + 1])
        out = []
        for run in runs:
            seg = run["clear"][0, n0:n + 1]
            out.append((100 * np.mean(seg < 0),
                        int(np.sum((seg[1:] < 0) & (seg[:-1] >= 0)))))
        stats.set_text("%.1f m: awash %4.1f%% of the time, %d submersions"
                       % (D_NEAR, *out[0]))
        stats2.set_text("%.1f m: awash %4.1f%% of the time, %d submersions"
                        % (d_far, *out[1]))
        return []

    if "--preview" in sys.argv:
        for k in (0, nf // 2):
            frame(k)
            fig.savefig(os.path.join(os.environ.get("PREVIEW_DIR", "."),
                                     "anim04_%03d.png" % k), facecolor=BG)
        return
    anim = FuncAnimation(fig, frame, frames=nf, interval=1000 * dt)
    save_animation(anim, path("animations", "anim04_hull"),
                   fps=int(round(1 / dt)), gif_width=640, gif_fps=12)
    plt.close(fig)


if __name__ == "__main__":
    main()
