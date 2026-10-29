"""Animation 3: a radar sweep over tow speed and heading.

For each environment a sweep line reveals the closed-form (M0)
probability that a rescue breath meets water, as a function of tow
heading (angle; the waves travel up the screen) and tow speed (radius).
Monte Carlo (M1) estimates light up as the sweep passes them, on the
same color scale.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.anim_style import BG, DIM, TEXT, save_animation  # noqa
from rescue_tow_airway.io import has_cache, load_cache, path  # noqa: E402

ENVS = [("E2", "Wind chop", "Hs 0.6 m, Tp 3.5 s"),
        ("E7", "Confused sea", "Hs 0.6 m, Tp 3.5 s, broad spreading"),
        ("E3", "Swell crossed by a wind sea", "swell 0.8 m / 11 s, wind sea"
         " 0.5 m / 4 s at 60°"),
        ("E4m", "Wind sea against a 0.75 m/s current", "Hs 0.5 m, Tp 4 s "
         "before the current")]
FPS = 25
SWEEP, HOLD = 4.0, 2.4
NORM = LogNorm(1e-3, 1.0)
CMAP = plt.get_cmap("magma_r")


def main():
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.30, 0.06, 0.48, 0.85], projection="polar",
                      facecolor="#06131d")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5 m/s", "", "1 m/s"], color=DIM, fontsize=9)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["tow with\nthe waves", "across", "tow into\nthe "
                        "waves", "across"], color=TEXT, fontsize=10)
    ax.grid(color="#2a4656", lw=0.6)
    ax.spines["polar"].set_color("#2a4656")
    data = []
    for tag, title, sub in ENVS:
        m = load_cache("m0_map_" + tag)
        th = np.radians(np.r_[m["headings"], 360.0])
        p = np.c_[m["p_breath"], m["p_breath"][:, :1]]
        pts = None
        if has_cache("m1_map_" + tag):
            m1 = load_cache("m1_map_" + tag)
            a, r, c = [], [], []
            for i, u in enumerate(m1["speeds"]):
                for j, h in enumerate(m1["headings"]):
                    hs = {h, (360 - h) % 360} if tag != "E3" else {h}
                    for hm in hs:
                        a.append(np.radians(hm))
                        r.append(u)
                        c.append(m1["p_breath"][i, j])
            pts = (np.array(a), np.array(r), np.array(c))
        data.append((th, m["speeds"], p, pts, title, sub))
    ttl = fig.text(0.03, 0.93, "", color=TEXT, fontsize=17,
                   fontweight="bold", va="top")
    sub = fig.text(0.03, 0.875, "", color=DIM, fontsize=10.5, va="top")
    fig.text(0.03, 0.80, "Probability that a rescue breath\nmeets water, "
             "for every tow\nheading and speed.", color=TEXT, fontsize=11,
             va="top", linespacing=1.4)
    fig.text(0.03, 0.62, "shading: closed form (M0)\ndots: Monte Carlo "
             "(M1), same color scale", color=DIM,
             fontsize=10, va="top", linespacing=1.5)
    fig.text(0.03, 0.42, "↑  waves travel up", color="#9fd8e8",
             fontsize=11, va="top")
    best = fig.text(0.03, 0.30, "", color=TEXT, fontsize=10.5, va="top",
                    linespacing=1.5, family="monospace")
    cax = fig.add_axes([0.83, 0.18, 0.015, 0.6])
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(colors=DIM, labelsize=9)
    cb.set_label("P(breath meets water)", color=DIM, fontsize=10)
    cb.outline.set_edgecolor("#2a4656")
    per = int((SWEEP + HOLD) * FPS)
    state = dict(k=-1, mesh=None, dots=None, line=None,
                 glow=[])

    def frame(i):
        k = i // per
        f = (i % per) / FPS
        th, sp, p, pts, title, subt = data[k]
        if state["k"] != k:
            for key in ("mesh", "dots", "line"):
                if state[key] is not None:
                    state[key].remove()
                    state[key] = None
            for g in state["glow"]:
                g.remove()
            state["glow"] = []
            state["mesh"] = ax.pcolormesh(th, sp, np.clip(p, 1.01e-3, 1),
                                          norm=NORM, cmap=CMAP,
                                          shading="gouraud", zorder=1)
            state["line"], = ax.plot([], [], color="#ffe2a8", lw=2.2,
                                     zorder=5)
            if pts is not None:
                state["dots"] = ax.scatter(np.empty(0), np.empty(0),
                                           c=np.empty(0), s=90,
                                           edgecolors="white", lw=1.5,
                                           zorder=6, cmap=CMAP,
                                           norm=NORM)
            ttl.set_text(title)
            sub.set_text(subt)
            i0 = np.unravel_index(np.argmin(p[3:8, :-1]), p[3:8, :-1].shape)
            hb = np.degrees(th[i0[1]])
            j1 = np.unravel_index(np.argmax(p[3:8, :-1]), p[3:8, :-1].shape)
            hw = np.degrees(th[j1[1]])
            best.set_text("M0, tows of 0.3 to 0.7 m/s\nsafest heading %5.1f°"
                          "\n  P = %.1e\nworst heading  %5.1f°\n  P = "
                          "%.1e" % (hb, p[3 + i0[0], i0[1]], hw,
                                    p[3 + j1[0], j1[1]]))
            state["k"] = k
        ang = min(f / SWEEP, 1.0) * 2 * np.pi
        alpha_mask = np.where(th <= ang + 1e-9, 1.0, 0.0)
        cols = np.clip(p, 1.01e-3, 1)
        arr = np.ma.masked_where(np.broadcast_to(alpha_mask[None, :],
                                                 cols.shape) < 0.5, cols)
        state["mesh"].set_array(arr)
        if f < SWEEP:
            state["line"].set_data([ang, ang], [0, 1.0])
        else:
            state["line"].set_data([], [])
        if pts is not None:
            sel = pts[0] <= ang + 1e-9
            if sel.any():
                state["dots"].set_offsets(np.c_[pts[0][sel], pts[1][sel]])
                state["dots"].set_array(np.clip(pts[2][sel], 1.01e-3, 1))
                fresh = np.abs(pts[0][sel] - ang) < 0.25
                sizes = np.where(fresh, 160, 70)
                state["dots"].set_sizes(sizes)
        return []

    nfr = per * len(data)
    if "--preview" in sys.argv:
        for k in (int(per * 0.4), int(per * 0.9)):
            frame(k)
            fig.savefig(os.path.join(os.environ.get("PREVIEW_DIR", "."),
                                     "anim03_%03d.png" % k), facecolor=BG)
        return
    anim = FuncAnimation(fig, frame, frames=nfr, interval=1000 / FPS)
    save_animation(anim, path("animations", "anim03_riskradar"), fps=FPS,
                   gif_width=640, gif_fps=12)
    plt.close(fig)


if __name__ == "__main__":
    main()
