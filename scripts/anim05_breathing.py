"""Animation 5: the trained rhythm against breaths timed to a clear airway.

One tow, one sea, two rescuers' choices. The face cam shows the victim's
head seen from the crown with the local water line, exactly as M1
computed it. Breathing does not change the dynamics, so both strategies
act on the same clearance record:

* left counter: a breath every 5 s (the trained rhythm);
* right counter: when the next breath is due, wait until the airway is
  clear of the water by at least 3 cm and rising, for at most 3 s.

The excerpt is chosen among 12 seeds as the one where the two strategies
differ most, to show the mechanism; the statistics over all tows are in
Fig. 6.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.anim_style import (BAD, BG, DIM, OK, TEXT,  # noqa
                                          CrownInset, save_animation)
from rescue_tow_airway.body import build_body  # noqa: E402
from rescue_tow_airway.breathing import evaluate  # noqa: E402
from rescue_tow_airway.environments import e2_chop  # noqa: E402
from rescue_tow_airway.io import path  # noqa: E402
from rescue_tow_airway.montecarlo import single_run  # noqa: E402
from rescue_tow_airway.waves import field_at_points  # noqa: E402

SPEED, HEADING = 0.5, 180.0
T_SHOW, SPIN = 30.0, 20.0
STRATS = (("fixed", "every 5 s"), ("clear", "when the airway is clear"))


def pick(body, sea):
    """Seed whose excerpt separates the two strategies most."""
    best = None
    h = np.radians(HEADING)
    for seed in range(40, 52):
        run = single_run(body, sea, SPEED, h, seed,
                         duration=SPIN + T_SHOW + 1)
        dt = run["dt"]
        n0 = int(round(SPIN / dt))
        res = [evaluate(run["clear"], run["water"], dt, s, first=n0,
                        rng=np.random.default_rng(4)) for s, _ in STRATS]
        nf = [int((~r["ok"][0]).sum()) for r in res]
        score = nf[0] - nf[1]
        if best is None or score > best[0]:
            best = (score, seed, run, res)
    return best[1:]


def main():
    body = build_body()
    sea = e2_chop()
    seed, run, res = pick(body, sea)
    dt = run["dt"]
    n0 = int(round(SPIN / dt))
    nf = int(round(T_SHOW / dt))
    t, q = run["t"], run["q"][0]
    clear, water = run["clear"][0], run["water"][0]
    h = np.radians(HEADING)
    ex, ey = np.cos(h), np.sin(h)
    vx, vy = run["velocity"]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    cam = fig.add_axes([0.03, 0.22, 0.42, 0.62])
    inset = CrownInset(cam, body)
    tl = fig.add_axes([0.52, 0.30, 0.45, 0.42], facecolor="#06131d")
    for s in tl.spines.values():
        s.set_color("#2a4656")
    tl.tick_params(colors=DIM, labelsize=9)
    tl.set_xlim(-12, 4)
    tl.set_ylim(-0.25, 0.55)
    tl.axhline(0, color=DIM, lw=0.6, ls="--")
    tl.axvline(0, color=TEXT, lw=0.8, alpha=0.5)
    tl.set_xlabel("time relative to now (s)", color=DIM)
    tl.set_ylabel("airway clearance (m)", color=DIM)
    trace, = tl.plot([], [], color=TEXT, lw=1.2)
    fills = [None]
    rows_y = (0.36, 0.47)
    marks = [tl.scatter([], [], s=80, marker="v", lw=0) for _ in STRATS]
    for (s, lab), y in zip(STRATS, rows_y):
        tl.text(-11.7, y + 0.035, lab, color=DIM, fontsize=9)
    rings = []
    counters = []
    for k, (s, lab) in enumerate(STRATS):
        x0 = 0.53 + 0.235 * k
        axr = fig.add_axes([x0, 0.76, 0.06, 0.106])
        axr.set_xlim(-1.2, 1.2)
        axr.set_ylim(-1.2, 1.2)
        axr.set_aspect("equal")
        axr.axis("off")
        bg = Wedge((0, 0), 1.0, 0, 360, width=0.22, fc="#1b3240")
        fg = Wedge((0, 0), 1.0, 90, 90, width=0.22, fc=OK)
        axr.add_patch(bg)
        axr.add_patch(fg)
        txt = axr.text(0, 0, "", color=TEXT, ha="center", va="center",
                       fontsize=10)
        rings.append((fg, txt))
        fig.text(x0 + 0.07, 0.85, lab, color=TEXT, fontsize=11,
                 fontweight="bold", va="top")
        counters.append(fig.text(x0 + 0.07, 0.81, "", color=TEXT,
                                 fontsize=10, va="top",
                                 family="monospace"))
    fig.text(0.03, 0.965, "When to give the rescue breath", color=TEXT,
             fontsize=16, fontweight="bold", va="top")
    fig.text(0.03, 0.92, "Wind chop (Hs 0.6 m, Tp 3.5 s) | tow 0.5 m/s into "
             "the waves | same tow, two breathing rules | model M1",
             color=DIM, fontsize=10, va="top")
    fig.text(0.03, 0.185, "Face cam: the victim's head and chest seen from "
             "the crown; the line is the local water surface.",
             color=DIM, fontsize=9)
    fig.text(0.52, 0.19, "▼ breath: green = dry, red = met water. "
             "Ring: time until the next breath is due.", color=DIM,
             fontsize=9)
    fig.text(0.97, 0.03, "excerpt chosen among 12 simulated tows (seed %d) "
             "to show the mechanism; statistics over all tows in Fig. 6"
             % seed,
             color="#5f7f8f", fontsize=8, ha="right")

    def frame(i):
        n = n0 + i
        tn = t[n]
        X = run["origin"][0] + vx * tn
        Y = run["origin"][1] + vy * tn
        dn = 0.05
        e_l = field_at_points(run["waves"], X - dn * ey, Y + dn * ex, tn)
        e_r = field_at_points(run["waves"], X + dn * ey, Y - dn * ex, tn)
        inset.update(q[n], water[n], (e_l - e_r) / (2 * dn), clear[n])
        lo = max(n - int(12 / dt), 0)
        tt = t[lo:n + 1] - tn
        cc = clear[lo:n + 1]
        trace.set_data(tt, cc)
        if fills[0] is not None:
            fills[0].remove()
        fills[0] = tl.fill_between(tt, cc, 0, where=cc < 0, color=BAD,
                                   alpha=0.6, lw=0)
        for k, r in enumerate(res):
            bt = r["times"][0]
            ok = r["ok"][0]
            sel = (bt >= lo) & (bt <= n)
            marks[k].set_offsets(np.c_[t[bt[sel]] - tn,
                                       np.full(sel.sum(), rows_y[k])]
                                 if sel.any() else np.empty((0, 2)))
            marks[k].set_color([OK if o else BAD for o in ok[sel]])
            done = (bt >= n0) & (bt <= n)
            nb = int(done.sum())
            nbad = int((~ok[done]).sum())
            counters[k].set_text("%d breaths, %d met water" % (nb, nbad))
            past = bt[bt <= n]
            last = past[-1] if past.size else n0
            due = last + int(round(5.0 / dt))
            frac = np.clip((n - last) / (due - last), 0, 1)
            fg, txt = rings[k]
            fg.set_theta1(90 - 360 * frac)
            waiting = n >= due
            fg.set_color("#ffb547" if waiting else OK)
            txt.set_text("wait" if waiting else "%.0f" % max(
                (due - n) * dt, 0))
        return []

    if "--preview" in sys.argv:
        for k in (nf // 3, 2 * nf // 3):
            frame(k)
            fig.savefig(os.path.join(os.environ.get("PREVIEW_DIR", "."),
                                     "anim05_%03d.png" % k), facecolor=BG)
        return
    anim = FuncAnimation(fig, frame, frames=nf, interval=1000 * dt)
    save_animation(anim, path("animations", "anim05_breathing"),
                   fps=int(round(1 / dt)), gif_width=640, gif_fps=12)
    plt.close(fig)


if __name__ == "__main__":
    main()
