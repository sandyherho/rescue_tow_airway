"""Animation 1: the tow, the waves and the airway, seen from the side.

A rescuer tows an unconscious diver at 0.5 m/s into a wind chop (E2)
and gives a rescue breath every 5 s. The surface is the simulated sea
along the tow line; the victim's heave, pitch and roll are the M1
state. The airway marker turns red whenever the airway is awash. The
inset shows the head and chest seen from the crown, with the local water
line, so roll toward the water is visible. The strip at the bottom is
the airway clearance over the last 20 s.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.anim_style import (BAD, BG, DIM, OK, TEXT,  # noqa
                                          CrownInset, SideScene,
                                          save_animation)
from rescue_tow_airway.body import build_body  # noqa: E402
from rescue_tow_airway.breathing import evaluate  # noqa: E402
from rescue_tow_airway.environments import e2_chop  # noqa: E402
from rescue_tow_airway.io import path  # noqa: E402
from rescue_tow_airway.montecarlo import single_run  # noqa: E402
from rescue_tow_airway.waves import field_at_points  # noqa: E402

SPEED, HEADING, SEED = 0.5, 180.0, 21
T_SHOW, SPIN = 24.0, 20.0


def main():
    body = build_body()
    sea = e2_chop()
    h = np.radians(HEADING)
    run = single_run(body, sea, SPEED, h, SEED, duration=SPIN + T_SHOW + 1)
    dt = run["dt"]
    n0 = int(round(SPIN / dt))
    nf = int(round(T_SHOW / dt))
    t = run["t"]
    q = run["q"][0]
    clear = run["clear"][0]
    water = run["water"][0]
    br = evaluate(run["clear"], run["water"], dt, "fixed", first=n0,
                  rng=np.random.default_rng(4))
    btimes, bok = br["times"][0], br["ok"][0]
    ex, ey = np.cos(h), np.sin(h)
    vx, vy = run["velocity"]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.0, 0.215, 1.0, 0.785])
    xw = (-3.0, 2.6)
    yh = (xw[1] - xw[0]) * (0.785 * 7.2) / 12.8
    scene = SideScene(ax, body, xlim=xw, ylim=(0.72 - yh, 0.72), seed=3)
    axi = fig.add_axes([0.015, 0.25, 0.20, 0.36])
    inset = CrownInset(axi, body)
    hud = fig.add_axes([0.06, 0.065, 0.90, 0.125], facecolor="#06131d")
    for s in hud.spines.values():
        s.set_color("#2a4656")
    hud.tick_params(colors=DIM, labelsize=8)
    hud.set_xlim(-20, 0)
    hud.set_ylim(-0.25, 0.30)
    hud.axhline(0, color=DIM, lw=0.6, ls="--")
    hud.set_ylabel("airway\nclearance (m)", color=DIM, fontsize=8)
    hud.set_xlabel("time before now (s)", color=DIM, fontsize=8,
                   labelpad=1)
    trace, = hud.plot([], [], color=TEXT, lw=1.0)
    fills = [None]
    bmarks = hud.scatter([], [], s=30, marker="v", lw=0)
    fig.patches.append(matplotlib.patches.FancyBboxPatch(
        (0.012, 0.835), 0.56, 0.15, boxstyle="round,pad=0.006",
        transform=fig.transFigure, fc="#02070d", ec="none", alpha=0.55,
        zorder=0))
    fig.text(0.02, 0.965, "Surface tow of an unconscious diver",
             color=TEXT, fontsize=15, fontweight="bold", va="top")
    fig.text(0.02, 0.925, "Wind chop (Hs 0.6 m, Tp 3.5 s) | tow 0.5 m/s "
             "into the waves | rescue breath every 5 s | model M1",
             color=DIM, fontsize=10, va="top")
    clock = fig.text(0.02, 0.885, "", color=TEXT, fontsize=10, va="top",
                     family="monospace")
    stats = fig.text(0.02, 0.85, "", color=TEXT, fontsize=10, va="top",
                     family="monospace")
    fig.text(0.02, 0.235, "airway:", color=DIM, fontsize=9)
    fig.text(0.065, 0.235, "● clear", color=OK, fontsize=9)
    fig.text(0.112, 0.235, "● awash", color=BAD, fontsize=9)
    fig.text(0.162, 0.235, "▼ breath (green: dry, red: met water)",
             color=DIM, fontsize=9)
    fig.text(0.985, 0.235, "rescuer drawn, not simulated",
             color="#5f7f8f", fontsize=7.5, ha="right")

    def frame(i):
        n = n0 + i
        tn = t[n]
        X = run["origin"][0] + vx * tn
        Y = run["origin"][1] + vy * tn
        xs = scene.x
        eta = field_at_points(run["waves"], X + xs * ex, Y + xs * ey, tn)
        past = btimes[btimes <= n]
        breath = None
        if past.size:
            k = past.size - 1
            breath = ((n - past[-1]) * dt, bool(bok[k]))
        scene.update(tn, eta, q[n], clear[n], breath, SPEED)
        dn = 0.05
        e_l = field_at_points(run["waves"], X - dn * ey, Y + dn * ex, tn)
        e_r = field_at_points(run["waves"], X + dn * ey, Y - dn * ex, tn)
        inset.update(q[n], water[n], (e_l - e_r) / (2 * dn), clear[n])
        lo = max(n0 - int(20 / dt), 0)
        tt = t[lo:n + 1] - tn
        cc = clear[lo:n + 1]
        trace.set_data(tt, cc)
        if fills[0] is not None:
            fills[0].remove()
        fills[0] = hud.fill_between(tt, cc, 0, where=cc < 0, color=BAD,
                                    alpha=0.6, lw=0)
        sel = (btimes >= lo) & (btimes <= n)
        if np.any(sel):
            bt = t[btimes[sel]] - tn
            bmarks.set_offsets(np.c_[bt, np.full(bt.size, 0.25)])
            bmarks.set_color([OK if o else BAD for o in bok[sel]])
        seg = clear[n0:n + 1]
        wet = 100.0 * np.mean(seg < 0)
        ev = int(np.sum((seg[1:] < 0) & (seg[:-1] >= 0)))
        done = btimes <= n
        nb = int(done.sum())
        nf_ = int((~bok[done]).sum())
        clock.set_text("t = %5.1f s   roll %+5.1f°   head roll %+5.1f°"
                       % (tn - t[n0], np.degrees(q[n, 2]),
                          np.degrees(q[n, 3])))
        stats.set_text("airway awash %4.1f%% of the time   %d submersions"
                       "   breaths %d (%d met water)" % (wet, ev, nb, nf_))
        return []

    if "--preview" in sys.argv:
        for k in (0, nf // 3, 2 * nf // 3):
            frame(k)
            fig.savefig(os.path.join(os.environ.get("PREVIEW_DIR", "."),
                                     "anim01_%03d.png" % k),
                        facecolor=BG)
        return
    anim = FuncAnimation(fig, frame, frames=nf, interval=1000 * dt,
                         blit=False)
    save_animation(anim, path("animations", "anim01_tow"),
                   fps=int(round(1 / dt)), gif_width=640, gif_fps=12)
    plt.close(fig)


if __name__ == "__main__":
    main()
