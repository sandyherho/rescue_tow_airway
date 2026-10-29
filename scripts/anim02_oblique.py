"""Animation 2: oblique view of a confused sea and the rolling victim.

The victim is towed at 0.5 m/s at 150 deg to the mean wave direction
through a short-crested, broadly spread wind sea (E7). Oblique seas
excite roll, and the unsupported head rolls further about the neck.
Water surface, body pose and airway clearance come from M1; the
rescuer's figure is drawn, not simulated.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.anim_style import (BAD, BG, DIM, OK, TEXT,  # noqa
                                          save_animation)
from rescue_tow_airway.body import build_body  # noqa: E402
from rescue_tow_airway.environments import e7_confused  # noqa: E402
from rescue_tow_airway.io import path  # noqa: E402
from rescue_tow_airway.montecarlo import single_run  # noqa: E402
from rescue_tow_airway.render3d import (Camera, lambert, quad_normals,  # noqa
                                        shade_water, tube_quads)
from rescue_tow_airway.waves import field_at_points  # noqa: E402

SPEED, HEADING, SEED = 0.5, 150.0, 5
T_SHOW, SPIN = 20.0, 20.0
M = 18

DEPTH_COL = np.array(to_rgb("#0b4d63"))
SKY_COL = np.array(to_rgb("#b9d3e6"))
SUN_COL = np.array([1.0, 0.95, 0.85])
HAZE = np.array(to_rgb("#c9dbe6"))


def body_rings(body, q):
    """Rings (in the towed frame: along, left, up) for all primitives."""
    s = np.linspace(0, 2 * np.pi, M, endpoint=False)
    zeta, th, ph, chi = q
    py, pz = body.pivot
    y = body.y0[:, None] + body.b[:, None] * np.cos(s)
    z = body.z0[:, None] + body.c[:, None] * np.sin(s)
    head = body.head[:, None] == 1
    ch, sh = np.cos(chi), np.sin(chi)
    yh = py + ch * (y - py) - sh * (z - pz)
    zh = pz + sh * (y - py) + ch * (z - pz)
    y = np.where(head, yh, y)
    z = np.where(head, zh, z)
    cp, sp = np.cos(ph), np.sin(ph)
    y1 = y * cp - z * sp
    z1 = y * sp + z * cp
    ct, st = np.cos(th), np.sin(th)
    x2 = body.xi[:, None] * ct - z1 * st
    z2 = zeta + body.xi[:, None] * st + z1 * ct
    return np.stack([x2, y1, z2], -1)


def capped(rings):
    """Close a tube with shrunken end rings."""
    c0 = rings[0].mean(0)
    c1 = rings[-1].mean(0)
    a = c0 + 0.45 * (rings[0] - c0)
    b = c1 + 0.45 * (rings[-1] - c1)
    return np.concatenate([[c0 + 0.01 * (rings[0] - c0)], [a], rings, [b],
                           [c1 + 0.01 * (rings[-1] - c1)]])


def victim_mesh(body, q):
    """Quads and base colors of the victim in the towed frame."""
    r = body_rings(body, q)
    quads, cols = [], []
    k = body.kind
    order = np.argsort(body.xi)
    bodyidx = [i for i in order if k[i] == 0]
    rb = capped(r[bodyidx])
    qd = tube_quads(rb)
    nseg = rb.shape[0] - 1
    s = np.linspace(0, 2 * np.pi, M, endpoint=False)
    face = (np.sin(s) > 0.55)
    ishead = np.r_[0, 0, body.head[bodyidx], 1, 1]
    col = np.empty((nseg, M, 3))
    col[:] = to_rgb("#3b434d")
    for j in range(nseg):
        if ishead[j] and ishead[j + 1]:
            col[j, face] = to_rgb("#c8977a")
    stripe = (np.abs(np.cos(s)) > 0.97)
    for j in range(nseg):
        if not ishead[j]:
            col[j, stripe] = to_rgb("#2b6f8f")
    quads.append(qd)
    cols.append(col.reshape(-1, 3))
    for kind, color, split in ((1, "#b9c2c9", False), (2, "#4a525a", True)):
        idx = [i for i in order if k[i] == kind]
        if not idx:
            continue
        groups = ([[i for i in idx if body.y0[i] < 0],
                   [i for i in idx if body.y0[i] > 0]] if split else [idx])
        for g in groups:
            if len(g) < 2:
                continue
            rr = capped(r[g])
            qd = tube_quads(rr)
            quads.append(qd)
            cols.append(np.tile(to_rgb(color), (qd.shape[0], 1)))
    # Fins as thin plates.
    xg, zg = body.cg_shift
    zeta, th, ph, _ = q
    for side in (-1, 1):
        pts = np.array([(-1.72, side * 0.04, -0.03), (-1.72, side * 0.13,
                                                      -0.04),
                        (-2.42, side * 0.17, -0.09), (-2.42, side * 0.0,
                                                      -0.08)])
        xi = pts[:, 0] - xg
        y = pts[:, 1]
        z = pts[:, 2] - zg
        cp, sp = np.cos(ph), np.sin(ph)
        y1, z1 = y * cp - z * sp, y * sp + z * cp
        ct, st = np.cos(th), np.sin(th)
        quads.append(np.stack([xi * ct - z1 * st, y1,
                               zeta + xi * st + z1 * ct], -1)[None])
        cols.append(np.array([to_rgb("#f2c230")]))
    return np.concatenate(quads), np.concatenate(cols)


def path_tube(pts, rad, color):
    """Tube along a polyline with radii."""
    s = np.linspace(0, 2 * np.pi, M, endpoint=False)
    rings = []
    for i in range(len(pts)):
        a = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        a = a / np.linalg.norm(a)
        ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else \
            np.array([1.0, 0.0, 0.0])
        u = np.cross(a, ref)
        u /= np.linalg.norm(u)
        v = np.cross(a, u)
        rings.append(pts[i] + rad[i] * (np.outer(np.cos(s), u)
                                        + np.outer(np.sin(s), v)))
    qd = tube_quads(capped(np.array(rings)))
    return qd, np.tile(to_rgb(color), (qd.shape[0], 1))


def rescuer_mesh(anchor, t):
    """Rescuer on the victim's right, head at the surface, kicking."""
    quads, cols = [], []
    head = anchor
    neck = head + np.array([-0.10, 0.0, -0.10])
    hip = neck + np.array([-0.55, -0.05, -0.38])
    q, c = path_tube(np.array([head + [0.0, 0, 0.12], head + [0.03, 0, 0.06],
                               head, head + [-0.02, 0, -0.08]]),
                     np.array([0.03, 0.10, 0.115, 0.07]), "#151b24")
    quads.append(q)
    cols.append(c)
    q, c = path_tube(np.array([neck, 0.5 * (neck + hip), hip]),
                     np.array([0.14, 0.16, 0.14]), "#1b2433")
    quads.append(q)
    cols.append(c)
    q, c = path_tube(np.array([neck + [-0.02, 0, 0.11],
                               0.5 * (neck + hip) + [0, 0, 0.13]]),
                     np.array([0.09, 0.09]), "#b9c2c9")
    quads.append(q)
    cols.append(c)
    ph = 2 * np.pi * t / 1.1
    for side, sgn in ((-0.09, 1), (0.09, -1)):
        sw = 0.28 * sgn * np.sin(ph)
        knee = hip + np.array([-0.40, side, -0.20 + sw])
        ank = knee + np.array([-0.40, 0, -0.18 + 1.4 * sw])
        q, c = path_tube(np.array([hip + [0, side, 0], knee, ank]),
                         np.array([0.08, 0.06, 0.045]), "#1b2433")
        quads.append(q)
        cols.append(c)
        d = ank - knee
        d /= np.linalg.norm(d)
        tip = ank + 0.6 * d + np.array([0, 0, 0.25 * sgn * np.cos(ph)])
        quads.append(np.array([[ank + [0, -0.09, 0], ank + [0, 0.09, 0],
                                tip + [0, 0.11, 0], tip + [0, -0.11, 0]]]))
        cols.append(np.array([to_rgb("#33506b")]))
    # Snorkel and mask.
    q, c = path_tube(np.array([head + [-0.06, -0.06, 0.05],
                               head + [-0.07, -0.08, 0.38]]),
                     np.array([0.014, 0.014]), "#f08a24")
    quads.append(q)
    cols.append(c)
    return np.concatenate(quads), np.concatenate(cols)


def main():
    body = build_body()
    sea = e7_confused()
    h = np.radians(HEADING)
    run = single_run(body, sea, SPEED, h, SEED, duration=SPIN + T_SHOW + 1)
    dt = run["dt"]
    n0 = int(round(SPIN / dt))
    nf = int(round(T_SHOW / dt))
    q, clear, t = run["q"][0], run["clear"][0], run["t"]
    ex, ey = np.cos(h), np.sin(h)
    lx, ly = -ey, ex
    vx, vy = run["velocity"]

    # Body-following grid with fine spacing near the victim.
    u = np.linspace(-1, 1, 150)
    g = 15.0 * np.sinh(3.0 * u) / np.sinh(3.0)
    A, L = np.meshgrid(g, g, indexing="ij")

    cam = Camera(eye=(1.9, 3.5, 2.0), target=(-0.45, 0.0, -0.2), fov=48,
                 aspect=16 / 9)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-9 / 16, 9 / 16)
    ax.axis("off")
    sky = np.linspace(0, 1, 256)[:, None]
    ax.imshow(np.dstack([HAZE[0] + (0.30 - HAZE[0]) * sky,
                         HAZE[1] + (0.52 - HAZE[1]) * sky,
                         HAZE[2] + (0.74 - HAZE[2]) * sky]),
              extent=(-1, 1, -9 / 16, 9 / 16), origin="lower",
              aspect="auto", zorder=0)
    under = PolyCollection([], zorder=3.5, edgecolors="none", alpha=0.5)
    water = PolyCollection([], zorder=3, edgecolors="face", linewidths=0.4)
    over = PolyCollection([], zorder=4, edgecolors="face", linewidths=0.2)
    for pc in (under, water, over):
        ax.add_collection(pc)
    dot = ax.scatter([], [], s=60, c=OK, edgecolors="white", lw=0.8,
                     zorder=6)
    glow = ax.scatter([], [], s=500, c=OK, alpha=0.2, lw=0, zorder=5)
    fig.patches.append(matplotlib.patches.FancyBboxPatch(
        (0.012, 0.835), 0.52, 0.15, boxstyle="round,pad=0.006",
        transform=fig.transFigure, fc="#02070d", ec="none", alpha=0.55))
    fig.text(0.02, 0.965, "Oblique seas roll the victim", color=TEXT,
             fontsize=15, fontweight="bold", va="top")
    fig.text(0.02, 0.925, "Confused wind sea (Hs 0.6 m, Tp 3.5 s, broad "
             "spreading) | tow 0.5 m/s at 150° to the waves",
             color=DIM, fontsize=10, va="top")
    info = fig.text(0.02, 0.885, "", color=TEXT, fontsize=10, va="top",
                    family="monospace")
    info2 = fig.text(0.02, 0.85, "", color=TEXT, fontsize=10, va="top",
                     family="monospace")
    fig.text(0.985, 0.02, "surface, body pose and airway from model M1; "
             "rescuer drawn, not simulated; water shaded by elevation",
             color=DIM, fontsize=8,
             ha="right")
    # Wave-direction compass.
    axc = fig.add_axes([0.86, 0.80, 0.12, 0.18])
    axc.set_xlim(-1.3, 1.3)
    axc.set_ylim(-1.3, 1.3)
    axc.set_aspect("equal")
    axc.axis("off")

    def frame(i):
        n = n0 + i
        tn = t[n]
        X = run["origin"][0] + vx * tn
        Y = run["origin"][1] + vy * tn
        ex_ = X + A * ex + L * lx
        ey_ = Y + A * ey + L * ly
        eta = field_at_points(run["waves"], ex_, ey_, tn)
        P = np.stack([A, L, eta], -1)
        da = np.gradient(eta, g, axis=0)
        dl = np.gradient(eta, g, axis=1)
        nrm = np.stack([-da, -dl, np.ones_like(eta)], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        quads = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]],
                         2).reshape(-1, 4, 3)
        nq = 0.25 * (nrm[:-1, :-1] + nrm[1:, :-1] + nrm[1:, 1:]
                     + nrm[:-1, 1:]).reshape(-1, 3)
        cen = quads.mean(1)
        view = cen - cam.eye
        dist = np.linalg.norm(view, axis=-1)
        view = view / dist[:, None]
        col = shade_water(nq, view, DEPTH_COL, SKY_COL, SUN_COL)
        fog = 1 - np.exp(-np.maximum(dist - 6.0, 0) / 9.0)
        col = col * (1 - fog[:, None]) + HAZE * fog[:, None]
        slope = np.hypot(da, dl)[:-1, :-1].ravel()
        foam = np.clip((slope - 0.22) / 0.15, 0, 1) * (1 - fog)
        col = col * (1 - 0.7 * foam[:, None]) + 0.7 * foam[:, None]
        hs = 4.0 * eta.std()
        col = col + 0.45 * np.clip(cen[:, 2] / max(hs, 1e-3), -0.6, 1.0)[
            :, None] * np.array([0.05, 0.35, 0.33]) * (1 - fog[:, None])
        col = np.clip(col, 0, 1)
        p2, z = cam.project(quads)
        vis = z.min(1) > 0.3
        order = np.argsort(-z.mean(1))
        order = order[vis[order]]
        water.set_verts(p2[order])
        rgba = np.c_[col[order], np.ones(order.size)]
        water.set_facecolor(rgba)
        water.set_edgecolor(rgba)

        vq, vc = victim_mesh(body, q[n])
        a_air = vq.mean(1)
        hx = 0.68
        anchor = np.array([hx + 0.15, -0.62,
                           float(field_at_points(run["waves"],
                                                 X + (hx + 0.15) * ex
                                                 - 0.62 * lx,
                                                 Y + (hx + 0.15) * ey
                                                 - 0.62 * ly, tn)) + 0.05])
        rq, rc = rescuer_mesh(anchor, tn)
        allq = np.concatenate([vq, rq])
        allc = np.concatenate([vc, rc])
        cen = allq.mean(1)
        etac = field_at_points(run["waves"], X + cen[:, 0] * ex
                               + cen[:, 1] * lx, Y + cen[:, 0] * ey
                               + cen[:, 1] * ly, tn)
        depth = etac - cen[:, 2]
        shade = lambert(allc, quad_normals(allq))
        p2, z = cam.project(allq)
        below = depth > 0
        tint = 1 - np.exp(-np.maximum(depth, 0) / 0.25)
        cu = shade * (1 - 0.65 * tint[:, None]) + DEPTH_COL * \
            (0.65 * tint[:, None])
        o = np.argsort(-z.mean(1))
        ob = o[below[o]]
        oa = o[~below[o]]
        under.set_verts(p2[ob])
        under.set_facecolor(cu[ob])
        over.set_verts(p2[oa])
        over.set_facecolor(shade[oa])
        over.set_edgecolor(shade[oa])
        del a_air
        # Airway marker.
        from rescue_tow_airway.dynamics import point_height
        xa, ya, za = body.airway
        yg, zg = point_height(q[n][None], body, xa, ya, za, True)
        xa2 = xa * np.cos(q[n, 1]) - (zg[0] - q[n, 0] - xa
                                      * np.sin(q[n, 1])) * np.sin(q[n, 1])
        pa, _ = cam.project(np.array([xa2, yg[0], zg[0] + 0.02]))
        c = OK if clear[n] >= 0 else BAD
        dot.set_offsets([pa])
        dot.set_color(c)
        glow.set_offsets([pa])
        glow.set_color(c)
        seg = clear[n0:n + 1]
        info.set_text("t = %5.1f s   torso roll %+6.1f°   head roll "
                      "%+6.1f°" % (tn - t[n0], np.degrees(q[n, 2]),
                                   np.degrees(q[n, 3])))
        info2.set_text("airway clearance %+5.2f m   awash %4.1f%% of the "
                       "time" % (clear[n], 100 * np.mean(seg < 0)))
        axc.cla()
        axc.set_xlim(-1.3, 1.3)
        axc.set_ylim(-1.3, 1.3)
        axc.axis("off")
        circ = plt.Circle((0, 0), 1.1, fc="#02070d", ec=DIM, alpha=0.6)
        axc.add_patch(circ)
        # Screen directions of the wave propagation and the tow.
        w0 = cam.project(np.array([0.0, 0.0, 0.0]))[0]
        wdir = np.array([np.cos(-h), np.sin(-h), 0.0])
        w1 = cam.project(wdir * 2.0)[0] - w0
        w1 /= np.linalg.norm(w1)
        t1 = cam.project(np.array([2.0, 0.0, 0.0]))[0] - w0
        t1 /= np.linalg.norm(t1)
        axc.annotate("", xy=0.85 * w1, xytext=-0.85 * w1,
                     arrowprops=dict(arrowstyle="-|>", color="#9fd8e8",
                                     lw=1.6))
        axc.annotate("", xy=0.85 * t1, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color="#ffb547",
                                     lw=1.6))
        axc.text(0, -1.28, "waves", color="#9fd8e8", ha="right",
                 fontsize=8, va="top")
        axc.text(0.05, -1.28, " tow", color="#ffb547", ha="left",
                 fontsize=8, va="top")
        return []

    if "--preview" in sys.argv:
        for k in (0, nf // 2):
            frame(k)
            fig.savefig(os.path.join(os.environ.get("PREVIEW_DIR", "."),
                                     "anim02_%03d.png" % k), facecolor=BG)
        return
    anim = FuncAnimation(fig, frame, frames=nf, interval=1000 * dt)
    save_animation(anim, path("animations", "anim02_oblique"),
                   fps=int(round(1 / dt)), gif_width=560, gif_fps=10)
    plt.close(fig)


if __name__ == "__main__":
    main()
