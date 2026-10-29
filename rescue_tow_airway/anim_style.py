"""Shared scene rendering for the animations.

The side-view scene follows the towed victim. Everything drawn in the
water comes from the simulation: the free surface is the plane-wave
realization that forced the body, evaluated along the tow line, and the
victim's heave, pitch and roll are the M1 state. Decorative elements
(light shafts, suspended particles, foam on steep faces, sky) are added
for legibility and carry no physical meaning, except that foam is drawn
only where the simulated surface slope is steep.
"""

import os
import subprocess

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, Polygon

from .diver_art import (airway_point, rescuer_shapes, to_screen,
                        victim_shapes)
from .dynamics import point_height

DEEP = "#03101c"
MID = "#0a3f5c"
SHALLOW = "#1f8a9a"
SURF_LINE = "#e9fbff"
FOAM = "#f4fdff"
SKY_TOP = "#3f7fb8"
SKY_LOW = "#f3d6b4"
BG = "#02070d"
TEXT = "#e8f1f5"
DIM = "#8fb2c2"
OK = "#46e08a"
BAD = "#ff4d4d"
WARN = "#ffb547"

WATER_CMAP = LinearSegmentedColormap.from_list(
    "water", [DEEP, MID, SHALLOW])
SKY_CMAP = LinearSegmentedColormap.from_list(
    "sky", [SKY_LOW, "#c9d8e3", SKY_TOP])


def gradient(cmap, n=256, gamma=1.0):
    """Vertical gradient image (bottom to top) for imshow."""
    v = np.linspace(0, 1, n) ** gamma
    return cmap(v)[:, None, :]


class SideScene:
    """A side-view underwater scene that tracks a towed victim.

    Parameters
    ----------
    ax : matplotlib Axes
    body : Body
    xlim, ylim : tuple
        View window in meters relative to the victim's center of gravity.
    seed : int
        Seed of the decorative particles.
    rescuer : bool
        Draw the rescuer.
    """

    def __init__(self, ax, body, xlim=(-3.0, 1.6), ylim=(-1.35, 0.85),
                 seed=0, rescuer=True, nx=400):
        self.ax = ax
        self.body = body
        self.xlim, self.ylim = xlim, ylim
        self.x = np.linspace(xlim[0] - 0.2, xlim[1] + 0.2, nx)
        rng = np.random.default_rng(seed)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        ext = (xlim[0] - 1, xlim[1] + 1, ylim[0] - 0.5, 0.25)
        ax.imshow(gradient(WATER_CMAP, gamma=1.6), extent=ext,
                  origin="lower", aspect="auto", zorder=0,
                  interpolation="bicubic")
        self.sky_patch = Polygon(np.zeros((3, 2)), closed=True,
                                 transform=ax.transData)
        self.sky = ax.imshow(gradient(SKY_CMAP, gamma=0.7),
                             extent=(xlim[0] - 1, xlim[1] + 1, -0.6,
                                     ylim[1] + 0.2), origin="lower",
                             aspect="auto", zorder=1,
                             interpolation="bicubic")
        self.water_patch = Polygon(np.zeros((3, 2)), closed=True,
                                   transform=ax.transData)
        # Light shafts.
        self.rays = []
        for k in range(9):
            x0 = rng.uniform(xlim[0], xlim[1] + 1.0)
            w = rng.uniform(0.04, 0.16)
            a = rng.uniform(0.025, 0.06)
            p = Polygon(np.zeros((4, 2)), closed=True, fc="#bff3ff",
                        ec="none", alpha=a, zorder=2)
            ax.add_patch(p)
            self.rays.append((p, x0, w, rng.uniform(0, 2 * np.pi),
                              rng.uniform(0.15, 0.3)))
        # Suspended particles drifting relative to the camera.
        n = 140
        self.px = rng.uniform(xlim[0], xlim[1], n)
        self.pz = rng.uniform(ylim[0], 0.0, n)
        self.ps = rng.uniform(0.3, 2.2, n)
        self.part = ax.scatter(self.px, self.pz, s=self.ps, c="#d8f6ff",
                               alpha=0.35, lw=0, zorder=3)
        # Divers.
        self.vshapes = sorted(victim_shapes(body), key=lambda d: d["z"])
        self.vpatches = []
        for s in self.vshapes:
            p = Polygon(s["pts"], closed=s["closed"], fc=s["fc"],
                        ec=s["ec"], lw=s["lw"], alpha=s["alpha"],
                        zorder=40 + s["z"], joinstyle="round")
            ax.add_patch(p)
            self.vpatches.append(p)
        self.rpatches = []
        self.rarm = None
        if rescuer:
            shapes, _ = rescuer_shapes(0.0)
            for s in sorted(shapes, key=lambda d: d["z"]):
                p = Polygon(s["pts"], closed=s["closed"], fc=s["fc"],
                            ec=s["ec"], lw=s["lw"], alpha=s["alpha"],
                            zorder=20 + s["z"], joinstyle="round")
                ax.add_patch(p)
                self.rpatches.append(p)
            self.rarm, = ax.plot([], [], color="#1b2433", lw=5.5,
                                 solid_capstyle="round", zorder=62)
            self.rhand, = ax.plot([], [], "o", ms=5, color="#0f141c",
                                  zorder=63)
        # Water tint over submerged parts, surface line, foam.
        self.tint = Polygon(np.zeros((3, 2)), closed=True, fc="#0d5a72",
                            ec="none", alpha=0.42, zorder=70)
        ax.add_patch(self.tint)
        self.surf, = ax.plot([], [], color=SURF_LINE, lw=1.6, zorder=72,
                             alpha=0.95)
        self.glint, = ax.plot([], [], color="white", lw=0.6, zorder=73,
                              alpha=0.7)
        self.foam = ax.scatter([], [], s=4, c=FOAM, alpha=0.8, lw=0,
                               zorder=74)
        self.spray = ax.scatter([], [], s=6, c=FOAM, alpha=0.9, lw=0,
                                zorder=81)
        self.glow = ax.scatter([], [], s=420, c=OK, alpha=0.18, lw=0,
                               zorder=79)
        self.dot = ax.scatter([], [], s=26, c=OK, lw=0.6,
                              edgecolors="white", zorder=80)
        self.puff = ax.scatter([], [], s=[], c=[], alpha=0.85, lw=0,
                               zorder=82)
        self.rng = rng
        self.apt = airway_point(body)[None]

    def update(self, t, eta, q, clear, breath=None, speed=0.0, slope=None):
        """Redraw for one frame.

        Parameters
        ----------
        t : float
            Time (s).
        eta : ndarray
            Surface elevation on ``self.x`` (m).
        q : ndarray (4,)
            Victim state.
        clear : float
            Airway clearance (m).
        breath : tuple or None
            ``(age, ok)`` of the most recent breath (s, bool).
        speed : float
            Tow speed, used for particle drift.
        """
        x = self.x
        y0, y1 = self.ylim
        top = np.r_[x, x[::-1]], np.r_[eta, np.full(x.size, y1 + 0.5)]
        bot = np.r_[x, x[::-1]], np.r_[eta, np.full(x.size, y0 - 0.5)]
        self.sky_patch.set_xy(np.stack(top, 1))
        self.sky.set_clip_path(self.sky_patch)
        self.water_patch.set_xy(np.stack(bot, 1))
        self.tint.set_xy(np.stack(bot, 1))
        self.surf.set_data(x, eta)
        self.glint.set_data(x, eta - 0.012)
        # Light shafts from the moving surface.
        for p, x0, w, ph, tilt in self.rays:
            xs = ((x0 - 0.35 * speed * t) - self.xlim[0]) % \
                (self.xlim[1] - self.xlim[0] + 1.2) + self.xlim[0] - 0.6
            sway = 0.08 * np.sin(0.6 * t + ph)
            zt = np.interp(xs, x, eta) - 0.02
            p.set_xy([(xs - w, zt), (xs + w, zt),
                      (xs + w + tilt + sway + 0.25, y0 - 0.3),
                      (xs - w + tilt + sway - 0.05, y0 - 0.3)])
            p.set_clip_path(self.water_patch)
        # Particles drift backward with the tow.
        dx = -speed * 0.04
        self.px = (self.px + dx - self.xlim[0]) % \
            (self.xlim[1] - self.xlim[0]) + self.xlim[0]
        self.pz = self.pz + 0.002 * np.sin(0.7 * t + self.px * 3)
        self.part.set_offsets(np.c_[self.px, self.pz])
        self.part.set_clip_path(self.water_patch)
        # Foam where the simulated surface is steep and near a crest.
        sl = np.gradient(eta, x)
        crest = eta > np.percentile(eta, 70)
        sel = (np.abs(sl) > 0.12) & crest
        if np.any(sel):
            xf = np.repeat(x[sel], 2) + self.rng.normal(0, 0.01,
                                                        2 * sel.sum())
            zf = np.repeat(eta[sel], 2) + self.rng.normal(0, 0.006,
                                                          2 * sel.sum())
            self.foam.set_offsets(np.c_[xf, zf])
        else:
            self.foam.set_offsets(np.empty((0, 2)))
        # Victim.
        for p, s in zip(self.vpatches, self.vshapes):
            p.set_xy(to_screen(s["pts"], q))
        a = to_screen(self.apt, q)[0]
        # Rescuer: ahead of the victim's head, riding the surface.
        if self.rpatches:
            hx = a[0] + 0.42
            hz = np.interp(hx, x, eta) + 0.075
            slope_r = np.interp(hx, x, sl)
            rot = np.arctan(slope_r) * 0.6
            cr, sr = np.cos(rot), np.sin(rot)
            shapes, sh = rescuer_shapes(t)
            for p, s in zip(self.rpatches,
                            sorted(shapes, key=lambda d: d["z"])):
                pts = s["pts"]
                pts = np.stack([pts[:, 0] * cr - pts[:, 1] * sr,
                                pts[:, 0] * sr + pts[:, 1] * cr], 1)
                p.set_xy(pts + np.array([hx, hz]))
            shw = np.array([sh[0] * cr - sh[1] * sr,
                            sh[0] * sr + sh[1] * cr]) + np.array([hx, hz])
            vsh = to_screen(np.array([[-0.05 - self.body.cg_shift[0],
                                       0.03 - self.body.cg_shift[1]]]), q)[0]
            elbow = 0.5 * (shw + vsh) + np.array([0.0, -0.08])
            self.rarm.set_data([shw[0], elbow[0], vsh[0]],
                               [shw[1], elbow[1], vsh[1]])
            self.rhand.set_data([vsh[0]], [vsh[1]])
        # Airway marker and spray when awash.
        col = OK if clear >= 0 else BAD
        self.dot.set_offsets([a])
        self.dot.set_color(col)
        self.glow.set_offsets([a])
        self.glow.set_color(col)
        self.glow.set_sizes([420 + 160 * np.sin(6 * t) ** 2])
        if clear < 0:
            n = 26
            ang = self.rng.uniform(0.2, np.pi - 0.2, n)
            rr = self.rng.uniform(0.02, 0.12, n)
            self.spray.set_offsets(np.c_[a[0] + rr * np.cos(ang),
                                         a[1] + 0.02 + rr * np.sin(ang)])
        else:
            self.spray.set_offsets(np.empty((0, 2)))
        if breath is not None and breath[0] < 1.2:
            age, ok = breath
            n = 12
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.4 * t
            rr = 0.03 + 0.12 * age
            pts = np.c_[a[0] + rr * np.cos(ang), a[1] + 0.06 + 0.6 * rr
                        + 0.5 * rr * np.sin(ang)]
            self.puff.set_offsets(pts)
            self.puff.set_sizes(np.full(n, 40 * (1.2 - age)))
            self.puff.set_color(OK if ok else BAD)
        else:
            self.puff.set_offsets(np.empty((0, 2)))
        return a


def save_animation(anim, name, fps, gif_width=640, gif_fps=None,
                   dpi=None):
    """Write an MP4 (H.264) and a palette-optimized GIF.

    Parameters
    ----------
    anim : matplotlib.animation.FuncAnimation
    name : str
        Base path without extension.
    fps : int
        Frame rate of the MP4.
    gif_width : int
        Width of the GIF in pixels.
    gif_fps : int, optional
        Frame rate of the GIF (defaults to ``fps``).
    """
    from matplotlib.animation import FFMpegWriter
    mp4 = name + ".mp4"
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                          extra_args=["-pix_fmt", "yuv420p", "-preset",
                                      "veryfast", "-crf", "20"])
    anim.save(mp4, writer=writer, dpi=dpi)
    gif = mp4_to_gif(mp4, gif_width, gif_fps or fps)
    return mp4, gif


def mp4_to_gif(mp4, width=640, fps=12, colors=128):
    """Palette-optimized GIF made from an MP4 with FFmpeg.

    Parameters
    ----------
    mp4 : str
        Path of the source video.
    width : int
        GIF width in pixels.
    fps : int
        GIF frame rate.
    colors : int
        Size of the palette.

    Returns
    -------
    str
        Path of the GIF, next to the MP4.
    """
    name = os.path.splitext(mp4)[0]
    gif = name + ".gif"
    pal = name + "_palette.png"
    vf = "fps=%d,scale=%d:-1:flags=lanczos" % (fps, width)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf",
                    vf + ",palettegen=max_colors=%d:stats_mode=diff"
                    % colors, pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-i",
                    pal, "-lavfi",
                    vf + " [x]; [x][1:v] paletteuse=dither=bayer:"
                    "bayer_scale=4:diff_mode=rectangle", gif], check=True)
    os.remove(pal)
    return gif


class CrownInset:
    """Head and chest seen from the crown, with the water line."""

    def __init__(self, ax, body):
        self.ax = ax
        self.body = body
        ax.set_xlim(-0.42, 0.42)
        ax.set_ylim(-0.38, 0.30)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)],
                             fc="#06131d", ec="none", zorder=0))
        self.sky = Polygon(np.zeros((3, 2)), fc="#c5d8e6", ec="none",
                           zorder=1, alpha=0.9)
        ax.add_patch(self.sky)
        self.chest = Ellipse((0, 0), 0.35, 0.24, fc="#14181c",
                             ec="#000", lw=0.6, zorder=3)
        self.lobe_l = Ellipse((0, 0), 0.1, 0.1, fc="#2f3439", ec="#000",
                              lw=0.5, zorder=4)
        self.lobe_r = Ellipse((0, 0), 0.1, 0.1, fc="#2f3439", ec="#000",
                              lw=0.5, zorder=4)
        self.tank = Ellipse((0, 0), 0.184, 0.184, fc="#b9c2c9",
                            ec="#59636b", lw=0.6, zorder=2)
        self.head = Ellipse((0, 0), 0.16, 0.20, fc="#16191d", ec="#000",
                            lw=0.6, zorder=6)
        self.face = Ellipse((0, 0), 0.12, 0.07, fc="#c8977a", ec="none",
                            zorder=7)
        self.nose, = ax.plot([], [], color="#9c6f56", lw=2.5, zorder=8,
                             solid_capstyle="round")
        for p in (self.chest, self.lobe_l, self.lobe_r, self.tank,
                  self.head, self.face):
            ax.add_patch(p)
        self.water = Polygon(np.zeros((3, 2)), fc="#0d5a72", ec="none",
                             alpha=0.55, zorder=9)
        ax.add_patch(self.water)
        self.line, = ax.plot([], [], color="#e9fbff", lw=1.2, zorder=10)
        self.mark = ax.scatter([0], [0], s=30, c=OK, zorder=11,
                               edgecolors="white", lw=0.5)
        ax.text(0.0, 0.27, "view from the crown", color=DIM, ha="center",
                va="top", fontsize=7.5, zorder=12)

    def update(self, q, eta_a, slope_a, clear):
        b = self.body
        sel = b.head == 1
        hx = float(np.mean(b.xi[sel]))
        hz = float(np.mean(b.z0[sel]))
        ya, za = point_height(q[None], b, hx, 0.0, hz, True)
        yc, zc = point_height(q[None], b, -0.15 - b.cg_shift[0], 0.0,
                              -b.cg_shift[1], False)
        ref = eta_a
        alpha = q[2] + q[3]
        ang = np.degrees(alpha)
        self.head.set_center((ya[0], za[0] - ref))
        self.head.set_angle(ang)
        nx, nz = -np.sin(alpha), np.cos(alpha)
        self.face.set_center((ya[0] + 0.065 * nx, za[0] - ref + 0.065 * nz))
        self.face.set_angle(ang)
        tip = (ya[0] + 0.125 * nx, za[0] - ref + 0.125 * nz)
        base = (ya[0] + 0.095 * nx, za[0] - ref + 0.095 * nz)
        self.nose.set_data([base[0], tip[0]], [base[1], tip[1]])
        self.chest.set_center((yc[0], zc[0] - ref))
        self.chest.set_angle(np.degrees(q[2]))
        cn, sn = np.cos(q[2]), np.sin(q[2])
        for p, sgn in ((self.lobe_l, -1), (self.lobe_r, 1)):
            ly, lz = sgn * 0.21, 0.02
            p.set_center((yc[0] + ly * cn - lz * sn,
                          zc[0] - ref + ly * sn + lz * cn))
        ty, tz = 0.0, -0.21
        self.tank.set_center((yc[0] + ty * cn - tz * sn,
                              zc[0] - ref + ty * sn + tz * cn))
        yy = np.array([-1.0, 1.0])
        zz = slope_a * yy
        self.line.set_data(yy, zz)
        self.water.set_xy([(-1, zz[0]), (1, zz[1]), (1, -1), (-1, -1)])
        self.sky.set_xy([(-1, zz[0]), (1, zz[1]), (1, 1), (-1, 1)])
        self.mark.set_offsets([(tip[0] - 0.01 * nx, tip[1] - 0.02 * nz)])
        self.mark.set_color(OK if clear >= 0 else BAD)
