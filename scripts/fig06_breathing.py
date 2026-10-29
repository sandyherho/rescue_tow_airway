"""Figure 6: timing rescue breaths, clustering and the longest immersion."""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.breathing import STRATEGIES, evaluate  # noqa: E402
from rescue_tow_airway.io import load_cache, save_figure, write_csv  # noqa
from rescue_tow_airway.plotting import (C, ENV_COLORS, legend_below,  # noqa
                                        line, panel_label, patch, style)

ENVS = ["E2", "E7", "E3", "E4m"]
NAMES = {"E4m": "E4-"}
SCOL = {"fixed": C["grey"], "clear": C["sky"], "crest": C["blue"],
        "oracle": C["green"]}
SLAB = {"fixed": "every 5 s", "clear": "wait: airway clear",
        "crest": "wait: crest passed", "oracle": "best possible"}


def main():
    style()
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45)
    rows = []

    # (a) a 40 s excerpt with both strategies.
    ax = fig.add_subplot(gs[0, :2])
    d = load_cache("breath_E2_tow")
    t, c, w = d["sample_t"], d["sample_clear"], d["sample_water"]
    sel = t <= 40.0
    ax.plot(t[sel], 100 * c[sel], color="k", lw=0.8)
    ax.fill_between(t[sel], 100 * c[sel], 0, where=c[sel] < 0,
                    color=C["red"], alpha=0.6, lw=0)
    ax.axhline(0, color="k", lw=0.4, ls="--")
    dt = t[1] - t[0]
    for k, (s, yy, mk) in enumerate((("fixed", 26.0, "v"),
                                     ("clear", 31.0, "D"))):
        r = evaluate(c[None], w[None], dt, s, rng=np.random.default_rng(3))
        tb = r["times"][0]
        ok = r["ok"][0]
        tb_t = t[tb]
        keep = tb_t <= 40.0
        ax.scatter(tb_t[keep], np.full(keep.sum(), yy), marker=mk,
                   s=22 if mk == "v" else 14,
                   c=[C["green"] if o else C["red"] for o in ok[keep]],
                   zorder=4)
    ax.set_xlim(0, 40)
    ax.set_ylim(-20, 38)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("airway clearance (cm)")
    panel_label(ax, "a", x=-0.09)

    # (b) strategies across environments.
    ax = fig.add_subplot(gs[0, 2])
    xs = np.arange(len(ENVS))
    wdt = 0.2
    for k, s in enumerate(STRATEGIES):
        vals = []
        for e in ENVS:
            d = load_cache("breath_%s_tow" % e)
            vals.append(d["fail_" + s])
            rows.append([e, "tow", s, d["fail_" + s], d["interval_" + s],
                         d["wait_" + s]])
            d2 = load_cache("breath_%s_hold" % e)
            rows.append([e, "hold", s, d2["fail_" + s], d2["interval_" + s],
                         d2["wait_" + s]])
        ax.bar(xs + (k - 1.5) * wdt, vals, wdt, color=SCOL[s],
               label=SLAB[s])
    ax.set_xticks(xs)
    ax.set_xticklabels([NAMES.get(e, e) for e in ENVS])
    ax.set_ylabel("P(breath meets water)")
    ax.set_ylim(0, 1.0)
    panel_label(ax, "b", x=-0.3)

    # (c) Fano factor against window length.
    ax = fig.add_subplot(gs[1, 0])
    for e in ENVS[:4]:
        d = load_cache("breath_%s_tow" % e)
        ax.plot(d["windows"], d["fano"], "o-", ms=3,
                color=ENV_COLORS[NAMES.get(e, e)], label=NAMES.get(e, e))
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.text(58, 1.05, "Poisson", ha="right", fontsize=6)
    ax.set_xlabel("counting window (s)")
    ax.set_ylabel("Fano factor of submersions")
    panel_label(ax, "c", x=-0.3)

    # (d) timing information against oracle gain.
    ax = fig.add_subplot(gs[1, 1])
    for e in ENVS[:4]:
        for sc, mk in (("tow", "o"), ("hold", "s")):
            d = load_cache("breath_%s_%s" % (e, sc))
            if d["fail_fixed"] <= 0:
                continue
            gain = d["fail_fixed"] / max(d["fail_clear"], 1e-3)
            ax.scatter(d["info"], gain, marker=mk, s=22,
                       color=ENV_COLORS[NAMES.get(e, e)])
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set_xlabel("timing information D (bits)")
    ax.set_ylabel("failures: rhythm / clear cue")
    panel_label(ax, "d", x=-0.3)

    # (e) longest immersion per 5 min tow.
    ax = fig.add_subplot(gs[1, 2])
    for e in ENVS[:4]:
        d = load_cache("breath_%s_tow" % e)
        x = np.sort(d["longest"])
        if x.max() <= 0:
            continue
        p = np.arange(1, x.size + 1) / (x.size + 1)
        ax.plot(x, p, drawstyle="steps-post",
                color=ENV_COLORS[NAMES.get(e, e)], label=NAMES.get(e, e))
    ax.set_xlabel("longest immersion per 270 s (s)")
    ax.set_ylabel("cumulative probability")
    panel_label(ax, "e", x=-0.3)
    names = [NAMES.get(e, e) for e in ENVS]
    legend_below(fig, [
        ("(a)", [line("k", "none", "v"), line("k", "none", "D"),
                 patch(C["green"]), patch(C["red"])],
         ["breath every 5 s", "breath when airway clear", "dry",
          "wet (breath or airway)"]),
        ("(b)", [patch(SCOL[s]) for s in STRATEGIES],
         [SLAB[s] for s in STRATEGIES]),
        ("(c)\u2013(e)", [line(ENV_COLORS[n]) for n in names]
         + [line("k", "none", "o"), line("k", "none", "s")],
         names + ["(d) towing", "(d) stopped"])])
    save_figure(fig, "fig06_breathing")
    write_csv("fig06_strategies", ["env", "scenario", "strategy",
                                   "p_fail", "interval_s", "wait_s"], rows)


if __name__ == "__main__":
    main()
