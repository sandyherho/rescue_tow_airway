"""Figure 9: a decision chart for rescuers.

For each open-water environment at a 0.5 m/s tow, the probability that
a rescue breath meets water when towing with the waves, across them, or
into them (M0 in the cell color and upper number, M1 below when
simulated), and the reduction in failed breaths when the rescuer waits until
the airway is clear and rising (M1).
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
from rescue_tow_airway.plotting import (C, legend_below,  # noqa: E402
                                        panel_label, style)

ENVS = [("E1", "E1\nswell"), ("E2", "E2\nwind chop"),
        ("E3", "E3\nmixed"), ("E7", "E7\nconfused"),
        ("E4m", "E4−\nopposing\ncurrent"),
        ("E4p", "E4+\nfollowing\ncurrent")]
HEADS = [(0.0, "with the waves"), (90.0, "across"), (180.0, "into the waves")]


def lookup(m, speed, heading):
    i = int(np.argmin(np.abs(m["speeds"] - speed)))
    d = np.abs((m["headings"] - heading + 180) % 360 - 180)
    j = int(np.argmin(d))
    if d[j] > 1e-6:
        return np.nan
    return m["p_breath"][i, j]


def main():
    style()
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(6.6, 4.6),
                                  gridspec_kw=dict(height_ratios=[1.6, 1]))
    fig.subplots_adjust(hspace=0.55)
    norm = LogNorm(1e-3, 1)
    grid = np.full((len(HEADS), len(ENVS)), np.nan)
    rows = []
    for c, (tag, lab) in enumerate(ENVS):
        m0 = load_cache("m0_map_" + tag)
        m1 = load_cache("m1_map_" + tag) if has_cache("m1_map_" + tag) \
            else None
        for r, (h, hl) in enumerate(HEADS):
            p0 = lookup(m0, 0.5, h)
            grid[r, c] = p0
            p1 = lookup(m1, 0.5, h) if m1 is not None else np.nan
            rows.append([tag, h, p0, p1])
            txt = "%.3f" % p0 if p0 >= 1e-3 else "<0.001"
            if np.isfinite(p1):
                txt += "\n(%s)" % ("%.3f" % p1 if p1 >= 1e-3 else "<0.001")
            dark = np.clip(p0, 1e-3, 1) > 0.05
            ax.text(c, r, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if dark else "black")
    ax.imshow(np.clip(grid, 1.01e-3, 1), norm=norm, cmap="magma_r",
              aspect="auto")
    ax.set_xticks(range(len(ENVS)))
    ax.set_xticklabels([e[1] for e in ENVS], fontsize=6.5)
    ax.set_yticks(range(len(HEADS)))
    ax.set_yticklabels([h[1] for h in HEADS])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    panel_label(ax, "a", x=-0.2)

    envs_b = [("E2", "E2"), ("E7", "E7"), ("E3", "E3"), ("E4m", "E4−")]
    x = np.arange(len(envs_b))
    for k, (sc, col) in enumerate((("tow", C["blue"]),
                                   ("hold", C["orange"]))):
        vals = []
        for tag, _ in envs_b:
            d = load_cache("breath_%s_%s" % (tag, sc))
            f0, f1 = d["fail_fixed"], d["fail_clear"]
            vals.append(100 * (1 - f1 / f0) if f0 >= 0.01 else np.nan)
            rows.append([tag, sc, f0, f1])
        axb.bar(x + (k - 0.5) * 0.36, vals, 0.36, color=col,
                label="towing into the waves" if sc == "tow" else
                "stopped, head to the waves")
    for i, (tag, _) in enumerate(envs_b):
        d = load_cache("breath_%s_tow" % tag)
        if d["fail_fixed"] < 0.01:
            axb.text(i, 3, "too few\nfailures", ha="center", fontsize=6,
                     color=C["grey"])
    axb.axhline(0, color="k", lw=0.5)
    axb.set_xticks(x)
    axb.set_xticklabels([e[1] for e in envs_b])
    axb.set_ylabel("fewer failed breaths (%)")
    panel_label(axb, "b", x=-0.2)
    h, lab = axb.get_legend_handles_labels()
    legend_below(fig, [("(b)", h, lab)])
    save_figure(fig, "fig09_decision")
    write_csv("fig09_decision", ["env", "heading_or_scenario", "a", "b"],
              rows)


if __name__ == "__main__":
    main()
