"""Figure 8: Sobol indices and the robustness of the heading advice."""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.io import load_cache, save_figure, write_csv  # noqa
from rescue_tow_airway.plotting import (C, legend_below,  # noqa: E402
                                        panel_label, style)

LABELS = {"c_a": "added mass $c_a$", "beta_r": "radiation $\\beta_r$",
          "C_D": "heave drag $C_D$", "beta_44": "roll damping $\\beta_{44}$",
          "c_44": "roll drag $c_{44}$", "rho_body": "body density",
          "k_neck": "neck stiffness"}


def main():
    style()
    s = load_cache("sensitivity")
    names = [str(n) for n in s["names"]]
    s1, st = s["s1"], s["st"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7),
                            gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    fig.subplots_adjust(wspace=0.75)
    y = np.arange(len(names))
    for k, j in enumerate((0, 2)):
        ax = axs[k]
        ax.barh(y + 0.18, np.clip(s1[:, j], 0, 1), 0.34, color=C["blue"],
                label="first order")
        ax.barh(y - 0.18, np.clip(st[:, j], 0, 1), 0.34, color=C["orange"],
                label="total")
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[n] for n in names], fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Sobol index")
        panel_label(ax, "ab"[k], x=-0.75)
    ax = axs[2]
    yv = s["sample_y"]
    ax.hist(yv[:, 2], bins=30, color=C["purple"], alpha=0.85)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("log10 [P(into) / P(across)]")
    ax.set_ylabel("samples")
    panel_label(ax, "c", x=-0.3)
    h, lab = axs[0].get_legend_handles_labels()
    legend_below(fig, [("(a), (b)", h, ["first-order Sobol index",
                                        "total Sobol index"])])
    save_figure(fig, "fig08_sensitivity")
    write_csv("fig08_sobol", ["input", "S1_into", "S1_across", "S1_ratio",
                              "ST_into", "ST_across", "ST_ratio"],
              [[n] + list(s1[i]) + list(st[i]) for i, n in
               enumerate(names)])


if __name__ == "__main__":
    main()
