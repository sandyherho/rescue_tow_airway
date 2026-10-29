"""Figure style shared by all static figures."""

import matplotlib as mpl

# Colorblind-safe qualitative palette (Okabe-Ito order).
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", black="#000000",
         grey="#7f7f7f")

ENV_COLORS = {"E1": C["blue"], "E2": C["red"], "E3": C["purple"],
              "E7": C["orange"], "E4-": C["green"], "E4+": C["sky"],
              "E5": C["grey"], "E6": C["black"]}


def style():
    """Apply the journal figure style."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })


def panel_label(ax, s, x=-0.14, y=1.04):
    """Bold panel label, such as (a), in axes coordinates."""
    ax.text(x, y, "(%s)" % s, transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="bottom", ha="left")


def line(color="k", ls="-", marker=None, mfc=None, lw=1.1):
    """Proxy line handle for a legend."""
    from matplotlib.lines import Line2D
    return Line2D([], [], color=color, ls=ls, lw=lw if ls != "none" else 0,
                  marker=marker, ms=4, mfc=mfc or color)


def patch(color):
    """Proxy patch handle for a legend."""
    from matplotlib.patches import Patch
    return Patch(fc=color, ec="none")


def legend_below(fig, rows, gap=0.012, row_gap=0.004):
    """Stack one-line legends centered below all panels of a figure.

    Parameters
    ----------
    fig : matplotlib Figure
    rows : list of tuple
        ``(title, handles, labels)`` for each row; ``title`` names the
        panels the row applies to, such as ``"(a)"``, or is ``None``.
    gap : float
        Space between the lowest panel element and the first row, in
        figure fraction.
    row_gap : float
        Space between rows, in figure fraction.
    """
    from matplotlib.lines import Line2D
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y = min(ax.get_tightbbox(rend).transformed(inv).y0 for ax in fig.axes)
    y -= gap
    for title, handles, labels in rows:
        handles, labels = list(handles), list(labels)
        if title:
            handles.insert(0, Line2D([], [], lw=0))
            labels.insert(0, title)
        leg = fig.legend(handles, labels, loc="upper center",
                         bbox_to_anchor=(0.5, y), ncol=len(handles),
                         frameon=False, fontsize=7, handlelength=1.8,
                         columnspacing=1.4, handletextpad=0.5,
                         borderaxespad=0.0)
        if title:
            leg.get_texts()[0].set_fontweight("bold")
        fig.canvas.draw()
        y = leg.get_window_extent(rend).transformed(inv).y0 - row_gap
