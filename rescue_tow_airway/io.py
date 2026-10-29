"""Output paths, cache files, CSV tables and figure saving."""

import csv
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT = os.path.join(ROOT, "outputs")
DIRS = {k: os.path.join(OUT, k) for k in
        ("figures", "animations", "data", "reports", "cache")}


def path(kind, name):
    """Absolute path of an output file, creating its folder."""
    d = DIRS[kind]
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def save_cache(name, **arrays):
    """Store arrays in ``outputs/cache/<name>.npz``."""
    np.savez_compressed(path("cache", name + ".npz"), **arrays)


def load_cache(name):
    """Load ``outputs/cache/<name>.npz`` as a dict of arrays."""
    fn = path("cache", name + ".npz")
    if not os.path.exists(fn):
        raise FileNotFoundError(fn + " is missing; run the compute scripts "
                                "first (scripts/run_all.py)")
    with np.load(fn, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def has_cache(name):
    """True if the cache file exists."""
    return os.path.exists(path("cache", name + ".npz"))


def write_csv(name, header, rows):
    """Write a CSV table to ``outputs/data``."""
    with open(path("data", name + ".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([("%.6g" % v) if isinstance(v, (float, np.floating))
                        else v for v in r])


def save_figure(fig, name):
    """Save a figure as PDF and as a 600 dpi PNG."""
    fig.savefig(path("figures", name + ".pdf"))
    fig.savefig(path("figures", name + ".png"), dpi=600)


def write_report(name, lines):
    """Write a plain-text report to ``outputs/reports``."""
    with open(path("reports", name + ".txt"), "w") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")
