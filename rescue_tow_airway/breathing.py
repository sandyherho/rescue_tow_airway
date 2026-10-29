"""Rescue-breath timing strategies evaluated on airway-clearance series.

A breath of duration ``tau`` fails when the airway is awash (clearance
below zero) at any instant during the breath. Breaths are due every
``period`` seconds after the previous breath.

Strategies
----------
fixed
    Breathe exactly when due (the trained rhythm).
clear
    From the due time, wait until the airway is visibly clear by at least
    ``r_c`` and the clearance is rising, then breathe. Causal.
crest
    From the due time, wait until a wave crest has just passed the face
    (the water level at the airway has just reached a local maximum and
    is falling) with the airway clear by at least ``r_c``. Causal: it
    uses only what the rescuer has already seen.
oracle
    Within the waiting window, choose the start that maximizes the
    minimum clearance during the breath. Not causal; an upper bound.

Every strategy except ``fixed`` gives up waiting after ``max_wait``
seconds and breathes regardless.
"""

import numpy as np


def _window_min(clear, tau_n):
    """Minimum of each series over the window [n, n + tau_n]."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([clear, np.repeat(clear[:, -1:], tau_n, 1)], 1)
    return sliding_window_view(pad, tau_n + 1, axis=1).min(-1)


def evaluate(clear, water, dt, strategy="fixed", period=5.0, tau=1.0,
             max_wait=3.0, r_c=0.03, first=0, rng=None):
    """Simulate a breathing strategy on clearance series.

    Parameters
    ----------
    clear, water : ndarray (R, N)
        Airway clearance and water level at the airway.
    dt : float
        Sampling interval (s).
    strategy : str
        ``"fixed"``, ``"clear"``, ``"crest"`` or ``"oracle"``.
    period, tau, max_wait, r_c : float
        Breath period, breath duration, maximum wait, clearance cue (s, m).
    first : int
        First usable sample (after spin-up).
    rng : numpy.random.Generator, optional
        Random start offset of the first breath in each series.

    Returns
    -------
    dict
        ``fail`` failure fraction, ``n`` breaths, ``interval`` mean time
        between breaths (s), ``wait`` mean wait after the due time (s),
        ``times`` list of breath start indices per series, ``ok`` list of
        success flags per series.
    """
    rng = rng or np.random.default_rng(0)
    nr, nt = clear.shape
    tau_n = max(1, int(round(tau / dt)))
    per_n = int(round(period / dt))
    wait_n = int(round(max_wait / dt))
    wmin = _window_min(clear, tau_n)
    fails, total, waits, gaps = 0, 0, [], []
    times, oks = [], []
    for i in range(nr):
        c, wl, wm = clear[i], water[i], wmin[i]
        due = first + int(rng.integers(0, per_n))
        last = None
        tl, ol = [], []
        while due + wait_n + tau_n < nt:
            n = due
            if strategy == "fixed":
                n = due
            elif strategy == "oracle":
                seg = wm[due:due + wait_n + 1]
                n = due + int(np.argmax(seg))
            else:
                n = due + wait_n
                for m in range(due, due + wait_n + 1):
                    if m < 2:
                        continue
                    if strategy == "clear":
                        cue = c[m] > r_c and c[m] > c[m - 1]
                    elif strategy == "crest":
                        cue = (c[m] > r_c and wl[m - 1] >= wl[m - 2]
                               and wl[m] < wl[m - 1])
                    else:
                        raise ValueError("unknown strategy " + strategy)
                    if cue:
                        n = m
                        break
            ok = wm[n] >= 0.0
            fails += int(not ok)
            total += 1
            waits.append((n - due) * dt)
            if last is not None:
                gaps.append((n - last) * dt)
            last = n
            tl.append(n)
            ol.append(ok)
            due = n + per_n
        times.append(np.array(tl))
        oks.append(np.array(ol, bool))
    return dict(fail=fails / max(total, 1), n=total,
                interval=float(np.mean(gaps)) if gaps else np.nan,
                wait=float(np.mean(waits)) if waits else np.nan,
                times=times, ok=oks)


STRATEGIES = ("fixed", "clear", "crest", "oracle")
