"""Event statistics, information measures, extremes and sensitivity.

Awash events are the maximal intervals with negative airway clearance;
dry intervals are the gaps between them.

Information available for timing breaths
----------------------------------------
For a positive random variable with a fixed mean ``m``, the exponential
law has the largest differential entropy, ``1 + ln m``. Its deviation
from the observed dry-interval law is the Kullback-Leibler divergence

    D = (1 + ln m) - h_obs,

which is zero for a memoryless (Poisson) sequence of awash events and
positive when dry windows are regular enough to be anticipated. ``D`` is
reported in bits.
"""

import numpy as np
from scipy import stats


def events(clear, dt, first=0):
    """Durations of awash events and of dry intervals.

    Parameters
    ----------
    clear : ndarray (R, N)
    dt : float
    first : int
        First retained sample.

    Returns
    -------
    dict
        ``wet`` and ``dry`` durations (s, pooled), ``count`` per series,
        ``longest`` awash duration per series, ``p_awash``, ``rate``
        (entries per second).
    """
    c = clear[:, first:]
    wet_all, dry_all, counts, longest = [], [], [], []
    for row in c:
        w = row < 0.0
        d = np.diff(w.astype(int))
        starts = np.flatnonzero(d == 1) + 1
        ends = np.flatnonzero(d == -1) + 1
        if w[0]:
            starts = np.r_[0, starts]
        if w[-1]:
            ends = np.r_[ends, w.size]
        wet = (ends - starts) * dt
        # Only complete events and complete dry gaps enter the samples.
        full = (starts > 0) & (ends < w.size)
        wet_all.append(wet[full])
        gaps = (starts[1:] - ends[:-1]) * dt
        dry_all.append(gaps)
        counts.append(int(np.sum(starts > 0)))
        longest.append(wet.max() if wet.size else 0.0)
    tot = c.shape[1] * dt
    return dict(wet=np.concatenate(wet_all), dry=np.concatenate(dry_all),
                count=np.array(counts), longest=np.array(longest),
                p_awash=float((c < 0).mean()),
                rate=float(np.sum(counts) / (c.shape[0] * tot)))


def fano(clear, dt, window=30.0, first=0):
    """Fano factor of awash-event counts in windows of fixed length."""
    c = clear[:, first:]
    w = c < 0.0
    ent = (w[:, 1:] & ~w[:, :-1]).astype(int)
    n = int(round(window / dt))
    nb = ent.shape[1] // n
    if nb < 2:
        return np.nan
    counts = ent[:, :nb * n].reshape(ent.shape[0], nb, n).sum(-1).ravel()
    m = counts.mean()
    return float(counts.var(ddof=1) / m) if m > 0 else np.nan


def histogram_entropy(x, width):
    """Differential entropy (nats) from a histogram of bin ``width``.

    Durations measured on a sampling grid are discrete, so the bin width
    is tied to the sampling interval rather than to spacings between
    sorted values, which vanish for tied samples.
    """
    x = np.asarray(x, float)
    edges = np.arange(0.0, x.max() + 2 * width, width)
    cnt, _ = np.histogram(x, edges)
    p = cnt[cnt > 0] / x.size
    return float(-np.sum(p * np.log(p / width)))


def timing_information(dry, width=0.2):
    """KL divergence (bits) of dry intervals from the exponential law.

    Parameters
    ----------
    dry : array_like
        Dry-interval durations (s).
    width : float
        Histogram bin width (s), a multiple of the sampling interval.
    """
    dry = np.asarray(dry, float)
    if dry.size < 30:
        return np.nan
    h_exp = 1.0 + np.log(dry.mean())
    return float((h_exp - histogram_entropy(dry, width)) / np.log(2.0))


def gev_fit(x):
    """Generalized extreme-value fit (shape, location, scale) of maxima.

    The shape follows the hydrology sign convention ``xi = -c`` of SciPy,
    so ``xi > 0`` is a heavy (Frechet) upper tail.
    """
    x = np.asarray(x, float)
    c, loc, scale = stats.genextreme.fit(x)
    return -c, loc, scale


def sobol_jansen(f, bounds, n=512, seed=0, log=None):
    """First-order and total Sobol indices (Saltelli sampling, Jansen).

    Parameters
    ----------
    f : callable
        Maps an (N, d) array of inputs to (N,) or (N, k) outputs.
    bounds : list of (lo, hi)
        Uniform input ranges.
    n : int
        Base sample size; the cost is ``n (d + 2)`` evaluations.
    log : list of bool, optional
        Sample the corresponding input uniformly in the logarithm.

    Returns
    -------
    S1, ST : ndarray (d,) or (d, k)
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)
    log = log or [False] * d
    lo = np.array([np.log(b[0]) if lg else b[0] for b, lg in zip(bounds, log)])
    hi = np.array([np.log(b[1]) if lg else b[1] for b, lg in zip(bounds, log)])

    def scale(u):
        x = lo + u * (hi - lo)
        for j, lg in enumerate(log):
            if lg:
                x[:, j] = np.exp(x[:, j])
        return x

    A = rng.random((n, d))
    B = rng.random((n, d))
    fA = np.asarray(f(scale(A)), float)
    fB = np.asarray(f(scale(B)), float)
    var = np.var(np.concatenate([fA, fB]), axis=0, ddof=1)
    S1, ST = [], []
    for j in range(d):
        ABj = A.copy()
        ABj[:, j] = B[:, j]
        fAB = np.asarray(f(scale(ABj)), float)
        S1.append(np.mean(fB * (fAB - fA), axis=0) / var)
        ST.append(0.5 * np.mean((fA - fAB) ** 2, axis=0) / var)
    return np.array(S1), np.array(ST)
