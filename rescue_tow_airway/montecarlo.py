"""Monte Carlo driver for model M1."""

import numpy as np

from .dynamics import equilibrium, simulate
from .waves import field_along_body


def run_m1(body, sea, speed, heading, n_real=32, duration=300.0, dt=0.04,
           spinup=30.0, seed=0, batch=16, origin=(0.0, 0.0), q0=None,
           n_omega=200, keep_states=False):
    """Simulate many independent tows in one sea state.

    Parameters
    ----------
    body : Body
    sea : SeaState
    speed : float
        Tow speed through the water (m/s).
    heading : float
        Tow direction relative to the sea-state frame (rad).
    n_real : int
        Number of independent realizations.
    duration : float
        Length of each tow, including spin-up (s).
    dt : float
        Integration step (s).
    spinup : float
        Initial interval excluded from statistics (s).
    seed : int
        Base random seed. Realization ``i`` uses ``seed * 100003 + i``.
    origin : tuple
        Body position at t = 0 (m).
    q0 : ndarray, optional
        Initial coordinates.
    keep_states : bool
        Return the full state history.

    Returns
    -------
    dict
        ``t``, ``clear`` (R, N), ``water`` (R, N), ``q`` (R, N, 4) when
        requested, plus ``dt`` and the index of the first retained step.
    """
    nsteps = int(round(duration / dt))
    th = np.arange(2 * nsteps + 1) * 0.5 * dt
    ex, ey = np.cos(heading), np.sin(heading)
    vel = (speed * ex + sea.current[0], speed * ey + sea.current[1])
    if q0 is None:
        q0 = equilibrium(body, speed)
    clears, waters, states = [], [], []
    for i0 in range(0, n_real, batch):
        idx = range(i0, min(n_real, i0 + batch))
        fl = [[], [], [], []]
        for i in idx:
            rng = np.random.default_rng(seed * 100003 + i)
            waves = sea.realize(rng, n_omega)
            f = field_along_body(waves, body.strip_xi, th, origin, vel,
                                 heading)
            for k in range(4):
                fl[k].append(f[k])
        fields = tuple(np.stack(v) for v in fl)
        out = simulate(body, fields, dt, speed, q0)
        clears.append(out["clear"])
        waters.append(out["water"])
        if keep_states:
            states.append(out["q"])
    res = dict(t=out["t"], dt=dt, clear=np.concatenate(clears),
               velocity=np.array(vel), origin=np.array(origin),
               water=np.concatenate(waters),
               first=int(round(spinup / dt)))
    if keep_states:
        res["q"] = np.concatenate(states)
    return res


def single_run(body, sea, speed, heading, seed, duration=60.0, dt=0.04,
               origin=(0.0, 0.0), q0=None, n_omega=200):
    """One tow with its plane-wave realization (for rendering).

    Uses the same random stream as realization 0 of :func:`run_m1` with
    the same seed, so the rendered sea is the sea the body felt.

    Returns
    -------
    dict
        Simulation output for one realization plus ``waves``,
        ``velocity`` and ``origin``.
    """
    nsteps = int(round(duration / dt))
    th = np.arange(2 * nsteps + 1) * 0.5 * dt
    ex, ey = np.cos(heading), np.sin(heading)
    vel = (speed * ex + sea.current[0], speed * ey + sea.current[1])
    rng = np.random.default_rng(seed * 100003)
    waves = sea.realize(rng, n_omega)
    f = field_along_body(waves, body.strip_xi, th, origin, vel, heading)
    fields = tuple(v[None] for v in f)
    if q0 is None:
        q0 = equilibrium(body, speed)
    out = simulate(body, fields, dt, speed, q0)
    out.update(waves=waves, velocity=np.array(vel), origin=np.array(origin),
               heading=heading, dt=dt)
    return out
