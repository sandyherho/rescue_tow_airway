"""Idealized random sea states as superpositions of plane waves.

Every sea state in the library, including waves on a current and waves
partially reflected from a boat hull, is represented as a finite set of
plane waves

    eta(x, t) = Re sum_m c_m exp(i (k_m . x - omega_m t)),

where ``c_m`` is a complex amplitude that carries the random phase.
Each plane wave also carries its intrinsic frequency ``sigma_m``, the
frequency seen in a frame moving with the current, which sets the
vertical particle velocity ``-i sigma_m eta`` and acceleration
``-sigma_m**2 eta`` at the free surface.

Two outputs are produced from a ``SeaState``:

* a deterministic spectral grid of independent wave groups, used by the
  linear model M0 (a group holds the incident wave and, when a hull is
  present, its phase-locked reflection);
* random realizations for the nonlinear model M1 (random phases,
  jittered frequencies, and one direction per frequency drawn from the
  spreading function).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

G = 9.81
RHO = 1025.0


# ---------------------------------------------------------------------------
# Dispersion and spectra
# ---------------------------------------------------------------------------

def wavenumber(omega, depth=None):
    """Solve the linear dispersion relation omega**2 = g k tanh(k h).

    Parameters
    ----------
    omega : array_like
        Intrinsic angular frequency (rad/s).
    depth : float or None
        Water depth (m). ``None`` means deep water.

    Returns
    -------
    ndarray
        Wavenumber (rad/m).
    """
    omega = np.asarray(omega, dtype=float)
    k = omega ** 2 / G
    if depth is None:
        return k
    h = float(depth)
    # Newton iteration from the Eckart approximation.
    k = omega ** 2 / G / np.sqrt(np.tanh(omega ** 2 * h / G))
    for _ in range(50):
        th = np.tanh(k * h)
        f = G * k * th - omega ** 2
        df = G * th + G * k * h * (1.0 - th ** 2)
        k = k - f / df
    return k


def group_velocity(k, depth=None):
    """Group velocity of linear gravity waves (m/s)."""
    k = np.asarray(k, dtype=float)
    if depth is None:
        return 0.5 * np.sqrt(G / k)
    h = float(depth)
    kh = k * h
    c = np.sqrt(G / k * np.tanh(kh))
    return 0.5 * c * (1.0 + 2.0 * kh / np.sinh(2.0 * kh))


def jonswap(omega, hs, tp, gamma=3.3):
    """JONSWAP frequency spectrum normalized to a significant wave height.

    Parameters
    ----------
    omega : array_like
        Angular frequency (rad/s).
    hs : float
        Significant wave height ``4 sqrt(m0)`` (m).
    tp : float
        Peak period (s).
    gamma : float
        Peak enhancement factor.

    Returns
    -------
    ndarray
        Spectral density S(omega) (m^2 s).
    """
    omega = np.asarray(omega, dtype=float)
    wp = 2.0 * np.pi / tp
    sig = np.where(omega <= wp, 0.07, 0.09)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        shape = omega ** -5.0 * np.exp(-1.25 * (wp / omega) ** 4)
        peak = gamma ** np.exp(-((omega - wp) ** 2)
                               / (2.0 * sig ** 2 * wp ** 2))
        s = np.where(omega > 0, shape * peak, 0.0)
    # Normalize numerically on a fine reference grid.
    wref = np.linspace(0.2 * wp, 8.0 * wp, 8000)
    sref = (wref ** -5.0 * np.exp(-1.25 * (wp / wref) ** 4)
            * gamma ** np.exp(-((wref - wp) ** 2)
                              / (2.0 * np.where(wref <= wp, 0.07, 0.09) ** 2
                                 * wp ** 2)))
    m0 = np.trapezoid(sref, wref)
    return s * (hs ** 2 / 16.0) / m0


def spreading_pdf(theta, theta0, s):
    """Mitsuyasu cos^(2s) directional spreading, normalized over 2 pi."""
    from scipy.special import gammaln
    d = np.cos(0.5 * (np.asarray(theta) - theta0)) ** 2
    lognorm = (gammaln(s + 1.0) - gammaln(s + 0.5)
               - 0.5 * np.log(np.pi) - np.log(2.0))
    return np.exp(lognorm) * d ** s


# ---------------------------------------------------------------------------
# Sea-state description
# ---------------------------------------------------------------------------

@dataclass
class WaveSystem:
    """One wave system (a swell or a wind sea).

    Parameters
    ----------
    hs : float
        Significant wave height (m), in deep water before any transform.
    tp : float
        Peak period (s).
    gamma : float
        JONSWAP peak enhancement factor.
    theta0 : float
        Mean propagation direction (rad), measured from the +x axis.
    s : float or None
        Spreading exponent of cos^(2s). ``None`` gives long-crested waves.
    """

    hs: float
    tp: float
    gamma: float = 3.3
    theta0: float = 0.0
    s: Optional[float] = None


@dataclass
class Wall:
    """A vertical reflecting boundary (an idealized boat hull).

    The wall is the line ``n . x = n . x0``; ``normal`` points from the
    water toward the wall. ``refl`` is the amplitude reflection
    coefficient.
    """

    x0: Tuple[float, float] = (0.0, 0.0)
    normal: Tuple[float, float] = (1.0, 0.0)
    refl: float = 0.9


@dataclass
class SeaState:
    """An idealized sea state.

    Parameters
    ----------
    systems : list of WaveSystem
        Wave systems, specified in deep water.
    depth : float or None
        Local depth (m). With a finite depth each deep-water component is
        shoaled at normal incidence (energy-flux conservation), and the
        total significant height is capped at ``0.6 depth``.
    current : tuple of float
        Uniform surface current (m/s). Deep-water only. Components are
        transformed by wave-action conservation; blocked components are
        removed.
    wall : Wall or None
        Optional reflecting hull.
    name : str
        Short label.
    """

    systems: List[WaveSystem]
    depth: Optional[float] = None
    current: Tuple[float, float] = (0.0, 0.0)
    wall: Optional[Wall] = None
    name: str = ""
    label: str = ""
    extra: dict = field(default_factory=dict)

    # -- frequency band ---------------------------------------------------
    def band(self):
        """Angular-frequency band that holds the energy of all systems."""
        wps = [2 * np.pi / s.tp for s in self.systems]
        return 0.55 * min(wps), 5.0 * max(wps)

    def hs_total(self):
        """Deep-water significant height of all systems combined."""
        return np.sqrt(sum(s.hs ** 2 for s in self.systems))

    # -- transforms applied to plane waves -----------------------------------
    def _transform(self, omega, theta, amp):
        """Apply depth and current transforms to incident plane waves.

        Returns intrinsic frequency, wavenumber magnitude, amplitude and a
        mask of surviving (unblocked) components.
        """
        ux, uy = self.current
        if self.depth is None and ux == 0.0 and uy == 0.0:
            k = wavenumber(omega)
            return omega.copy(), k, amp.copy(), np.ones_like(omega, bool)
        if self.depth is not None:
            if ux != 0.0 or uy != 0.0:
                raise ValueError("currents are supported in deep water only")
            k = wavenumber(omega, self.depth)
            cg0 = G / (2.0 * omega)
            cg = group_velocity(k, self.depth)
            amp2 = amp * np.sqrt(cg0 / cg)
            return omega.copy(), k, amp2, np.ones_like(omega, bool)
        # Deep water on a uniform current, absolute frequency conserved.
        uk = ux * np.cos(theta) + uy * np.sin(theta)
        ok = np.ones_like(omega, bool)
        x = np.empty_like(omega)
        small = np.abs(uk) < 1e-12
        x[small] = omega[small] / np.sqrt(G)
        disc = G + 4.0 * uk * omega
        ok &= disc > 0.0
        big = ~small & ok
        x[big] = (-np.sqrt(G) + np.sqrt(disc[big])) / (2.0 * uk[big])
        x[~ok] = 1.0
        k = x ** 2
        sigma = np.sqrt(G * k)
        cg = 0.5 * np.sqrt(G / k)
        denom = cg + uk
        ok &= denom > 1e-6
        ratio = np.where(ok, (G / (2.0 * omega)) * sigma
                         / (np.maximum(denom, 1e-6) * omega), 0.0)
        amp2 = amp * np.sqrt(ratio)
        return sigma, k, amp2, ok

    def _reflect(self, kx, ky):
        """Reflected wavenumber and the phase giving a crest at the wall."""
        nx, ny = self.wall.normal
        kn = kx * nx + ky * ny
        krx = kx - 2.0 * kn * nx
        kry = ky - 2.0 * kn * ny
        x0, y0 = self.wall.x0
        dphase = (kx - krx) * x0 + (ky - kry) * y0
        towards = kn > 0.0
        return krx, kry, dphase, towards

    def _cap(self, amp):
        """Depth-limited saturation of the total significant height."""
        if self.depth is None:
            return amp
        hs = 4.0 * np.sqrt(0.5 * np.sum(np.abs(amp) ** 2))
        cap = 0.6 * self.depth
        if hs > cap:
            amp = amp * cap / hs
            self.extra["saturated"] = True
        return amp

    # -- deterministic spectral grid for M0 --------------------------------
    def spectral_grid(self, n_omega=160, n_theta=48):
        """Independent wave groups on a deterministic (omega, theta) grid.

        Returns
        -------
        dict
            ``a2`` variance weight (amplitude squared) per group (C,);
            ``omega`` absolute frequency (C,); ``sigma`` intrinsic
            frequency (C, M); ``kx``, ``ky`` (C, M); ``coef`` complex
            multiplier of each plane wave in a group (C, M).
        """
        w_lo, w_hi = self.band()
        omega_g = np.linspace(w_lo, w_hi, n_omega)
        dw = omega_g[1] - omega_g[0]
        out_w, out_th, out_a2 = [], [], []
        for sysw in self.systems:
            sw = jonswap(omega_g, sysw.hs, sysw.tp, sysw.gamma)
            if sysw.s is None:
                out_w.append(omega_g)
                out_th.append(np.full_like(omega_g, sysw.theta0))
                out_a2.append(2.0 * sw * dw)
            else:
                th = sysw.theta0 + np.linspace(-np.pi, np.pi, n_theta,
                                               endpoint=False)
                dth = th[1] - th[0]
                d = spreading_pdf(th, sysw.theta0, sysw.s)
                ww, tt = np.meshgrid(omega_g, th, indexing="ij")
                a2 = 2.0 * sw[:, None] * d[None, :] * dw * dth
                keep = a2 > 1e-9 * a2.max()
                out_w.append(ww[keep])
                out_th.append(tt[keep])
                out_a2.append(a2[keep])
        omega = np.concatenate(out_w)
        theta = np.concatenate(out_th)
        a2 = np.concatenate(out_a2)
        sigma, k, amp, ok = self._transform(omega, theta, np.sqrt(a2))
        omega, theta, sigma, k, amp = (v[ok] for v in
                                       (omega, theta, sigma, k, amp))
        amp = self._cap(amp)
        kx, ky = k * np.cos(theta), k * np.sin(theta)
        if self.wall is None:
            return dict(a2=amp ** 2, omega=omega, sigma=sigma[:, None],
                        kx=kx[:, None], ky=ky[:, None],
                        coef=np.ones((omega.size, 1), complex))
        krx, kry, dph, tow = self._reflect(kx, ky)
        coef = np.ones((omega.size, 2), complex)
        coef[:, 1] = np.where(tow, self.wall.refl * np.exp(1j * dph), 0.0)
        return dict(a2=amp ** 2, omega=omega,
                    sigma=np.stack([sigma, sigma], 1),
                    kx=np.stack([kx, krx], 1), ky=np.stack([ky, kry], 1),
                    coef=coef)

    # -- random realizations for M1 -----------------------------------------
    def realize(self, rng, n_omega=200):
        """Draw one random realization as a flat list of plane waves.

        Uses the single-summation method: one frequency per bin (jittered
        within the bin), a direction drawn from the spreading function,
        a uniform random phase, and amplitude ``sqrt(2 S dw)``.

        Returns
        -------
        dict
            Arrays ``coef`` (complex), ``kx``, ``ky``, ``omega``,
            ``sigma`` over all plane waves.
        """
        w_lo, w_hi = self.band()
        edges = np.linspace(w_lo, w_hi, n_omega + 1)
        dw = edges[1] - edges[0]
        ws, ths, amps = [], [], []
        for sysw in self.systems:
            w = edges[:-1] + dw * rng.random(n_omega)
            sw = jonswap(w, sysw.hs, sysw.tp, sysw.gamma)
            if sysw.s is None:
                th = np.full(n_omega, sysw.theta0)
            else:
                grid = np.linspace(-np.pi, np.pi, 4001)
                cdf = np.cumsum(spreading_pdf(grid, 0.0, sysw.s))
                cdf /= cdf[-1]
                th = sysw.theta0 + np.interp(rng.random(n_omega), cdf, grid)
            ws.append(w)
            ths.append(th)
            amps.append(np.sqrt(2.0 * sw * dw))
        omega = np.concatenate(ws)
        theta = np.concatenate(ths)
        amp = np.concatenate(amps)
        sigma, k, amp, ok = self._transform(omega, theta, amp)
        omega, theta, sigma, k, amp = (v[ok] for v in
                                       (omega, theta, sigma, k, amp))
        amp = self._cap(amp)
        phase = 2.0 * np.pi * rng.random(omega.size)
        coef = amp * np.exp(1j * phase)
        kx, ky = k * np.cos(theta), k * np.sin(theta)
        waves = dict(coef=coef, kx=kx, ky=ky, omega=omega, sigma=sigma)
        if self.wall is not None:
            krx, kry, dph, tow = self._reflect(kx, ky)
            refl = dict(coef=np.where(tow, self.wall.refl * coef
                                      * np.exp(1j * dph), 0.0),
                        kx=krx, ky=kry, omega=omega, sigma=sigma)
            waves = _concat(waves, refl)
        return waves


def _concat(a, b):
    return {key: np.concatenate([a[key], b[key]]) for key in a}


# ---------------------------------------------------------------------------
# Kinematics along a moving body
# ---------------------------------------------------------------------------

def field_along_body(waves, xi, t, origin, velocity, heading):
    """Surface kinematics at points fixed along a moving body axis.

    The points are ``X(t) + xi_j e`` with ``X(t) = origin + velocity t``
    and ``e = (cos heading, sin heading)``.

    Parameters
    ----------
    waves : dict
        Plane waves from :meth:`SeaState.realize`.
    xi : ndarray (J,)
        Along-axis coordinates of the points (m).
    t : ndarray (T,)
        Times (s).
    origin, velocity : tuple of float
        Position at t = 0 and ground velocity of the body (m, m/s).
    heading : float
        Direction of the body axis (rad).

    Returns
    -------
    eta, w, acc, slope : ndarray (J, T)
        Elevation, vertical particle velocity, vertical particle
        acceleration, and the surface slope across the body axis
        (positive toward the body's left, +y').
    """
    ex, ey = np.cos(heading), np.sin(heading)
    nx, ny = -ey, ex
    kx, ky = waves["kx"], waves["ky"]
    ke = kx * ex + ky * ey
    kn = kx * nx + ky * ny
    phase0 = kx * origin[0] + ky * origin[1]
    rate = kx * velocity[0] + ky * velocity[1] - waves["omega"]
    cmat = (waves["coef"] * np.exp(1j * phase0))[None, :] \
        * np.exp(1j * np.outer(xi, ke))
    emat = np.exp(1j * np.outer(rate, t))
    sig = waves["sigma"]
    eta = (cmat @ emat).real
    w = ((cmat * (-1j * sig)[None, :]) @ emat).real
    acc = ((cmat * (-(sig ** 2))[None, :]) @ emat).real
    slope = ((cmat * (1j * kn)[None, :]) @ emat).real
    return eta, w, acc, slope


def field_at_points(waves, x, y, t):
    """Surface elevation on an arbitrary grid of points (for rendering).

    Parameters
    ----------
    waves : dict
        Plane waves.
    x, y : ndarray
        Point coordinates (same shape).
    t : float
        Time (s).

    Returns
    -------
    ndarray
        Elevation with the shape of ``x``.
    """
    ph = (np.multiply.outer(x, waves["kx"])
          + np.multiply.outer(y, waves["ky"]) - waves["omega"] * t)
    return (waves["coef"] * np.exp(1j * ph)).real.sum(-1)
