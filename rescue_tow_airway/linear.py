"""Model M0: exact linearization of M1 and closed-form airway statistics.

M0 is not a separate set of equations. It is the first-order expansion
of the M1 generalized forces about the floating equilibrium,

    M0 q'' = -K q - C q' + sum_j (G^eta_j eta_j + G^w_j w_j
                                  + G^a_j a_j + G^s_j s_j),

with all Jacobians computed by central differences of the same force
routine. The quadratic drag and roll damping are replaced by
statistically equivalent linear coefficients,

    b_eq = sqrt(8 / pi) sigma_v (1/2 rho C_D 2b),

where ``sigma_v`` is the standard deviation of the relative velocity,
iterated to convergence (stochastic linearization).

For a plane wave of unit amplitude the forcing is harmonic at the
encounter frequency ``omega_e = omega - k . V``. Solving

    (K - omega_e^2 M0 - i omega_e C) q_hat = F_hat

gives the transfer function of the airway clearance ``r``. Because the
linear response to a Gaussian sea is Gaussian, the airway statistics
are closed form:

* fraction of time awash ``P = Phi(-mu / sigma_r)``;
* rate of submersion events (Rice) ``nu = sigma_rdot / (2 pi sigma_r)
  exp(-mu^2 / (2 sigma_r^2))``;
* mean awash duration ``P / nu``;
* probability that a breath of duration ``tau`` taken at a random time
  meets water, ``P + (1 - P)(1 - exp(-nu tau / (1 - P)))``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .body import FIN_AREA
from .dynamics import (NQ, airway_clearance, equilibrium,
                       generalized_forces, kinematics, point_height)
from .waves import RHO

SQ8PI = np.sqrt(8.0 / np.pi)


@dataclass
class AirwayStats:
    """Closed-form airway statistics from M0."""

    mean: float
    sigma: float
    sigma_dot: float
    p_awash: float
    rate: float
    duration: float
    p_breath: float
    sig_heave: float
    sig_pitch: float
    sig_roll: float
    sig_head: float
    tz_enc: float


class LinearModel:
    """Linearized dynamics of a towed victim at a given tow speed.

    Parameters
    ----------
    body : Body
    speed : float
        Tow speed through the water (m/s).
    q0 : ndarray (4,), optional
        Linearization point; defaults to the still-water equilibrium.
    """

    def __init__(self, body, speed=0.0, q0=None):
        self.body = body
        self.speed = speed
        self.q0 = equilibrium(body, speed) if q0 is None else q0
        self._jacobians()

    # -- finite-difference Jacobians -------------------------------------
    def _jacobians(self):
        b = self.body
        nj = b.strip_xi.size
        hq, hw = 1e-5, 1e-5
        base_q = np.tile(self.q0, (1, 1))
        qs, vs = [base_q], [np.zeros((1, NQ))]
        ws = [np.zeros((1, 4, nj))]
        for i in range(NQ):
            for sgn in (1, -1):
                dq = np.zeros((1, NQ))
                dq[0, i] = sgn * hq
                qs.append(base_q + dq)
                vs.append(np.zeros((1, NQ)))
                ws.append(np.zeros((1, 4, nj)))
        for i in range(NQ):
            for sgn in (1, -1):
                dv = np.zeros((1, NQ))
                dv[0, i] = sgn * hq
                qs.append(base_q)
                vs.append(dv)
                ws.append(np.zeros((1, 4, nj)))
        for kind in range(4):
            for j in range(nj):
                for sgn in (1, -1):
                    dw = np.zeros((1, 4, nj))
                    dw[0, kind, j] = sgn * hw
                    qs.append(base_q)
                    vs.append(np.zeros((1, NQ)))
                    ws.append(dw)
        q = np.concatenate(qs)
        v = np.concatenate(vs)
        wv = np.concatenate(ws)
        lin_drag = np.zeros(b.xi.size + 1)
        lin_roll = np.zeros(b.xi.size)
        Q, M = generalized_forces(q, v, wv[:, 0], wv[:, 1], wv[:, 2],
                                  wv[:, 3], b, self.speed,
                                  lin_drag=lin_drag, lin_roll=lin_roll)
        self.Q0 = Q[0]
        self.M0 = M[0]
        k = 1
        dQdq = np.empty((NQ, NQ))
        for i in range(NQ):
            dQdq[:, i] = (Q[k] - Q[k + 1]) / (2 * hq)
            k += 2
        dQdv = np.empty((NQ, NQ))
        for i in range(NQ):
            dQdv[:, i] = (Q[k] - Q[k + 1]) / (2 * hq)
            k += 2
        gw = np.empty((4, NQ, nj))
        for kind in range(4):
            for j in range(nj):
                gw[kind, :, j] = (Q[k] - Q[k + 1]) / (2 * hw)
                k += 2
        self.K = -dQdq
        self.C0 = -dQdv
        self.Geta, self.Gw0, self.Gacc, self.Gs = gw
        # Airway kinematic gradient.
        xa, ya, za = b.airway
        ja = np.empty(NQ)
        for i in range(NQ):
            qp = self.q0.copy()
            qm = self.q0.copy()
            qp[i] += hq
            qm[i] -= hq
            ja[i] = (point_height(qp[None], b, xa, ya, za, True)[1][0]
                     - point_height(qm[None], b, xa, ya, za, True)[1][0]) \
                / (2 * hq)
        self.Ja = ja
        self.mean_clear = airway_clearance(self.q0[None], np.zeros(1),
                                           np.zeros(1), b)[0]
        kin = kinematics(self.q0[None], b)
        ct = kin["ct"][0, 0]
        head = b.head
        self.Jc = np.stack([np.ones(b.xi.size), kin["x2"][0],
                            kin["yg"][0] * ct,
                            head * (kin["yg"][0] - kin["ypiv"][0]) * ct], -1)
        # Equilibrium wetted fraction and effective breadth of sections.
        zero = np.zeros(b.xi.size)
        from .dynamics import _ellipse_below_line
        _, _, _, frac = _ellipse_below_line(kin["yg"][0], kin["zg"][0], b.b,
                                            b.c, kin["alpha"][0], zero, zero)
        self.frac = frac * b.wet
        self.beff = b.b.copy()
        self.jfin = np.array([1.0, b.fin_xi * ct, 0.0, 0.0])

    # -- equivalent damping ------------------------------------------------
    def _damping(self, ld, lr):
        """Assemble linear damping and velocity forcing from equivalent
        drag coefficients ``ld`` (per primitive, plus fins) and roll
        coefficients ``lr``."""
        b = self.body
        wgt = ld[:-1] * self.frac * b.dxi
        C = self.C0 + np.einsum("p,pi,pj->ij", wgt, self.Jc, self.Jc)
        C += ld[-1] * np.outer(self.jfin, self.jfin)
        isbody = (b.kind == 0)
        rw = lr * self.frac * b.dxi * isbody
        jw = np.stack([np.zeros(b.xi.size), np.zeros(b.xi.size),
                       np.ones(b.xi.size), b.head * 1.0], -1)
        C += np.einsum("p,pi,pj->ij", rw, jw, jw)
        gw = self.Gw0.copy()
        nj = b.strip_xi.size
        add = np.zeros((NQ, nj))
        np.add.at(add.T, b.strip, wgt[:, None] * self.Jc)
        add[:, b.fin_strip] += ld[-1] * self.jfin
        return C, gw + add

    # -- frequency response ---------------------------------------------
    def response(self, grid, heading, current=(0.0, 0.0), origin=(0.0, 0.0),
                 tau_breath=1.0, iterations=6, return_transfer=False):
        """Airway statistics for a sea state at a given tow heading.

        Parameters
        ----------
        grid : dict
            Output of :meth:`SeaState.spectral_grid`.
        heading : float
            Direction of the tow (and of the body axis) in the frame of
            the sea state (rad).
        current : tuple
            Surface current (m/s), added to the tow velocity.
        origin : tuple
            Position of the body center (m); matters only near a wall.
        tau_breath : float
            Duration of one rescue breath (s).
        iterations : int
            Stochastic-linearization iterations.

        Returns
        -------
        AirwayStats
        """
        b = self.body
        hy = b.hydro
        ex, ey = np.cos(heading), np.sin(heading)
        nx, ny = -ey, ex
        vx = self.speed * ex + current[0]
        vy = self.speed * ey + current[1]
        kx, ky, coef = grid["kx"], grid["ky"], grid["coef"]
        sig = grid["sigma"]
        omega = grid["omega"]
        a2 = grid["a2"]
        if kx.shape[1] > 1 and (abs(vx) > 1e-9 or abs(vy) > 1e-9):
            # A moving body meets incident and reflected waves at different
            # encounter frequencies; treat them as incoherent.
            a2 = (a2[:, None] * np.abs(coef) ** 2).ravel()
            kx, ky, sig = kx.reshape(-1, 1), ky.reshape(-1, 1), \
                sig.reshape(-1, 1)
            omega = np.repeat(omega, coef.shape[1])
            coef = np.ones_like(kx, complex)
            keep = a2 > 0
            a2, kx, ky, sig, omega, coef = (v[keep] for v in
                                            (a2, kx, ky, sig, omega, coef))
        ke = kx * ex + ky * ey
        kn = kx * nx + ky * ny
        we = omega - (kx[:, 0] * vx + ky[:, 0] * vy)
        c0 = coef * np.exp(1j * (kx * origin[0] + ky * origin[1]))
        ph = c0[:, :, None] * np.exp(1j * ke[:, :, None]
                                     * b.strip_xi[None, None, :])
        eta_j = ph.sum(1)
        w_j = (ph * (-1j * sig)[:, :, None]).sum(1)
        a_j = (ph * (-(sig ** 2))[:, :, None]).sum(1)
        s_j = (ph * (1j * kn)[:, :, None]).sum(1)

        ld = np.full(b.xi.size + 1, 0.0)
        lr = np.zeros(b.xi.size)
        beff = self.beff
        sv = np.full(b.xi.size, 0.1)
        sfin = 0.1
        som = np.full(b.xi.size, 0.1)
        for _ in range(iterations):
            ld[:-1] = SQ8PI * sv * 0.5 * RHO * hy.cd * 2 * beff
            ld[-1] = SQ8PI * sfin * 0.5 * RHO * hy.cd_fin * FIN_AREA
            lr = SQ8PI * som * hy.c44 * RHO * beff ** 5
            C, gw = self._damping(ld, lr)
            F = (eta_j @ self.Geta.T + w_j @ gw.T + a_j @ self.Gacc.T
                 + s_j @ self.Gs.T)
            H = (self.K[None] - we[:, None, None] ** 2 * self.M0[None]
                 - 1j * we[:, None, None] * C[None])
            qh = np.linalg.solve(H, F[..., None])[..., 0]
            vel = -1j * we[:, None] * qh
            wr = vel @ self.Jc.T - w_j[:, b.strip]
            sv = np.sqrt(0.5 * (a2[:, None] * np.abs(wr) ** 2).sum(0))
            wf = vel @ self.jfin - w_j[:, b.fin_strip]
            sfin = np.sqrt(0.5 * (a2 * np.abs(wf) ** 2).sum())
            om = vel[:, 2:3] + b.head[None, :] * vel[:, 3:4]
            som = np.sqrt(0.5 * (a2[:, None] * np.abs(om) ** 2).sum(0))
        rh = qh @ self.Ja - eta_j[:, b.airway_strip]
        s2 = 0.5 * np.sum(a2 * np.abs(rh) ** 2)
        sd2 = 0.5 * np.sum(a2 * we ** 2 * np.abs(rh) ** 2)
        st = airway_statistics(self.mean_clear, np.sqrt(s2), np.sqrt(sd2),
                               tau_breath)
        sq = np.sqrt(0.5 * (a2[:, None] * np.abs(qh) ** 2).sum(0))
        out = AirwayStats(*st, sig_heave=sq[0], sig_pitch=sq[1],
                          sig_roll=sq[2], sig_head=sq[3],
                          tz_enc=2 * np.pi * np.sqrt(s2 / max(sd2, 1e-30)))
        if return_transfer:
            return out, dict(we=we, rh=rh, qh=qh, a2=a2, eta_a=eta_j[:,
                             b.airway_strip])
        return out

    def eigen(self):
        """Damped natural frequencies (rad/s) and damping ratios."""
        n = NQ
        C, _ = self._damping(np.zeros(self.body.xi.size + 1),
                             np.zeros(self.body.xi.size))
        Minv = np.linalg.inv(self.M0)
        A = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-Minv @ self.K, -Minv @ C]])
        lam = np.linalg.eigvals(A)
        lam = lam[np.imag(lam) >= 0]
        wn = np.abs(lam)
        zeta = -np.real(lam) / np.maximum(wn, 1e-12)
        order = np.argsort(wn)
        return wn[order], zeta[order], lam[order]


def airway_statistics(mean, sigma, sigma_dot, tau_breath=1.0):
    """Gaussian (Rice) airway statistics.

    Parameters
    ----------
    mean : float
        Mean airway clearance above the awash margin (m).
    sigma, sigma_dot : float
        Standard deviations of the clearance and of its rate.
    tau_breath : float
        Breath duration (s).

    Returns
    -------
    tuple
        ``(mean, sigma, sigma_dot, p_awash, rate, duration, p_breath)``.
    """
    sigma = max(sigma, 1e-12)
    p = norm.cdf(-mean / sigma)
    nu = sigma_dot / (2 * np.pi * sigma) * np.exp(-0.5 * (mean / sigma) ** 2)
    dur = p / nu if nu > 0 else np.inf
    dry = max(1.0 - p, 1e-12)
    pb = p + dry * (1.0 - np.exp(-nu * tau_breath / dry))
    return mean, sigma, sigma_dot, p, nu, dur, pb
