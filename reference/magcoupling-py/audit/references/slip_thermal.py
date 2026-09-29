"""Independent references for slip losses, the slip thermal network and life totals.

Written from the cited sources; nothing here imports ``magcoupling``. The Task 6
checks (``audit/tests/test_slip_thermal.py``) compare the engine's Temperature
design slip-loss, thermal, slip-life, magnet-life, adhesive-life and summary rows
against these functions; ``audit/tools/placeholder_sensitivity.py`` uses
``central_difference`` for the report's placeholder table.

Units: SI throughout (m, s, T, S/m, W, J, K), except masses in grams where named.

Sources
  [J]  J. D. Jackson, Classical Electrodynamics, 3rd ed. (Wiley, 1999), sec. 8.1:
       skin depth delta = sqrt(2/(mu sigma omega)); mean loss per unit area of a
       good conductor mu_c omega delta |H_par|^2 / 4.
  [HB] W. H. Hayt, J. A. Buck, Engineering Electromagnetics (McGraw-Hill):
       copper (sigma = 5.8e7 S/m) has delta = 66.1/sqrt(f) mm.
  [S]  R. L. Stoll, The Analysis of Eddy Currents (Clarendon Press, 1974):
       travelling field over a conducting, permeable half-space,
       A = A0 exp(-gamma y) exp(j(omega t - k x)), gamma^2 = k^2 + j omega mu sigma.
  [RN] R. L. Russell, K. H. Norsworthy, "Eddy currents and wall losses in
       screened-rotor induction motors", Proc. IEE 105A (1958) 163-175:
       end factor of a finite-length thin screen, 1 - tanh(a)/a, a = k L / 2.
  [R]  J. R. Reitz, "Forces on moving magnets due to eddy currents",
       J. Appl. Phys. 41 (1970) 2067: thin-sheet characteristic speed
       w = 2/(mu0 sigma t); the eddy drag peaks at v = w.
  [B]  G. Bertotti, Hysteresis in Magnetism (Academic Press, 1998): classical
       eddy loss of a lamination, P/V = pi^2 sigma d^2 f^2 B^2 / 6.
  [TG] S. P. Timoshenko, J. N. Goodier, Theory of Elasticity, 3rd ed.
       (McGraw-Hill, 1970): torsion of a rectangular bar, J_t = beta a^3 b with
       beta = 0.141, 0.229, 0.263, 0.312, 1/3 for b/a = 1, 2, 3, 10, infinity.
       The low-frequency eddy stream function in a prism solves the same
       Poisson problem (Prandtl membrane analogy).
  [I]  F. P. Incropera, D. P. DeWitt et al., Fundamentals of Heat and Mass
       Transfer, 6th ed. (Wiley, 2007), ch. 5: lumped capacitance,
       theta(t) = theta_ss (1 - exp(-t/tau)), tau = C/G.
"""
from __future__ import annotations

import cmath
import math
from typing import Callable, Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded


# ============================================================ eddy currents: steel
def skin_depth_m(omega: float, mu_r: float, sigma: float, mu0: float) -> float:
    """[J] Skin depth delta = sqrt(2 / (omega mu0 mu_r sigma)), in metres."""
    return math.sqrt(2.0 / (omega * mu0 * mu_r * sigma))


def halfspace_gamma(omega: float, mu_r: float, sigma: float, k: float, mu0: float) -> complex:
    """[S] Decay constant inside the half-space, gamma = sqrt(k^2 + j omega mu0 mu_r sigma), Re gamma > 0."""
    return cmath.sqrt(k * k + 1j * omega * mu0 * mu_r * sigma)


def halfspace_surface_bn(b_image: float, omega: float, mu_r: float, sigma: float, k: float, mu0: float) -> float:
    """[S] Normal flux-density amplitude at the surface of a conducting, permeable half-space.

    ``b_image`` is the source's normal field at the surface doubled: the value an
    infinitely permeable, non-conducting surface carries (image method). With
    A = S e^{-ky} + R e^{ky} in air and T e^{-gamma y} inside, continuity of A and of
    H_t gives T = 2S / (1 + gamma/(mu_r k)), so Bn = b_image / |1 + gamma/(mu_r k)|.
    Limits: mu_r -> inf gives b_image; sigma = 0 gives b_image mu_r/(mu_r + 1), the
    static image coefficient (mu_r - 1)/(mu_r + 1).
    """
    g = halfspace_gamma(omega, mu_r, sigma, k, mu0)
    return abs(b_image / (1.0 + g / (mu_r * k)))


def halfspace_loss_per_area(bn: float, omega: float, mu_r: float, sigma: float, k: float, mu0: float) -> float:
    """[S] Mean loss per unit area for a surface normal amplitude ``bn``, valid for any k*delta.

    Bn = k |A0| and J = -j omega sigma A0 e^{-gamma y}, so
    P/A = int_0^inf |J|^2 / (2 sigma) dy = sigma omega^2 bn^2 / (4 k^2 Re gamma).
    """
    g = halfspace_gamma(omega, mu_r, sigma, k, mu0)
    return sigma * omega ** 2 * bn ** 2 / (4.0 * k * k * g.real)


def halfspace_loss_thin_skin(bn: float, omega: float, mu_r: float, sigma: float, k: float, mu0: float) -> float:
    """k*delta -> 0 limit of ``halfspace_loss_per_area`` (Re gamma -> 1/delta): sigma omega^2 bn^2 delta / (4 k^2)."""
    return sigma * omega ** 2 * bn ** 2 * skin_depth_m(omega, mu_r, sigma, mu0) / (4.0 * k * k)


def halfspace_poynting_per_area(bn: float, omega: float, mu_r: float, sigma: float, k: float, mu0: float) -> float:
    """[J] Mean Poynting flux into the surface, Re(E_z H_x*)/2, with E_z = -j omega A and H_x = -gamma A/(mu0 mu_r).

    Energy conservation makes it equal ``halfspace_loss_per_area``; the sanity test uses that."""
    g = halfspace_gamma(omega, mu_r, sigma, k, mu0)
    a = bn / k
    e_z = -1j * omega * a
    h_x = -g * a / (mu0 * mu_r)
    return 0.5 * (e_z * h_x.conjugate()).real


# ============================================================ eddy currents: thin sheets
def thin_sheet_loss_low_speed(bn: float, v: float, sigma: float, t: float) -> float:
    """Low-speed loss per unit area of a thin non-magnetic sheet swept at speed v by a normal field of amplitude bn.

    Infinitely long sheet, so the currents run straight across it: E = v x B,
    J = sigma v bn cos(kx - wt), P/A = t <J^2> / sigma = sigma t v^2 bn^2 / 2."""
    return sigma * t * v * v * bn * bn / 2.0


def thin_disk_loss_low_speed(omega: float, sigma: float, t: float, bz2_r2_integral: float) -> float:
    """Low-speed loss of a thin non-magnetic disk turning at omega relative to an axial field.

    v = omega r and E = v x B is radial, so an unbounded sheet has loss density sigma (omega r)^2 <Bz^2>
    (<.> = mean over angle, no return-path correction). Over the disk volume:
    sigma t omega^2 * int <Bz^2> r^2 dA, where ``bz2_r2_integral`` = int <Bz^2> r^2 dA in T^2 m^4."""
    return sigma * t * omega ** 2 * bz2_r2_integral


def thin_sheet_reaction_factor(v: float, sigma: float, t: float, mu0: float) -> float:
    """[R] Loss reduction from the sheet's own field: 1 / (1 + eps^2), eps = v / w = mu0 sigma t v / 2.

    One spatial harmonic k, sheet in free space: the sheet current K = -j omega sigma t A_s adds
    mu0 K / (2k) to the potential, so A_s = A_0 / (1 + j eps) with eps = omega sigma t mu0 / (2k)."""
    eps = mu0 * sigma * t * v / 2.0
    return 1.0 / (1.0 + eps * eps)


def sheet_end_factor(k: float, field_length: float, overhang: float = 0.0) -> float:
    """[RN] Low-speed loss of a finite thin sheet relative to the infinitely long one.

    Normal field bn cos(kx - wt), uniform over ``field_length`` and zero beyond; the sheet
    extends ``overhang`` past each end. With phi = f(z) cos(kx), J = sigma(-grad phi + v B z_hat)
    closes through the sheet (div J = 0, J_z = 0 at its edges): f = A sinh(kz) in the field,
    D cosh(k(L/2 + h - |z|)) in the overhang, f continuous and f' jumping by v bn at |z| = L/2.
    The loss (work of the motional field) relative to the long sheet is
        1 - tanh(a) / (a (1 + tanh(a) tanh(k h))),   a = k L / 2.
    ``overhang = 0`` is Russell and Norsworthy's 1 - tanh(a)/a."""
    a = k * field_length / 2.0
    return 1.0 - math.tanh(a) / (a * (1.0 + math.tanh(a) * math.tanh(k * overhang)))


def sheet_end_factor_numeric(k: float, field_length: float, overhang: float = 0.0, cells: int = 4000) -> float:
    """Finite-volume solution of the ``sheet_end_factor`` problem, loss integrated as int |J|^2 / sigma.

    Independent of the closed form: no jump conditions are imposed by hand and the loss is summed
    from the current density, not from the work of the motional field. sigma = v bn = 1.
    Unknown f_i: cos(kx) amplitude of phi per cell along the sheet. Face current
    J = (f_i - f_{i+1} + e_i d_i + e_{i+1} d_{i+1}) / (d_i + d_{i+1}) (e = 1 in the field,
    d = half cell width), J = 0 at the sheet edges; cell balance J_right - J_left + k^2 w_i f_i = 0,
    the last term being the divergence of J_x = k f sin(kx)."""
    n_over = max(1, round(cells * overhang / (field_length + 2 * overhang))) if overhang > 0 else 0
    n_field = cells - 2 * n_over
    over = np.full(n_over, overhang / n_over) if n_over else np.empty(0)
    widths = np.concatenate([over, np.full(n_field, field_length / n_field), over])
    emf = np.concatenate([np.zeros(n_over), np.ones(n_field), np.zeros(n_over)])
    half = widths / 2
    dist = half[:-1] + half[1:]                                   # centre-to-centre across each interior face
    face_emf = (emf[:-1] * half[:-1] + emf[1:] * half[1:]) / dist
    diag = k * k * widths
    diag[:-1] += 1 / dist
    diag[1:] += 1 / dist
    rhs = np.zeros(widths.size)
    rhs[:-1] -= face_emf
    rhs[1:] += face_emf
    bands = np.zeros((3, widths.size))
    bands[0, 1:] = -1 / dist
    bands[1] = diag
    bands[2, :-1] = -1 / dist
    f = solve_banded((1, 1), bands, rhs)
    jz = np.concatenate([[0.0], (f[:-1] - f[1:]) / dist + face_emf, [0.0]])
    faces = np.concatenate([[0.0], np.cumsum(widths)])
    int_jz2 = float(np.sum((jz[:-1] ** 2 + jz[1:] ** 2) / 2 * np.diff(faces)))   # trapezoid, kink on a face
    int_jx2 = float(np.sum((k * f) ** 2 * widths))                               # midpoint
    return (int_jz2 + int_jx2) / field_length


# ============================================================ eddy currents: magnets
def strip_loss_per_volume(b: float, omega: float, sigma: float, d: float) -> float:
    """[B] Low-frequency eddy loss per unit volume of a thin strip of thickness d in a uniform
    alternating field of amplitude b along its faces: sigma omega^2 b^2 d^2 / 24 (= pi^2 sigma d^2 f^2 b^2 / 6)."""
    return sigma * omega ** 2 * b ** 2 * d ** 2 / 24.0


def rect_section_factor(aspect: float, terms: int = 400) -> float:
    """[TG] Low-frequency eddy loss of a long rectangular prism relative to ``strip_loss_per_volume`` (d = short side).

    Section a x b with aspect = b/a >= 1, alternating field uniform and along the prism. The stream
    function psi (J = curl(psi z_hat)) solves laplacian psi = -sigma dB/dt with psi = 0 on the boundary:
    Prandtl's torsion problem. The loss per length is sigma (dB/dt)^2 J_t / 4 with J_t = beta a^3 b, and a
    thin strip has beta = 1/3, so the ratio is
        3 beta = 1 - (192 / (pi^5 aspect)) sum_{n odd} tanh(n pi aspect / 2) / n^5."""
    if aspect < 1:
        raise ValueError("aspect is the long side over the short side (>= 1)")
    s = sum(math.tanh(n * math.pi * aspect / 2) / n ** 5 for n in range(1, 2 * terms, 2))
    return 1.0 - 192.0 / (math.pi ** 5 * aspect) * s


def loglog_slope(x1: float, y1: float, x2: float, y2: float) -> float:
    """Exponent n of y = c x^n through (x1, y1) and (x2, y2)."""
    return math.log(y2 / y1) / math.log(x2 / x1)


def central_difference(f: Callable[[float], float], x0: float, rel_step: float = 1e-4) -> float:
    """df/dx at x0 by the central difference (f(x0 + h) - f(x0 - h)) / 2h with h = rel_step |x0|.

    Truncation error is f'''(x0) h^2 / 6, second order in h; x0 must be non-zero."""
    if x0 == 0.0:
        raise ValueError("central_difference needs a non-zero x0 (the step is relative)")
    h = rel_step * abs(x0)
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


# ============================================================ lumped thermal network
def heat_capacity_J_K(parts: Iterable[tuple[float, float]]) -> float:
    """[I] Lumped heat capacity sum(m c) for parts given as (mass in g, specific heat in J/(kg K))."""
    return sum(m_g * c for m_g, c in parts) / 1000.0


def first_order_rise(power_W: float, conductance_W_K: float, capacity_J_K: float, t_s: float) -> float:
    """[I] Rise after heating at constant power for t_s from equilibrium: (P/G) (1 - exp(-t G / C))."""
    return power_W / conductance_W_K * (1.0 - math.exp(-t_s * conductance_W_K / capacity_J_K))


def time_to_rise(target_rise: float, power_W: float, conductance_W_K: float, capacity_J_K: float) -> float:
    """[I] Time for ``first_order_rise`` to reach ``target_rise``.

    0 when target_rise <= 0 (the limit is already reached at the start); inf when the steady
    rise P/G does not exceed it (it is approached only asymptotically)."""
    if target_rise <= 0.0:
        return 0.0
    steady = power_W / conductance_W_K
    if steady <= target_rise:
        return math.inf
    return -capacity_J_K / conductance_W_K * math.log(1.0 - target_rise / steady)


def _network(power_W: float, conductance_W_K: float, capacity_J_K: float):
    return lambda t, y: [(power_W - conductance_W_K * y[0]) / capacity_J_K]


def ode_rise(power_W: float, conductance_W_K: float, capacity_J_K: float, t_s: float) -> float:
    """Numerical reference: integrate C dtheta/dt = P - G theta from theta = 0 to t_s (DOP853, rtol 1e-12)."""
    sol = solve_ivp(_network(power_W, conductance_W_K, capacity_J_K), (0.0, t_s), [0.0],
                    method="DOP853", rtol=1e-12, atol=1e-14)
    return float(sol.y[0, -1])


def ode_time_to_rise(target_rise: float, power_W: float, conductance_W_K: float, capacity_J_K: float,
                     t_max_s: float) -> float:
    """Numerical reference: first time the integrated rise reaches ``target_rise``.

    0 when target_rise <= 0 (reached at the start); inf when not reached by t_max_s."""
    if target_rise <= 0.0:
        return 0.0

    def hit(t, y):
        return y[0] - target_rise

    hit.terminal = True
    hit.direction = 1
    sol = solve_ivp(_network(power_W, conductance_W_K, capacity_J_K), (0.0, t_max_s), [0.0],
                    method="DOP853", events=hit, rtol=1e-12, atol=1e-14)
    return float(sol.t_events[0][0]) if sol.t_events[0].size else math.inf


def event_train_mean_rise(power_W: float, conductance_W_K: float, capacity_J_K: float,
                          t_on_s: float, period_s: float) -> float:
    """Mean rise of the periodic steady state when power flows for t_on_s in every period_s.

    A separate model from 'duty x steady rise': exact piecewise exponentials of the first-order
    network, the periodic condition solved in closed form, then averaged over one period."""
    tau = capacity_J_K / conductance_W_K
    steady = power_W / conductance_W_K
    e_on, e_off = math.exp(-t_on_s / tau), math.exp(-(period_s - t_on_s) / tau)
    top = steady * (1.0 - e_on) / (1.0 - e_on * e_off)      # rise at the end of each heating pulse
    bottom = top * e_off                                     # rise at the start of each heating pulse
    area_on = steady * t_on_s + (bottom - steady) * tau * (1.0 - e_on)
    area_off = top * tau * (1.0 - e_off)
    return (area_on + area_off) / period_s
