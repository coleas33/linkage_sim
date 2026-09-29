"""Independent references for the Temperature design demagnetization and adhesive sections (M1 audit, Task 5).

Nothing here imports ``magcoupling``. Every function is written from first principles or from a
cited source. Br(T) comes from Task 4's ``remanence.br_ratio`` and rpm -> rad/s from Task 4's
``metal_stack.omega_rad_s`` (one definition each for the audit).

Demagnetization (knee crossing)
    Reverse fields scale with Br:       H(T)  = H20 * Br(T) / Br20
    (for a recoil permeability of ~1 every field in a fixed geometry, both the
    self-demagnetizing field and the field of the other ring, is proportional to Br).
    Knee of the intrinsic curve:        Hk(T) = knee * Hcj20 * (1 + beta * (T - 20))
    Onset: the temperature where |H(T)| = Hk(T). Here it is found by bracketing
    root-finding (Brent), which is a different method from the engine's closed form.

Load line (permeance coefficient)
    A linear magnet with recoil permeability mu_rec on load line B = -Pc * mu0 * H:
    Br + mu_rec * mu0 * H = -Pc * mu0 * H  =>  |H| = Br / (mu0 * (mu_rec + Pc)).
    Textbook form, e.g. Campbell, *Permanent Magnet Materials and their Application*
    (Cambridge, 1994), ch. 5; Furlani, *Permanent Magnet and Electromechanical Devices*
    (Academic Press, 2001), sec. 3.4.

Supplier ratings (K&J Magnetics, the workbook's magnet vendor)
    Maximum operating temperature by grade suffix and Br ranges by grade, from the K&J
    "Neodymium Magnet Specifications" page (kjmagnetics.com/specs.asp, read 2026-09-28).
    K&J states the rating depends on magnet shape (permeance coefficient) and is a guideline.

Adhesive
    Design limit: the lower of the TDS service maximum and Tg - 20 degC.
    Bond shear: torque / (blocks * lever arm) spread over the bonded back face.
    Centrifugal force: m * omega^2 * r.

Volkersen shear lag for thermal mismatch (derived here)
    Adherend 1 (magnet, E1, t1) and adherend 2 (steel, E2, t2) per unit width, joined by
    an adhesive layer of shear modulus G and thickness eta over an overlap of length L.
    Free ends, no external load, so the axial forces satisfy N2 = -N1.
        strains:      eps1 = N1/(E1 t1) + a1 dT,  eps2 = -N1/(E2 t2) + a2 dT
        adhesive:     tau = G (u2 - u1) / eta,    equilibrium: dN1/dx = -tau
        =>            N1'' - lam^2 N1 = -(G/eta) (a2 - a1) dT,  lam^2 = (G/eta)(1/(E1 t1) + 1/(E2 t2))
        with N1(+-L/2) = 0:
                      N1(x)  = (G da dT / (eta lam^2)) (1 - cosh(lam x)/cosh(lam L/2))
                      tau(x) = (G da dT / (eta lam)) sinh(lam x)/cosh(lam L/2)
        peak at the ends:  tau_max = G da dT tanh(lam L/2) / (eta lam)
    Sources: O. Volkersen, Luftfahrtforschung 15 (1938) 41-47 (shear lag);
    W.T. Chen and C.W. Nelson, "Thermal stress in bonded joints", IBM J. Res. Dev. 23 (1979)
    179-188 (the thermal-mismatch form); L.F.M. da Silva et al., Int. J. Adhes. Adhes. 29
    (2009) 319-330 (review).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

from audit.references.metal_stack import omega_rad_s
from audit.references.remanence import br_ratio

# --------------------------------------------------------------------------- literature data
#: Arnold Magnetic Technologies, N42SH datasheet (Rev. 020821): reversible temperature
#: coefficients measured between 20 and 150 degC; CTE between 20 and 200 degC.
ARNOLD_N42SH = {
    "alpha_br_per_C": -0.0012,         # -0.12 %/degC
    "beta_hcj_per_C": -0.0055,         # -0.55 %/degC
    "hcj20_min_kA_m": 1592.0,          # 20,000 Oe minimum
    "br_nominal_T": 1.31,              # 13,100 G nominal
    "hcb_nominal_kA_m": 987.0,         # 12,400 Oe nominal
    "coefficient_range_C": (20.0, 150.0),
    "cte_perpendicular_per_C": -1.0e-6,
    "cte_parallel_per_C": 7.0e-6,
    "density_g_cm3": 7.6,
}

#: K&J maximum operating temperature by grade suffix ("" = plain N grade): N 176 F (80 C), M 212 F (100 C),
#: H 248 F (120 C), SH 302 F (150 C), UH 356 F (180 C), EH 392 F (200 C), AH 428 F (220 C).
KJ_GRADE_MAX_OPERATING_C = {"": 80.0, "M": 100.0, "H": 120.0, "SH": 150.0, "UH": 180.0, "EH": 200.0, "AH": 220.0}

#: K&J Br range by grade [T]: N42 13.0-13.2 kG, N42SH 13.0-13.3 kG, N52 14.5-14.8 kG.
KJ_BR_RANGE_T = {"N42": (1.30, 1.32), "N42SH": (1.30, 1.33), "N52": (1.45, 1.48)}

#: Grade of the K&J parts the checks use: the B842SH product page ("1/2 x 1/4 x 1/8 Inch ... N42SH", max operating
#: temperature 302 F (150 C), Br max 13,200 G) lists the same block in the alternative grades N42 (B842) and N52 (B842-N52).
KJ_PART_GRADE = {"B842SH": "N42SH", "B842": "N42", "B842-N52": "N52"}


@dataclass(frozen=True)
class AdhesiveTds:
    """Numbers read from a manufacturer technical data sheet (TDS)."""
    name: str
    service_max_C: float | None
    tg_C: float | None
    tensile_modulus_GPa: float
    source: str


#: TDS data for the candidates whose sheets were read for this audit.
ADHESIVE_TDS = (
    AdhesiveTds("Loctite AA 326 + SF 7649", 120.0, None, 0.300,
                "Henkel TDS LOCTITE AA 326 (Aug-2020): heat-ageing data at 100 and 120 degC (no higher), "
                "Tg not published, tensile modulus ISO 527-2 = 300 N/mm^2, elongation 135 %, "
                "lap shear 15.2 N/mm^2 with Activator 7649 on one side, recommended bondline 0.1 mm."),
    AdhesiveTds("Loctite EA 9514", 200.0, 133.0, 1.460,
                "Henkel TDS LOCTITE EA 9514 (Oct-2014): Tg 133 degC (ASTM E1640), heat-ageing data to "
                "200 degC, tensile modulus ISO 527-3 = 1,460 N/mm^2, lap shear 45 N/mm^2."),
)


# --------------------------------------------------------------------------- supplier ratings
def grade_max_operating_C(grade: str) -> float:
    """K&J maximum operating temperature for an NdFeB grade such as 'N42', 'N42SH' or 'N50M'."""
    m = re.fullmatch(r"N(\d+)([A-Z]*)", grade)
    if m is None or m.group(2) not in KJ_GRADE_MAX_OPERATING_C:
        raise ValueError(f"not an NdFeB grade with a K&J rating: {grade!r}")
    return KJ_GRADE_MAX_OPERATING_C[m.group(2)]


def part_rating_C(part: str) -> float | None:
    """K&J rating of a part the checks use; None for a blank part (the engine then runs uncalibrated).

    Raises KeyError for a part this module has no grade for, so a check can never silently fall back.
    """
    if not part:
        return None
    return grade_max_operating_C(KJ_PART_GRADE[part])


# --------------------------------------------------------------------------- demagnetization
def reverse_field_kA_m(t_C: float, h_rev20_kA_m: float, alpha_br_per_C: float) -> float:
    """Magnitude of a reverse field that scales with Br(T) (fixed geometry, recoil permeability ~1)."""
    return h_rev20_kA_m * br_ratio(alpha_br_per_C, t_C)


def knee_field_kA_m(t_C: float, hcj20_kA_m: float, beta_hcj_per_C: float, knee_fraction: float) -> float:
    """Knee of the intrinsic curve: knee * Hcj(T), Hcj linear in T with the signed coefficient beta."""
    return knee_fraction * hcj20_kA_m * (1.0 + beta_hcj_per_C * (t_C - 20.0))


def load_line_reverse_field_kA_m(br_T: float, permeance_coefficient: float, mu0: float,
                                 recoil_permeability: float = 1.0) -> float:
    """|H| at the intersection of the linear demagnetization line and the load line B = -Pc*mu0*H."""
    return br_T / (mu0 * (recoil_permeability + permeance_coefficient)) / 1000.0


def knee_crossing_C(h_rev20_kA_m: float, hcj20_kA_m: float, beta_hcj_per_C: float, knee_fraction: float,
                    alpha_br_per_C: float, t_lo_C: float = -273.15, t_hi_C: float = 1273.15) -> float:
    """Temperature where the Br-scaled reverse field equals the knee field, by Brent root-finding.

    Raises ValueError when the two lines do not cross inside [t_lo_C, t_hi_C].
    """
    def margin(t_C: float) -> float:
        return (knee_field_kA_m(t_C, hcj20_kA_m, beta_hcj_per_C, knee_fraction)
                - reverse_field_kA_m(t_C, h_rev20_kA_m, alpha_br_per_C))

    lo, hi = margin(t_lo_C), margin(t_hi_C)
    if lo * hi > 0:
        raise ValueError(f"no knee crossing between {t_lo_C} and {t_hi_C} degC (margins {lo:.3g}, {hi:.3g} kA/m)")
    return brentq(margin, t_lo_C, t_hi_C, xtol=1e-12, rtol=4 * np.finfo(float).eps, maxiter=500)


def calibration_offset_C(h_ref_kA_m: float, rating_C: float | None, hcj20_kA_m: float, beta_hcj_per_C: float,
                         knee_fraction: float, alpha_br_per_C: float) -> float:
    """Model onset of the reference magnet minus its supplier rating (0 when there is no rating)."""
    if rating_C is None:
        return 0.0
    return knee_crossing_C(h_ref_kA_m, hcj20_kA_m, beta_hcj_per_C, knee_fraction, alpha_br_per_C) - rating_C


# --------------------------------------------------------------------------- adhesive
def adhesive_design_limit_C(service_max_C: float | None, tg_C: float | None, tg_margin_C: float = 20.0) -> float:
    """Lower of the TDS service maximum and Tg - margin; a missing value simply does not constrain."""
    limits = [v for v in (service_max_C, None if tg_C is None else tg_C - tg_margin_C) if v is not None]
    if not limits:
        raise ValueError("need a service maximum or a Tg")
    return min(limits)


def bond_shear_MPa(torque_Nm: float, n_blocks: int, lever_arm_mm: float, bond_area_mm2: float) -> float:
    """Average bond shear: tangential force per block (torque / (blocks * lever arm)) over the bond area."""
    force_N = torque_Nm / (n_blocks * lever_arm_mm / 1000.0)
    return force_N / bond_area_mm2


def centrifugal_force_N(mass_g: float, rpm: float, radius_mm: float) -> float:
    """m * omega^2 * r, omega = omega_rad_s(rpm)."""
    omega = omega_rad_s(rpm)
    return mass_g / 1000.0 * omega ** 2 * radius_mm / 1000.0


def shear_modulus_GPa(tensile_modulus_GPa: float, poisson: float) -> float:
    """Isotropic G = E / (2 (1 + nu))."""
    return tensile_modulus_GPa / (2.0 * (1.0 + poisson))


# --------------------------------------------------------------------------- Volkersen thermal shear lag
def volkersen_lambda_per_m(G_Pa: float, eta_m: float, E1_Pa: float, t1_m: float, E2_Pa: float, t2_m: float) -> float:
    """Shear-lag parameter lam = sqrt((G/eta) (1/(E1 t1) + 1/(E2 t2)))."""
    return math.sqrt(G_Pa / eta_m * (1.0 / (E1_Pa * t1_m) + 1.0 / (E2_Pa * t2_m)))


def volkersen_thermal_peak_shear_Pa(G_Pa: float, eta_m: float, E1_Pa: float, t1_m: float, E2_Pa: float, t2_m: float,
                                    d_alpha_per_C: float, dT_C: float, overlap_m: float) -> float:
    """Peak (end) adhesive shear from the closed form derived in the module docstring."""
    lam = volkersen_lambda_per_m(G_Pa, eta_m, E1_Pa, t1_m, E2_Pa, t2_m)
    return G_Pa * d_alpha_per_C * dT_C * math.tanh(lam * overlap_m / 2.0) / (eta_m * lam)


def volkersen_thermal_fd(G_Pa: float, eta_m: float, E1_Pa: float, t1_m: float, E2_Pa: float, t2_m: float,
                         d_alpha_per_C: float, dT_C: float, overlap_m: float,
                         n_nodes: int = 20001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite-difference solution of N1'' - lam^2 N1 = -(G/eta) da dT with N1(+-L/2) = 0.

    Returns (x [m], N1 [N/m], tau [Pa]); tau = -dN1/dx by second-order differences
    (central inside, one-sided at the ends). Discretization error is O((lam dx)^2).
    """
    lam = volkersen_lambda_per_m(G_Pa, eta_m, E1_Pa, t1_m, E2_Pa, t2_m)
    q = G_Pa / eta_m * d_alpha_per_C * dT_C
    x = np.linspace(-overlap_m / 2.0, overlap_m / 2.0, n_nodes)
    dx = x[1] - x[0]
    m = n_nodes - 2                                     # interior unknowns
    bands = np.zeros((3, m))
    bands[0, 1:] = 1.0 / dx ** 2                        # super-diagonal
    bands[1, :] = -2.0 / dx ** 2 - lam ** 2             # diagonal
    bands[2, :-1] = 1.0 / dx ** 2                       # sub-diagonal
    n_interior = solve_banded((1, 1), bands, np.full(m, -q))
    n1 = np.concatenate(([0.0], n_interior, [0.0]))
    dn = np.empty_like(n1)
    dn[1:-1] = (n1[2:] - n1[:-2]) / (2.0 * dx)
    dn[0] = (-3.0 * n1[0] + 4.0 * n1[1] - n1[2]) / (2.0 * dx)
    dn[-1] = (3.0 * n1[-1] - 4.0 * n1[-2] + n1[-3]) / (2.0 * dx)
    return x, n1, -dn
