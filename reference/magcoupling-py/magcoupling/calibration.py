"""Prototype measurement and model calibration ('Calibration' sheet).

Purpose: reproduce the original analytical model for the 3D-printed,
no-back-iron prototype (20 × B842SH, 1.4 mm flat-face gap), compare it with the
1.8 N·m bench measurement, and derive a one-point correction.

That correction applies ONLY to the same magnet rings without intentional back
iron. The steel-backed candidate keeps the original 0.95 factor (see
`model.select_calibration_factor`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ._fields import out, param
from .constants import MU0


@dataclass
class CalibrationInputs:
    measured_torque_Nm: float = param(1.8, "N·m", "Measured pull-out torque",
                                      "User bench result. Method, repeatability and temperature not supplied.", "Calibration!C5")
    total_magnets: int = param(20, "count", "Total installed magnets", "Two equal rings, alternating polarity.", "Calibration!C12")
    spacing_mm: float = param(1.4, "mm", "Reported spacing", "Between the opposing flat faces.", "Calibration!C14")
    gap_definition: int = param(1, "selector", "Gap definition", "0 = corner, 1 = flat centre.", "Calibration!C15",
                                {0: "corner", 1: "flat centre"})
    test_temp_C: float = param(20, "°C", "Assumed test magnet temperature", "ASSUMED; replace with the recorded value.", "Calibration!C16")
    apothem_mm: float = param(9.85, "mm", "Prototype inner hub apothem", "", "Calibration!C17")
    magnet_length_mm: float = param(12.7, "mm", "Prototype magnet axial length", "", "Calibration!C18")
    magnet_width_mm: float = param(6.35, "mm", "Prototype magnet tangential width", "", "Calibration!C19")
    magnet_thickness_mm: float = param(3.17, "mm", "Prototype magnet radial thickness", "", "Calibration!C20")
    br_T: float = param(1.29, "T", "Prototype remanence at 20 °C", "", "Calibration!C21")
    alpha_br_per_C: float = param(-0.0012, "1/°C", "Reversible Br temperature coefficient",
                                  "Used by every sheet. Torque scales with the square of the Br ratio.", "Calibration!C22")
    c_end: float = param(0.15, "-", "Original end-effect coefficient", "", "Calibration!C23")
    f_cal_original: float = param(0.95, "-", "Original calibration coefficient", "Kept separate from the measured correction.", "Calibration!C24")
    mu0: float = param(MU0, "H/m", "Vacuum permeability", "", "Calibration!C25")
    fea_gap1_mm: float = 1.0      # earlier 3D result points (Calibration!C47:C48)
    fea_torque1_Nm: float = param(2.06, "N·m", "3D result at 1.0 mm corner gap", "Supplied context, not rerun.", "Calibration!C47")
    fea_gap2_mm: float = 1.5
    fea_torque2_Nm: float = param(1.7, "N·m", "3D result at 1.5 mm corner gap", "", "Calibration!C48")


@dataclass
class CalibrationResults:
    model_torque_Nm: float = out("N·m", "Original model at the assumed test conditions", cell="Calibration!C6")
    model_error: float = out("fraction", "Model error relative to measured torque", "Negative means underprediction.", "Calibration!C7")
    measured_over_model: float = out("factor", "Measured / original model", "One-point correction.", "Calibration!C8")
    f_cal_updated: float = out("factor", "Updated model calibration coefficient",
                               "Applies to the no-iron prototype only.", "Calibration!C9")
    poles_per_ring: float = out("count", "Poles per ring", cell="Calibration!C13")
    face_radius_mm: float = out("mm", "Inner magnet flat-face radius", cell="Calibration!C28")
    corner_radius_mm: float = out("mm", "Inner magnet corner radius", cell="Calibration!C29")
    corner_gap_mm: float = out("mm", "Equivalent minimum corner gap", cell="Calibration!C30")
    outer_face_apothem_mm: float = out("mm", "Outer magnet face apothem", cell="Calibration!C31")
    flat_gap_mm: float = out("mm", "Effective flat-centre gap", cell="Calibration!C32")
    gap_radius_mm: float = out("mm", "Mean gap radius", cell="Calibration!C33")
    fill_inner: float = out("-", "Inner magnet fill factor", cell="Calibration!C34")
    fill_outer: float = out("-", "Outer magnet fill factor", cell="Calibration!C35")
    br_test_T: float = out("T", "Br at assumed test temperature", cell="Calibration!C36")
    pole_pitch_mm: float = out("mm", "Pole pitch", cell="Calibration!C37")
    f_end: float = out("-", "End-effect factor", cell="Calibration!C38")
    tau1_Pa: float = out("Pa", "Shear stress, harmonic 1", cell="Calibration!C40")
    tau3_Pa: float = out("Pa", "Shear stress, harmonic 3", cell="Calibration!C41")
    tau5_Pa: float = out("Pa", "Shear stress, harmonic 5", cell="Calibration!C42")
    torque_2d_Nm: float = out("N·m", "2D torque before end and calibration factors", cell="Calibration!C43")
    original_model_Nm: float = out("N·m", "Original model pull-out torque", "Same value as model_torque_Nm.", "Calibration!C44")
    fea_interp_Nm: object = out("N·m", "3D interpolation at the prototype corner gap", "Linear; 'outside range' if off the 1.0–1.5 mm span.",
                                "Calibration!C49")
    fea_interp_error: object = out("fraction", "3D interpolation error versus measured", cell="Calibration!C50")


def compute(c: CalibrationInputs) -> CalibrationResults:
    poles = c.total_magnets / 2
    r_face = c.apothem_mm + c.magnet_thickness_mm
    r_corner = math.sqrt(r_face ** 2 + (c.magnet_width_mm / 2) ** 2)
    corner_gap = c.spacing_mm if c.gap_definition == 0 else c.spacing_mm - (r_corner - r_face)
    a_o = r_corner + corner_gap
    g = a_o - r_face
    r_g = r_face + g / 2
    fill_i = min(1.0, c.magnet_width_mm / (2 * math.pi * (c.apothem_mm + c.magnet_thickness_mm / 2) / poles))
    fill_o = min(1.0, c.magnet_width_mm / (2 * math.pi * (a_o + c.magnet_thickness_mm / 2) / poles))
    br_t = c.br_T * (1 + c.alpha_br_per_C * (c.test_temp_C - 20))
    tau_p = 2 * math.pi * r_g / poles
    f_end = 1 - c.c_end * tau_p / c.magnet_length_mm

    def tau_n(n: int) -> float:
        k = n * (poles / 2) / (r_g / 1000)
        return ((br_t * 4 / (n * math.pi)) ** 2 * math.sin(n * fill_i * math.pi / 2) * math.sin(n * fill_o * math.pi / 2)
                / (2 * c.mu0) * (1 - math.exp(-k * c.magnet_thickness_mm / 1000)) ** 2
                * math.exp(-k * g / 1000) / 2 * math.sin(n * math.pi / 2))

    t1, t3, t5 = tau_n(1), tau_n(3), tau_n(5)
    t2d = (t1 + t3 + t5) * 2 * math.pi * (r_g / 1000) ** 2 * (c.magnet_length_mm / 1000)
    model = t2d * f_end * c.f_cal_original
    if 1 <= corner_gap <= 1.5:
        interp = c.fea_torque1_Nm + (c.fea_torque2_Nm - c.fea_torque1_Nm) * (corner_gap - 1) / 0.5
        interp_err = interp / c.measured_torque_Nm - 1
    else:
        interp, interp_err = "outside range", "n.a."
    ratio = c.measured_torque_Nm / model
    return CalibrationResults(
        model_torque_Nm=model, model_error=model / c.measured_torque_Nm - 1, measured_over_model=ratio,
        f_cal_updated=c.f_cal_original * ratio, poles_per_ring=poles, face_radius_mm=r_face, corner_radius_mm=r_corner,
        corner_gap_mm=corner_gap, outer_face_apothem_mm=a_o, flat_gap_mm=g, gap_radius_mm=r_g, fill_inner=fill_i,
        fill_outer=fill_o, br_test_T=br_t, pole_pitch_mm=tau_p, f_end=f_end, tau1_Pa=t1, tau3_Pa=t3, tau5_Pa=t5,
        torque_2d_Nm=t2d, original_model_Nm=model, fea_interp_Nm=interp, fea_interp_error=interp_err,
    )
