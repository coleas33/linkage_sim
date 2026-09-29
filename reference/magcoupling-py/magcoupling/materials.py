"""Material data and material-driven checks ('Materials' sheet).

Plan: annealed 4140 with high-phosphorus electroless nickel for every part in
the magnetic path (inner hub, outer cup and web); 7075-T6 for clamp collars and
adapters; 6061-T6 for the cap, housing and brackets; ISO 4762 class 12.9 cap
screws; 316L sleeve, liner and endplates.

The steel values feed the Calculator back-iron check, the slip-loss estimate,
the heat capacity and the adhesive thermal-mismatch screen.
"""
from __future__ import annotations

from dataclasses import dataclass

from ._fields import ceiling, out, param


@dataclass
class Steel4140:
    bsat_T: float = param(1.5, "T", "Design flux density for the back-iron check",
                          "Annealed 4140. 1018 would be about 1.7 T; use about 1.4 T for pre-hardened stock.", "Materials!C13")
    conductivity_S_m: float = param(4.5e6, "S/m", "Electrical conductivity", "Resistivity about 0.22 µΩ·m.", "Materials!C14")
    mu_r_incremental: float = param(200, "-", "Incremental relative permeability (with the magnet bias)", "", "Materials!C15")
    specific_heat_J_kgK: float = param(473, "J/(kg·K)", "Specific heat", "", "Materials!C16")
    cte_per_C: float = param(12.3e-6, "1/°C", "Expansion coefficient", "", "Materials!C17")
    modulus_GPa: float = param(205, "GPa", "Elastic modulus", "", "Materials!C18")
    density_g_cm3: float = param(7.85, "g/cm³", "Density", "Same as 1018; the mass model uses Metal design C132.", "Materials!C19")


@dataclass
class ElectrolessNickel:
    thickness_mm: float = param(0.015, "mm", "Plating thickness per surface",
                                "High-phosphorus EN (10–12 % P): non-magnetic as plated. Typical 0.013–0.025 mm.", "Materials!C26")


@dataclass
class AluminiumAlloy:
    name: str
    yield_MPa: float
    shear_MPa: float
    head_pressure_limit_MPa: float
    key_bearing_allow_MPa: float
    conductivity_S_m: float


@dataclass
class Aluminium:
    al7075: AluminiumAlloy = None  # filled in __post_init__
    al6061: AluminiumAlloy = None

    def __post_init__(self):
        if self.al7075 is None:
            # Materials!C34:C38
            self.al7075 = AluminiumAlloy("7075-T6", 503, 331, 400, 100, 1.9e7)
        if self.al6061 is None:
            # Materials!C39:C43
            self.al6061 = AluminiumAlloy("6061-T6", 276, 207, 250, 60, 2.5e7)


@dataclass
class ScrewClasses:
    proof_12_9_MPa: float = param(970, "MPa", "Class 12.9 proof stress", "ISO 898-1.", "Materials!C48")
    proof_10_9_MPa: float = param(830, "MPa", "Class 10.9 proof stress", "", "Materials!C49")
    yield_A4_70_MPa: float = param(450, "MPa", "Stainless A4-70 yield stress", "", "Materials!C50")

    def proof(self, code: int) -> float:
        """code: 1 = 12.9, 2 = 10.9, 3 = A4-70 (workbook CHOOSE order)."""
        return {1: self.proof_12_9_MPa, 2: self.proof_10_9_MPa, 3: self.yield_A4_70_MPa}[code]


@dataclass
class MaterialsInputs:
    steel: Steel4140 = None
    nickel: ElectrolessNickel = None
    aluminium: Aluminium = None
    screws: ScrewClasses = None

    def __post_init__(self):
        self.steel = self.steel or Steel4140()
        self.nickel = self.nickel or ElectrolessNickel()
        self.aluminium = self.aluminium or Aluminium()
        self.screws = self.screws or ScrewClasses()


@dataclass
class MaterialsResults:
    backiron_thickness_needed_mm: float = out("mm", "Back-iron thickness needed at this flux density", cell="Materials!C20")
    cup_wall_corner_mm: float = out("mm", "Cup wall at the pocket corners", cell="Materials!C21")
    cup_wall_check: str = out("", "Cup wall check",
                              "A thicker corner wall grows the cup OD by twice the change; recheck the cap thread and envelope.",
                              cell="Materials!C22")
    hub_flats_under_mm: float = out("mm", "Machine the hub flats under by", "On the apothem.", cell="Materials!C27")
    cup_pockets_over_mm: float = out("mm", "Machine the cup pockets over by", "On the apothem.", cell="Materials!C28")
    bores_over_dia_mm: float = out("mm", "Machine bores over (on diameter)", cell="Materials!C29")
    ods_under_dia_mm: float = out("mm", "Machine outside diameters under (on diameter)", cell="Materials!C30")


def compute(mat: MaterialsInputs, t_bi_req_mm: float, wall_corner_mm: float) -> MaterialsResults:
    """Cup wall check against the back-iron need, and electroless-nickel pre-plate offsets."""
    if wall_corner_mm >= t_bi_req_mm:
        check = "OK"
    else:
        check = f"Too thin: raise Metal design C122 to at least {ceiling(t_bi_req_mm, 0.1):.1f} mm"
    t = mat.nickel.thickness_mm
    return MaterialsResults(
        backiron_thickness_needed_mm=t_bi_req_mm,
        cup_wall_corner_mm=wall_corner_mm,
        cup_wall_check=check,
        hub_flats_under_mm=t,
        cup_pockets_over_mm=t,
        bores_over_dia_mm=2 * t,
        ods_under_dia_mm=2 * t,
    )
