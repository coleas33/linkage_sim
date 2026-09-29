"""Temperature design ('Temperature design' sheet).

Answers: how hot can the coupling get before the magnets demagnetize or the
bond gives up, how much heat does slip make, and how long can it slip.

Approach
  * Demagnetization: the worst reverse field inside a block (from a 3D
    magpylib check of this geometry, see `fields3d.py`) is compared with the
    knee of the N42SH curve at temperature, Hk(T) = knee · Hcj20 · (1 + β(T−20)),
    with fields scaling with Br(T). The model is calibrated so a
    permeance-coefficient-1 magnet reproduces the library's 150 °C rating.
    Skipping (like poles facing once per pole pass) sets the lowest onset.
  * Adhesive: design limit = TDS service maximum or Tg − 20 °C. Bond loads
    are the pull-out shear per block; magnetics press the blocks onto the steel.
    A Volkersen shear-lag screen estimates thermal-mismatch shear at the block
    ends (NdFeB barely expands across its magnetization; steel does).
  * Slip heating: eddy-current estimates for solid-steel surfaces (travelling
    field on a permeable conductor, ∝ speed^1.5), thin 316L shells and the
    aluminium cap (thin-shell low-speed limit, ∝ speed²) and the magnets
    (thin-strip formula). A single thermal RC network (heat capacity from the
    part masses, one conductance to the surroundings) gives per-event rise,
    steady continuous-slip temperature and time to the limit.
  * Life: slip rotations, like-pole passes, average heating from slip duty,
    and magnet/adhesive checks at the hot-day peak.

The 3D field numbers are fixed for the current geometry (10 poles, 1.4 mm
face gap, 4140 hub and cup). Rerun `fields3d.py` after geometry changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ._fields import out, param


def _text0(x: float) -> str:
    """Excel TEXT(x, "0") (round half away from zero)."""
    return str(int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1))


# =========================================================================== inputs
@dataclass
class DutyInputs:
    wheel_rotor_rpm: float = param(2000, "rpm", "Wheel-side (inner) rotor maximum speed", "Used for the centrifugal load.", "Temperature design!C34")
    hot_ambient_C: float = param(55, "°C", "Hot-day ambient temperature", "User: 55 °C day.", "Temperature design!C35")
    driving_rise_C: float = param(10, "°C", "Coupling rise above ambient while driving (no slip)",
                                 "Placeholder: housing air, sun and gearbox heat. Measure it.", "Temperature design!C36")
    fault_trip_s: float = param(2, "s", "Slip fault trip time (unbroken slip)", "", "Temperature design!C38")
    life_hours: float = param(20000, "h", "Operating hours over the system life", "Placeholder; used for the average slip duty.",
                              "Temperature design!C39")


@dataclass
class DemagInputs:
    hcj20_kA_m: float = param(1592, "kA/m", "Intrinsic coercivity Hcj at 20 °C (grade minimum)", "N42SH ≥ 20 kOe.", "Temperature design!C44")
    beta_hcj_per_C: float = param(-0.005, "1/°C", "Hcj temperature coefficient (effective, 20–150 °C)", "", "Temperature design!C45")
    knee_fraction: float = param(0.9, "-", "Knee field as a fraction of Hcj", "", "Temperature design!C46")
    design_margin_C: float = param(10, "°C", "Design margin below the onset", "", "Temperature design!C51")
    h_rev_aligned_kA_m: float = param(354, "kA/m", "3D worst reverse field, rings aligned", "Outer blocks (inner 341).", "Temperature design!C52")
    h_rev_pullout_kA_m: float = param(791, "kA/m", "3D worst reverse field at pull-out", "Outer blocks (inner 760).", "Temperature design!C53")
    h_rev_likepole_kA_m: float = param(863, "kA/m", "3D worst reverse field, like poles facing",
                                       "Outer blocks (inner 844); once per pole pass while skipping.", "Temperature design!C54")
    h_rev_single_ring_kA_m: float = param(569, "kA/m", "3D worst reverse field, single ring on its carrier",
                                          "Adhesive-cure case (inner ring alone: 545).", "Temperature design!C55")


@dataclass
class AdhesiveCandidate:
    name: str
    design_limit_C: float     # TDS service maximum or Tg − 20 °C
    cure_C: float             # stress-free (cure) temperature
    lap_shear_MPa: float      # TDS lap shear at ~22 °C
    role: str = ""
    note: str = ""


def default_adhesives() -> list[AdhesiveCandidate]:
    return [
        AdhesiveCandidate("Loctite AA 326 + SF 7649", 120, 22, 15, "Recommended",
                          "No-mix acrylic for magnet bonding; room-temperature cure; 0.10 mm bondline; service to 120 °C."),
        AdhesiveCandidate("Loctite EA 9514", 113, 120, 45, "Alternative: more hot strength",
                          "One-part toughened heat-cure epoxy; Tg 133 °C; cure 60 min at 120 °C (not 150 °C)."),
        AdhesiveCandidate("3M Scotch-Weld 2214 Hi-Temp", 177, 121, 17, "Not recommended",
                          "Rated to 177 °C but brittle (1 % elongation, 9 N/cm T-peel)."),
        AdhesiveCandidate("3M Scotch-Weld DP460", 60, 23, 19, "Not recommended",
                          "Room-temperature epoxy; T-peel collapses by 82 °C."),
    ]


@dataclass
class AdhesiveInputs:
    candidates: list = field(default_factory=default_adhesives, metadata={
        "label": "Adhesive candidates", "unit": "", "help": "Temperature design!C65:C78", "cell": None, "kind": "input"})
    selected: int = param(1, "-", "Selected adhesive (code)", "1 = AA 326, 2 = EA 9514, 3 = 2214 Hi-Temp, 4 = DP460.",
                          "Temperature design!C75", {1: "AA 326", 2: "EA 9514", 3: "2214 Hi-Temp", 4: "DP460"})


@dataclass
class MismatchInputs:
    ndfeb_cte_per_C: float = param(-0.8e-6, "1/°C", "NdFeB expansion in the bond plane", "Across the magnetization.", "Temperature design!C95")
    adhesive_shear_modulus_GPa: float = param(0.55, "GPa", "Adhesive shear modulus", "", "Temperature design!C96")
    ndfeb_modulus_GPa: float = param(160, "GPa", "NdFeB elastic modulus", "", "Temperature design!C97")
    recommended_bondline_mm: float = param(0.1, "mm", "Recommended bondline", "", "Temperature design!C103")


@dataclass
class SlipLossInputs:
    sigma_316_S_m: float = param(1.35e6, "S/m", "316L conductivity", "", "Temperature design!C111")
    sigma_ndfeb_S_m: float = param(6.7e5, "S/m", "NdFeB conductivity", "", "Temperature design!C113")
    end_factor: float = param(0.7, "-", "End factor for thin shells and the cap", "", "Temperature design!C114")
    b_hub_T: float = param(0.207, "T", "Opposite-ring field at hub steel (fundamental)", "3D, doubled at the steel surface.", "Temperature design!C116")
    b_cup_T: float = param(0.214, "T", "Opposite-ring field at cup steel (fundamental)", "3D.", "Temperature design!C117")
    b_sleeve_T: float = param(0.416, "T", "Opposite-ring field at the inner sleeve (fundamental)", "3D.", "Temperature design!C118")
    b_liner_T: float = param(0.419, "T", "Opposite-ring field at the outer liner (fundamental)", "3D.", "Temperature design!C119")
    cap_integral_T2m4: float = param(5.27e-10, "T²·m⁴", "Cap-face end field, ∫Bz² r² dA", "3D.", "Temperature design!C120")
    web_integral_T2m2: float = param(1.035e-5, "T²·m²", "Rear-web end field, ∫B² dA", "3D.", "Temperature design!C121")
    b_magnet_T: float = param(0.19, "T", "Alternating radial field inside the blocks", "3D.", "Temperature design!C122")
    high_multiplier: float = param(3, "-", "High-case multiplier on the estimate", "Set to 1 once measured.", "Temperature design!C133")


@dataclass
class ThermalInputs:
    c_ndfeb: float = param(440, "J/(kg·K)", "Specific heat, NdFeB", "", "Temperature design!C138")
    c_316: float = param(500, "J/(kg·K)", "Specific heat, 316L", "", "Temperature design!C139")
    c_aluminium: float = param(900, "J/(kg·K)", "Specific heat, aluminium", "", "Temperature design!C140")
    conductance_W_K: float = param(0.3, "W/K", "Thermal conductance to the housing and shafts",
                                   "Both 10 mm shafts plus convection in the sealed housing. Measure it.", "Temperature design!C142")


@dataclass
class AdhesiveLifeInputs:
    hot_strength_retained: float = param(0.5, "-", "Share of lap-shear strength retained at the peak temperature",
                                         "Placeholder; not published for AA 326.", "Temperature design!C194")
    fatigue_endurance: float = param(0.2, "-", "Fatigue endurance at 10^8+ cycles, share of static strength", "", "Temperature design!C195")
    service_years: float = param(10, "years", "Service life", "Placeholder.", "Temperature design!C198")
    daily_swing_C: float = param(30, "°C", "Daily temperature swing at the coupling", "Placeholder.", "Temperature design!C199")


@dataclass
class TemperatureInputs:
    duty: DutyInputs = None
    demag: DemagInputs = None
    adhesive: AdhesiveInputs = None
    mismatch: MismatchInputs = None
    slip_loss: SlipLossInputs = None
    thermal: ThermalInputs = None
    adhesive_life: AdhesiveLifeInputs = None

    def __post_init__(self):
        self.duty = self.duty or DutyInputs()
        self.demag = self.demag or DemagInputs()
        self.adhesive = self.adhesive or AdhesiveInputs()
        self.mismatch = self.mismatch or MismatchInputs()
        self.slip_loss = self.slip_loss or SlipLossInputs()
        self.thermal = self.thermal or ThermalInputs()
        self.adhesive_life = self.adhesive_life or AdhesiveLifeInputs()


@dataclass
class TemperatureLinks:
    """Values the sheet reads from other sheets (filled by api.compute_all)."""
    op_temp_C: float            # Calculator C10
    npole: int                  # Calculator C5
    br20_T: float               # Calculator C21
    alpha_br: float             # Calibration C22
    tmax_lib_C: float           # Calculator C22
    mu0: float                  # Calculator C43
    pullout_op_Nm: float        # Calculator C93
    pullout_20C_Nm: float       # Calculator C94
    inner_back_apothem_mm: float  # Calculator C8
    inner_length_mm: float      # Calculator C18
    inner_width_mm: float       # Calculator C19
    inner_thickness_mm: float   # Calculator C20
    hub_wall_mm: float          # Calculator C38
    active_length_mm: float     # Calculator C33
    outer_back_apothem_mm: float  # Calculator C60
    mass_magnets_g: float       # Calculator C110
    mass_cup_g: float           # Calculator C111
    mass_hub_g: float           # Calculator C112
    mass_boss_g: float          # Calculator C113
    slip_rpm: float             # Metal design C85
    slip_event_s: float         # Metal design C87
    life_events: float          # Metal design C88
    measured_drag_Nm: object    # Metal design C90
    cold_high_Nm: float         # Metal design C10
    required_min_Nm: float      # Metal design C7
    variation: float            # Metal design C18
    min_temp_C: float           # Metal design C16
    magnetic_cycles: float      # Metal design C89
    bond_inner_mm: float        # Metal design C120
    bond_outer_mm: float        # Metal design C121
    sleeve_mm: float            # Metal design C25
    liner_mm: float             # Metal design C26
    sleeve_id_mm: float         # Metal design C175
    sleeve_od_mm: float         # Metal design C176
    liner_od_mm: float          # Metal design C177
    liner_id_mm: float          # Metal design C178
    cap_face_mm: float          # Metal design C167
    hardware_g: float           # Metal design C128
    retainers_g: float          # Metal design C46
    cap_g: float                # Metal design C180
    endplates_g: float          # Metal design C181
    steel_sigma_S_m: float      # Materials C14
    steel_mu_r: float           # Materials C15
    steel_c: float              # Materials C16
    steel_cte: float            # Materials C17
    steel_E_GPa: float          # Materials C18
    al6061_sigma_S_m: float     # Materials C43


# =========================================================================== results
@dataclass
class SummaryResults:
    service_max_C: float = out("°C", "Service maximum magnet temperature", cell="Temperature design!C6")
    onset_aligned_C: float = out("°C", "Demag onset, normal running (rings aligned)", cell="Temperature design!C7")
    onset_pullout_C: float = out("°C", "Demag onset at pull-out", cell="Temperature design!C8")
    onset_skipping_C: float = out("°C", "Demag onset while skipping (like poles facing)", cell="Temperature design!C9")
    magnet_limit_C: float = out("°C", "Magnet design limit", cell="Temperature design!C10")
    adhesive_limit_C: float = out("°C", "Adhesive design limit (selected adhesive)", cell="Temperature design!C11")
    governing_limit_C: float = out("°C", "Governing temperature limit", cell="Temperature design!C12")
    governing_note: str = out("", "Which limit governs", cell="Temperature design!F12")
    margin_service_C: float = out("°C", "Margin above the service maximum", cell="Temperature design!C13")
    hot_day_start_C: float = out("°C", "Hot-day starting magnet temperature", cell="Temperature design!C14")
    margin_hot_day_C: float = out("°C", "Margin above the hot-day start", cell="Temperature design!C15")
    torque_hot_day_Nm: float = out("N·m", "Pull-out torque at the hot-day start (reversible)", cell="Temperature design!C16")
    torque_hot_day_note: str = out("", "Against the requirement", cell="Temperature design!F16")
    steady_estimate_C: float = out("°C", "Continuous slip, steady magnet temp, estimate", cell="Temperature design!C17")
    steady_high_C: float = out("°C", "Continuous slip, steady magnet temp, high case", cell="Temperature design!C18")
    time_to_limit_high: object = out("s", "Unbroken slip time to the limit, high case", cell="Temperature design!C19")
    peak_with_fault_C: float = out("°C", "Peak magnet and bond temperature with the slip fault", cell="Temperature design!C20")
    life_rotations: float = out("rev", "Relative slip rotations over life", cell="Temperature design!C21")
    avg_slip_heating_high_C: float = out("°C", "Average slip heating over life, high case", cell="Temperature design!C22")
    critical_drag_Nm: float = out("N·m", "Slip drag torque that would reach the limit", cell="Temperature design!C23")
    cure_margin_C: float = out("°C", "Cure margin below the single-ring demag onset", cell="Temperature design!C24")
    verdict: str = out("", "Verdict", cell="Temperature design!C25")


@dataclass
class DutyResults:
    slip_rpm: float = out("rpm", "Relative slip speed at the wheel", cell="Temperature design!C28")
    slip_rad_s: float = out("rad/s", "Slip angular speed", cell="Temperature design!C29")
    pole_pairs: float = out("-", "Pole pairs per ring", cell="Temperature design!C30")
    field_freq_Hz: float = out("Hz", "Field frequency seen by the opposite ring", cell="Temperature design!C31")
    field_omega_rad_s: float = out("rad/s", "Field angular frequency", cell="Temperature design!C32")
    slip_event_s: float = out("s", "Slip duration per event", cell="Temperature design!C33")
    hot_day_start_C: float = out("°C", "Hot-day starting magnet temperature", cell="Temperature design!C37")


@dataclass
class DemagResults:
    br20_T: float = out("T", "Remanence at 20 °C", cell="Temperature design!C42")
    alpha_br: float = out("1/°C", "Br temperature coefficient", cell="Temperature design!C43")
    tmax_lib_C: float = out("°C", "Library maximum operating temperature", cell="Temperature design!C47")
    h_ref_kA_m: float = out("kA/m", "Reverse field of a magnet at permeance coefficient 1", cell="Temperature design!C48")
    t_ref_model_C: float = out("°C", "Model onset for that reference magnet", cell="Temperature design!C49")
    calibration_offset_C: float = out("°C", "Calibration offset (model minus library rating)", cell="Temperature design!C50")
    onset_aligned_C: float = out("°C", "Onset, rings aligned", cell="Temperature design!C56")
    onset_pullout_C: float = out("°C", "Onset at pull-out", cell="Temperature design!C57")
    onset_skipping_C: float = out("°C", "Onset while skipping", cell="Temperature design!C58")
    onset_single_ring_C: float = out("°C", "Onset, single ring during an adhesive cure", cell="Temperature design!C59")
    magnet_limit_C: float = out("°C", "Magnet design limit", cell="Temperature design!C60")
    torque_at_limit_Nm: float = out("N·m", "Pull-out torque at the magnet limit (reversible)", cell="Temperature design!C61")
    torque_at_service_Nm: float = out("N·m", "Pull-out torque at the service maximum", cell="Temperature design!C62")


@dataclass
class AdhesiveResults:
    selected_name: str = out("", "Selected adhesive")
    design_limit_C: float = out("°C", "Selected adhesive design limit", cell="Temperature design!C76")
    cure_C: float = out("°C", "Selected adhesive cure (stress-free) temperature", cell="Temperature design!C77")
    lap_shear_MPa: float = out("MPa", "Selected adhesive lap shear at 22 °C (TDS)", cell="Temperature design!C78")
    bond_area_mm2: float = out("mm²", "Bond area per block (back face)", cell="Temperature design!C81")
    block_mass_g: float = out("g", "Block mass", cell="Temperature design!C82")
    cold_high_torque_Nm: float = out("N·m", "Highest pull-out torque (cold, +variation)", cell="Temperature design!C83")
    inner_mid_radius_mm: float = out("mm", "Inner block mid radius", cell="Temperature design!C84")
    tangential_force_N: float = out("N", "Tangential force per inner block at pull-out", cell="Temperature design!C85")
    bond_shear_MPa: float = out("MPa", "Bond shear stress from magnetic torque", cell="Temperature design!C86")
    centrifugal_force_N: float = out("N", "Centrifugal force per inner block at wheel speed", cell="Temperature design!C88")
    static_ratio: float = out("-", "Static strength ratio at 22 °C", cell="Temperature design!C89")
    shear_reversals: float = out("cycles", "Shear reversals over life while slipping", cell="Temperature design!C90")
    fatigue_screen: str = out("", "Fatigue screen", cell="Temperature design!C91")


@dataclass
class MismatchResults:
    steel_cte: float = out("1/°C", "Steel expansion coefficient (4140)", cell="Temperature design!C94")
    steel_E_GPa: float = out("GPa", "Steel elastic modulus (4140)", cell="Temperature design!C98")
    steel_thickness_mm: float = out("mm", "Steel thickness under the inner blocks", cell="Temperature design!C99")
    cold_limit_C: float = out("°C", "Cold limit", cell="Temperature design!C100")
    worst_swing_C: float = out("°C", "Worst swing from the stress-free (cure) temperature", cell="Temperature design!C101")
    current_bondline_mm: float = out("mm", "Current bondline", cell="Temperature design!C102")
    peak_shear_current_MPa: float = out("MPa", "Peak end shear, current bondline", cell="Temperature design!C104")
    peak_shear_recommended_MPa: float = out("MPa", "Peak end shear, recommended bondline", cell="Temperature design!C105")
    reading: str = out("", "Reading", cell="Temperature design!C106")


@dataclass
class SlipLossResults:
    steel_sigma_S_m: float = out("S/m", "Steel conductivity (4140)", cell="Temperature design!C109")
    steel_mu_r: float = out("-", "Steel incremental relative permeability (4140)", cell="Temperature design!C110")
    cap_sigma_S_m: float = out("S/m", "Aluminium cap conductivity (6061-T6)", cell="Temperature design!C112")
    skin_depth_mm: float = out("mm", "Steel skin depth at the field frequency", cell="Temperature design!C115")
    hub_W: float = out("W", "Hub surface (solid steel)", cell="Temperature design!C123")
    cup_W: float = out("W", "Cup surface (solid steel)", cell="Temperature design!C124")
    web_W: float = out("W", "Rear web (solid steel)", cell="Temperature design!C125")
    sleeve_W: float = out("W", "Inner 316L sleeve", cell="Temperature design!C126")
    liner_W: float = out("W", "Outer 316L liner", cell="Temperature design!C127")
    cap_W: float = out("W", "Aluminium cap face", cell="Temperature design!C128")
    magnets_W: float = out("W", "Magnet eddy currents (both rings)", cell="Temperature design!C129")
    total_W: float = out("W", "Total estimated slip loss", cell="Temperature design!C130")
    drag_Nm: float = out("N·m", "Equivalent mean drag torque", cell="Temperature design!C131")
    used_W: float = out("W", "Loss used below", "Bench drag replaces the estimate once entered.", "Temperature design!C132")
    high_W: float = out("W", "Loss, high case", cell="Temperature design!C134")


@dataclass
class ThermalResults:
    steel_c: float = out("J/(kg·K)", "Specific heat, steel (4140)", cell="Temperature design!C137")
    heat_capacity_J_K: float = out("J/K", "Heat capacity of the rotating coupling", cell="Temperature design!C141")
    time_constant_s: float = out("s", "Thermal time constant", cell="Temperature design!C143")
    start_C: float = out("°C", "Starting magnet temperature", cell="Temperature design!C144")
    rise_per_event_C: float = out("°C", "Temperature rise per slip event", cell="Temperature design!C145")
    steady_rise_est_C: float = out("°C", "Continuous slip: steady rise, estimate", cell="Temperature design!C146")
    steady_rise_high_C: float = out("°C", "Continuous slip: steady rise, high case", cell="Temperature design!C147")
    steady_est_C: float = out("°C", "Continuous slip, steady magnet temp, estimate", cell="Temperature design!C148")
    steady_high_C: float = out("°C", "Continuous slip, steady magnet temp, high case", cell="Temperature design!C149")
    time_to_limit_high: object = out("s", "Continuous slip, time to the limit (high case)", cell="Temperature design!C150")
    rotations_to_limit_high: object = out("rev", "Continuous slip, rotations to the limit (high case)", cell="Temperature design!C151")
    time_to_limit_est: object = out("s", "Continuous slip, time to the limit (estimate)", cell="Temperature design!C152")
    critical_drag_Nm: float = out("N·m", "Slip drag torque that would reach the limit", cell="Temperature design!C153")
    heating_rate_est_C_s: float = out("°C/s", "Initial heating rate, estimate", cell="Temperature design!C154")
    heating_rate_high_C_s: float = out("°C/s", "Initial heating rate, high case", cell="Temperature design!C155")
    rev_per_C_est: float = out("rev/°C", "Relative rotations per °C at the start, estimate", cell="Temperature design!C156")
    rev_per_C_high: float = out("rev/°C", "Relative rotations per °C at the start, high case", cell="Temperature design!C157")
    rev_per_tau: float = out("rev", "Relative rotations per thermal time constant", cell="Temperature design!C158")
    t95_s: float = out("s", "Time to 95 % of the steady rise", cell="Temperature design!C159")
    rev95: float = out("rev", "Relative rotations to 95 % of the steady rise", cell="Temperature design!C160")
    temp_at_fault_C: float = out("°C", "Magnet temperature at the fault trip time, high case", cell="Temperature design!C161")


@dataclass
class SlipLifeResults:
    events: float = out("events", "Slip events over life", cell="Temperature design!C164")
    rev_per_event: float = out("rev", "Relative rotations per event", cell="Temperature design!C165")
    rotations: float = out("rev", "Relative rotations over life", cell="Temperature design!C166")
    slip_hours: float = out("h", "Total slip time over life", cell="Temperature design!C167")
    like_pole_passes: float = out("passes", "Like-pole passes per magnet over life", cell="Temperature design!C168")
    heat_per_event_est_J: float = out("J", "Heat per event, estimate", cell="Temperature design!C169")
    heat_per_event_high_J: float = out("J", "Heat per event, high case", cell="Temperature design!C170")
    rise_per_event_est_C: float = out("°C", "Temperature rise per event, estimate", cell="Temperature design!C171")
    rise_per_event_high_C: float = out("°C", "Temperature rise per event, high case", cell="Temperature design!C172")
    life_heat_high_MJ: float = out("MJ", "Total slip heat over life, high case", cell="Temperature design!C173")
    slip_duty: float = out("-", "Slip duty: share of operating time spent slipping", cell="Temperature design!C174")
    avg_rise_est_C: float = out("°C", "Average temperature rise from slip, estimate", cell="Temperature design!C175")
    avg_rise_high_C: float = out("°C", "Average temperature rise from slip, high case", cell="Temperature design!C176")
    rise_per_pct_duty_C: float = out("°C", "Average rise per 1 % of time slipping, high case", cell="Temperature design!C177")


@dataclass
class MagnetLifeResults:
    peak_C: float = out("°C", "Peak magnet temperature: hot day plus fault-limited slip", cell="Temperature design!C180")
    margin_onset_C: float = out("°C", "Margin to the skipping onset", cell="Temperature design!C181")
    margin_limit_C: float = out("°C", "Margin to the magnet design limit", cell="Temperature design!C182")
    torque_hot_day_Nm: float = out("N·m", "Pull-out torque at the hot-day start (reversible)", cell="Temperature design!C184")
    torque_hot_day_check: str = out("", "Against the requirement", cell="Temperature design!C185")
    torque_peak_Nm: float = out("N·m", "Pull-out torque at the peak temperature (reversible)", cell="Temperature design!C186")


@dataclass
class AdhesiveLifeResults:
    peak_C: float = out("°C", "Peak bond temperature", cell="Temperature design!C189")
    margin_C: float = out("°C", "Margin to the adhesive design limit", cell="Temperature design!C190")
    reversals: float = out("cycles", "Shear reversals over life", cell="Temperature design!C191")
    torque_peak_var_Nm: float = out("N·m", "Pull-out torque at the peak temperature, with +variation", cell="Temperature design!C192")
    shear_amplitude_MPa: float = out("MPa", "Shear stress amplitude per reversal, hot", cell="Temperature design!C193")
    hot_fatigue_margin: float = out("x", "Hot fatigue margin", cell="Temperature design!C196")
    hot_fatigue_screen: str = out("", "Hot fatigue screen", cell="Temperature design!C197")
    daily_cycles: float = out("cycles", "Daily thermal cycles over life", cell="Temperature design!C200")
    daily_peak_shear_MPa: float = out("MPa", "Peak end shear per daily cycle, recommended bondline", cell="Temperature design!C201")
    daily_screen: str = out("", "Daily-cycle screen", cell="Temperature design!C202")


@dataclass
class TemperatureResults:
    summary: SummaryResults
    duty: DutyResults
    demag: DemagResults
    adhesive: AdhesiveResults
    mismatch: MismatchResults
    slip_loss: SlipLossResults
    thermal: ThermalResults
    slip_life: SlipLifeResults
    magnet_life: MagnetLifeResults
    adhesive_life: AdhesiveLifeResults


# =========================================================================== model
def demag_onset_C(h_rev_kA_m: float, hcj20: float, beta: float, knee: float, alpha_br: float, offset_C: float = 0.0) -> float:
    """Temperature where the reverse field (scaling with Br) reaches the knee of the Hcj curve."""
    hk = knee * hcj20
    return 20 + (hk - h_rev_kA_m) / (hk * abs(beta) - h_rev_kA_m * abs(alpha_br)) - offset_C


def volkersen_peak_shear_MPa(G_GPa: float, d_alpha: float, dT: float, bondline_mm: float, magnet_E_GPa: float,
                             magnet_t_mm: float, steel_E_GPa: float, steel_t_mm: float, bond_length_mm: float) -> float:
    """Linear-elastic shear-lag peak shear at the ends of a bonded block from thermal mismatch."""
    G = G_GPa * 1e9
    lam = math.sqrt(G / (bondline_mm / 1000) * (1 / (magnet_E_GPa * 1e9 * magnet_t_mm / 1000) + 1 / (steel_E_GPa * 1e9 * steel_t_mm / 1000)))
    return G * d_alpha * dT * math.tanh(lam * bond_length_mm / 2000) / (lam * bondline_mm / 1000) / 1e6


def compute(ti: TemperatureInputs, k: TemperatureLinks) -> TemperatureResults:
    # ---- duty
    omega = k.slip_rpm * 2 * math.pi / 60
    pp = k.npole / 2
    f = pp * k.slip_rpm / 60
    we = 2 * math.pi * f
    T0 = ti.duty.hot_ambient_C + ti.duty.driving_rise_C
    duty = DutyResults(k.slip_rpm, omega, pp, f, we, k.slip_event_s, T0)

    # ---- demagnetization
    d = ti.demag
    h_ref = k.br20_T / (2 * k.mu0) / 1000
    t_ref = demag_onset_C(h_ref, d.hcj20_kA_m, d.beta_hcj_per_C, d.knee_fraction, k.alpha_br)
    # the workbook errors out when the magnet is not in the library; here the onset is left uncalibrated instead
    offset = t_ref - k.tmax_lib_C if isinstance(k.tmax_lib_C, (int, float)) else 0.0
    on = lambda h: demag_onset_C(h, d.hcj20_kA_m, d.beta_hcj_per_C, d.knee_fraction, k.alpha_br, offset)
    on_al, on_po, on_lp, on_cu = on(d.h_rev_aligned_kA_m), on(d.h_rev_pullout_kA_m), on(d.h_rev_likepole_kA_m), on(d.h_rev_single_ring_kA_m)
    mag_lim = on_lp - d.design_margin_C
    thf = lambda T: (1 + k.alpha_br * (T - 20)) ** 2
    demag = DemagResults(k.br20_T, k.alpha_br, k.tmax_lib_C, h_ref, t_ref, offset, on_al, on_po, on_lp, on_cu, mag_lim,
                         k.pullout_20C_Nm * thf(mag_lim), k.pullout_op_Nm)

    # ---- adhesive selection and loads
    sel = ti.adhesive.candidates[ti.adhesive.selected - 1]
    area = k.inner_length_mm * k.inner_width_mm
    m_block = k.inner_length_mm * k.inner_width_mm * k.inner_thickness_mm * 0.0075
    r_mid = k.inner_back_apothem_mm + k.inner_thickness_mm / 2
    Ft = k.cold_high_Nm / (k.npole * r_mid / 1000)
    tau_b = Ft / area
    Fc = m_block / 1000 * (ti.duty.wheel_rotor_rpm * 2 * math.pi / 60) ** 2 * r_mid / 1000
    fat = 0.2 * sel.lap_shear_MPa / tau_b
    adh = AdhesiveResults(sel.name, sel.design_limit_C, sel.cure_C, sel.lap_shear_MPa, area, m_block, k.cold_high_Nm, r_mid, Ft,
                          tau_b, Fc, sel.lap_shear_MPa / tau_b, k.magnetic_cycles,
                          f"OK: {_text0(fat)}x margin" if fat >= 4 else "CHECK")

    gov = min(mag_lim, sel.design_limit_C)

    # ---- thermal mismatch screen
    mm = ti.mismatch
    dT = max(sel.cure_C - k.min_temp_C, gov - sel.cure_C)
    d_alpha = k.steel_cte - mm.ndfeb_cte_per_C
    s1 = volkersen_peak_shear_MPa(mm.adhesive_shear_modulus_GPa, d_alpha, dT, k.bond_inner_mm, mm.ndfeb_modulus_GPa,
                                  k.inner_thickness_mm, k.steel_E_GPa, k.hub_wall_mm, k.inner_length_mm)
    s2 = volkersen_peak_shear_MPa(mm.adhesive_shear_modulus_GPa, d_alpha, dT, mm.recommended_bondline_mm, mm.ndfeb_modulus_GPa,
                                  k.inner_thickness_mm, k.steel_E_GPa, k.hub_wall_mm, k.inner_length_mm)
    mis = MismatchResults(k.steel_cte, k.steel_E_GPa, k.hub_wall_mm, k.min_temp_C, dT, k.bond_inner_mm, s1, s2,
                          "Above the lap-shear strength at the block ends" if s1 > sel.lap_shear_MPa else "Below the lap-shear strength")

    # ---- slip losses (estimates)
    sl = ti.slip_loss
    delta = math.sqrt(2 / (we * k.mu0 * k.steel_mu_r * k.steel_sigma_S_m)) * 1000       # mm
    L = k.active_length_mm / 1000
    r_hub = (k.inner_back_apothem_mm - k.bond_inner_mm) / 1000
    r_cup = (k.outer_back_apothem_mm + k.bond_outer_mm) / 1000
    surface = lambda B, r: k.steel_sigma_S_m * we ** 2 * B ** 2 * (delta / 1000) / (4 * (pp / r) ** 2) * 2 * math.pi * r * L
    p_hub = surface(sl.b_hub_T, r_hub)
    p_cup = surface(sl.b_cup_T, r_cup)
    p_web = k.steel_sigma_S_m * we ** 2 * (delta / 1000) / 4 * ((r_mid / 1000) / pp) ** 2 * sl.web_integral_T2m2
    r_s = (k.sleeve_id_mm + k.sleeve_od_mm) / 4 / 1000
    r_l = (k.liner_od_mm + k.liner_id_mm) / 4 / 1000
    shell = lambda t_mm, r, B: sl.end_factor * sl.sigma_316_S_m * (t_mm / 1000) * (omega * r) ** 2 * B ** 2 / 2 * 2 * math.pi * r * L
    p_slv = shell(k.sleeve_mm, r_s, sl.b_sleeve_T)
    p_lin = shell(k.liner_mm, r_l, sl.b_liner_T)
    p_cap = sl.end_factor * k.al6061_sigma_S_m * (k.cap_face_mm / 1000) * omega ** 2 * sl.cap_integral_T2m4
    p_mag = (sl.sigma_ndfeb_S_m * we ** 2 * sl.b_magnet_T ** 2 * (k.inner_width_mm / 1000) ** 2 / 24
             * (k.inner_length_mm * k.inner_width_mm * k.inner_thickness_mm * 1e-9) * 2 * k.npole)
    p_tot = p_hub + p_cup + p_web + p_slv + p_lin + p_cap + p_mag
    measured = isinstance(k.measured_drag_Nm, (int, float))
    p_use = k.measured_drag_Nm * omega if measured else p_tot
    p_hi = p_use if measured else p_use * sl.high_multiplier
    loss = SlipLossResults(k.steel_sigma_S_m, k.steel_mu_r, k.al6061_sigma_S_m, delta, p_hub, p_cup, p_web, p_slv, p_lin,
                           p_cap, p_mag, p_tot, p_tot / omega, p_use, p_hi)

    # ---- thermal network
    th = ti.thermal
    C = (k.mass_magnets_g * th.c_ndfeb + (k.mass_cup_g + k.mass_hub_g + k.mass_boss_g + k.hardware_g) * k.steel_c
         + (k.retainers_g + k.endplates_g) * th.c_316 + k.cap_g * th.c_aluminium) / 1000
    G = th.conductance_W_K
    tau_th = C / G
    rise_e, rise_h = p_use / G, p_hi / G
    Te, Th = T0 + rise_e, T0 + rise_h
    never = "never: steady state stays below the limit"
    t_lim_h = never if Th <= gov else -tau_th * math.log(1 - (gov - T0) / rise_h)
    t_lim_e = never if Te <= gov else -tau_th * math.log(1 - (gov - T0) / rise_e)
    rev_s = k.slip_rpm / 60
    ke, kh = p_use / C, p_hi / C
    T_fault = T0 + rise_h * (1 - math.exp(-ti.duty.fault_trip_s / tau_th))
    thermal = ThermalResults(k.steel_c, C, tau_th, T0, p_use * k.slip_event_s / C, rise_e, rise_h, Te, Th, t_lim_h,
                             t_lim_h * rev_s if isinstance(t_lim_h, float) else "never", t_lim_e,
                             (gov - T0) * G / omega, ke, kh, rev_s / ke, rev_s / kh, tau_th * rev_s, 3 * tau_th,
                             3 * tau_th * rev_s, T_fault)

    # ---- slip life
    rpe = rev_s * k.slip_event_s
    rot = k.life_events * rpe
    hrs = k.life_events * k.slip_event_s / 3600
    Ee, Eh = p_use * k.slip_event_s, p_hi * k.slip_event_s
    dTe, dTh = Ee / C, Eh / C
    duty_frac = hrs / ti.duty.life_hours
    life = SlipLifeResults(k.life_events, rpe, rot, hrs, rot * pp, Ee, Eh, dTe, dTh, k.life_events * Eh / 1e6, duty_frac,
                           duty_frac * rise_e, duty_frac * rise_h, 0.01 * rise_h)

    # ---- magnet life
    peak = max(T_fault, T0 + dTh) + duty_frac * rise_h
    tq_hot = k.pullout_20C_Nm * thf(T0)
    tq_peak = k.pullout_20C_Nm * thf(peak)
    mlife = MagnetLifeResults(peak, on_lp - peak, mag_lim - peak, tq_hot,
                              "Meets it nominally (no variation allowance)" if tq_hot >= k.required_min_Nm else "Below it", tq_peak)

    # ---- adhesive life
    al = ti.adhesive_life
    tq_var = tq_peak * (1 + k.variation)
    amp = tq_var / (k.npole * r_mid / 1000) / area
    hot_fm = sel.lap_shear_MPa * al.hot_strength_retained * al.fatigue_endurance / amp
    daily = s2 * al.daily_swing_C / dT
    alife = AdhesiveLifeResults(peak, sel.design_limit_C - peak, rot * pp, tq_var, amp, hot_fm,
                                "OK" if hot_fm >= 4 else "CHECK: get hot fatigue data", al.service_years * 365, daily,
                                "Below the fatigue endurance" if daily < sel.lap_shear_MPa * al.fatigue_endurance
                                else "Above the fatigue endurance: qualify by thermal cycling")

    # ---- summary
    cure_margin = on_cu - sel.cure_C
    margin_hot = gov - T0
    ok = margin_hot > 0 and (mag_lim - peak) > 0 and (sel.design_limit_C - peak) > 0 and cure_margin >= 10
    summary = SummaryResults(
        service_max_C=k.op_temp_C, onset_aligned_C=on_al, onset_pullout_C=on_po, onset_skipping_C=on_lp, magnet_limit_C=mag_lim,
        adhesive_limit_C=sel.design_limit_C, governing_limit_C=gov,
        governing_note="Magnets govern (skipping case)." if mag_lim <= sel.design_limit_C else "Adhesive governs.",
        margin_service_C=gov - k.op_temp_C, hot_day_start_C=T0, margin_hot_day_C=margin_hot, torque_hot_day_Nm=tq_hot,
        torque_hot_day_note=mlife.torque_hot_day_check, steady_estimate_C=Te, steady_high_C=Th, time_to_limit_high=t_lim_h,
        peak_with_fault_C=peak, life_rotations=rot, avg_slip_heating_high_C=duty_frac * rise_h,
        critical_drag_Nm=(gov - T0) * G / omega, cure_margin_C=cure_margin,
        verdict="OK on temperature. Confirm drag torque and thermal cycling by test." if ok else "CHECK: see the rows above.",
    )
    return TemperatureResults(summary, duty, demag, adh, mis, loss, thermal, life, mlife, alife)
