//! Temperature design sheet ('Temperature design').
//!
//! Port of `reference/magcoupling-py/magcoupling/temperature.py`. Answers how hot
//! the coupling can get before the magnets demagnetize or the bond gives up, how
//! much heat slip makes and how long it can slip: the demagnetization onsets
//! (calibrated to the library rating), the adhesive limit and load, a Volkersen
//! shear-lag screen for the thermal mismatch, eddy-current slip losses, one
//! thermal RC network, and the life checks.
//!
//! This module holds the seven input groups ([`TemperatureInputs`]), the adhesive
//! table ([`ADHESIVES`]: the Python `candidates` list has metadata but no cell
//! and is not in `input_schema()`, so it is static data, not an input), the
//! values the sheet reads from other sheets ([`TemperatureLinks`]), the ten
//! result groups ([`TemperatureResults`], 130 result cells and the uncelled
//! adhesive name), the two helpers ([`demag_onset_C`],
//! [`volkersen_peak_shear_MPa`]) and [`compute`].
//!
//! A selector code outside 1 to 4 selects no adhesive (decision D3): Python's
//! `candidates[selected - 1]` would silently take the last row for code 0.
//!
//! Deviations touching this sheet (see
//! [`crate::engine::deviations::REGISTRY`]): applied E1 (Temperature design!C96)
//! and E5 (C121), both corrected defaults; planned E9, E11 (C91), E12 (C19,
//! C23, C150 to C153), E13 (C156, C157).

use std::f64::consts::PI;

use super::compat::{py_max, py_min, text0};
use super::deviations::Deviations;
use super::meta::{NumOrText, inputs, out, out_uncelled, param, results};

// =========================================================================== inputs
inputs! {
    /// Duty and ambient inputs (Temperature design!C34:C39).
    pub struct DutyInputs {
        fields {
            wheel_rotor_rpm: f64 = 2000.0 => param("rpm", "Wheel-side (inner) rotor maximum speed",
                "Used for the centrifugal load.", "Temperature design!C34")
                .range(100.0, 6000.0, 10.0),
            hot_ambient_C: f64 = 55.0 => param("°C", "Hot-day ambient temperature",
                "User: 55 °C day.", "Temperature design!C35")
                .range(-20.0, 80.0, 0.5),
            driving_rise_C: f64 = 10.0 => param("°C", "Coupling rise above ambient while driving (no slip)",
                "Placeholder: housing air, sun and gearbox heat. Measure it.", "Temperature design!C36")
                .range(0.0, 60.0, 0.5)
                .assumption(), // spans the E12 boundary (37.55 °C at defaults)
            fault_trip_s: f64 = 2.0 => param("s", "Slip fault trip time (unbroken slip)",
                "", "Temperature design!C38")
                .range(0.1, 60.0, 0.1)
                .log(),
            life_hours: f64 = 20000.0 => param("h", "Operating hours over the system life",
                "Placeholder; used for the average slip duty.", "Temperature design!C39")
                .range(1000.0, 100000.0, 100.0), // > 0: divides the slip duty
        }
    }
}

inputs! {
    /// Demagnetization inputs (Temperature design!C44:C55).
    pub struct DemagInputs {
        fields {
            hcj20_kA_m: f64 = 1592.0 => param("kA/m", "Intrinsic coercivity Hcj at 20 °C (grade minimum)",
                "N42SH ≥ 20 kOe.", "Temperature design!C44")
                .range(800.0, 3000.0, 1.0),
            beta_hcj_per_C: f64 = -0.005 => param("1/°C", "Hcj temperature coefficient (effective, 20–150 °C)",
                "", "Temperature design!C45")
                .range(-0.008, -0.001, 0.0001)
                .assumption(),
            knee_fraction: f64 = 0.9 => param("-", "Knee field as a fraction of Hcj",
                "", "Temperature design!C46")
                .range(0.5, 1.0, 0.01)
                .assumption(),
            design_margin_C: f64 = 10.0 => param("°C", "Design margin below the onset",
                "", "Temperature design!C51")
                .range(0.0, 40.0, 0.5)
                .assumption(),
            h_rev_aligned_kA_m: f64 = 354.0 => param("kA/m", "3D worst reverse field, rings aligned",
                "Outer blocks (inner 341).", "Temperature design!C52")
                .range(0.0, 1500.0, 1.0),
            h_rev_pullout_kA_m: f64 = 791.0 => param("kA/m", "3D worst reverse field at pull-out",
                "Outer blocks (inner 760).", "Temperature design!C53")
                .range(0.0, 1500.0, 1.0),
            h_rev_likepole_kA_m: f64 = 863.0 => param("kA/m", "3D worst reverse field, like poles facing",
                "Outer blocks (inner 844); once per pole pass while skipping.", "Temperature design!C54")
                .range(0.0, 1500.0, 1.0),
            h_rev_single_ring_kA_m: f64 = 569.0 => param("kA/m", "3D worst reverse field, single ring on its carrier",
                "Adhesive-cure case (inner ring alone: 545).", "Temperature design!C55")
                .range(0.0, 1500.0, 1.0),
        }
    }
}

/// One adhesive of the Temperature design!C65:C78 table (Python `AdhesiveCandidate`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct AdhesiveCandidate {
    pub name: &'static str,
    /// TDS service maximum or Tg − 20 °C [°C].
    pub design_limit_C: f64,
    /// Stress-free (cure) temperature [°C].
    pub cure_C: f64,
    /// TDS lap shear at about 22 °C [MPa].
    pub lap_shear_MPa: f64,
    pub role: &'static str,
    pub note: &'static str,
}

pub const ADHESIVES: [AdhesiveCandidate; 4] = [
    AdhesiveCandidate {
        name: "Loctite AA 326 + SF 7649",
        design_limit_C: 120.0,
        cure_C: 22.0,
        lap_shear_MPa: 15.0,
        role: "Recommended",
        note: "No-mix acrylic for magnet bonding; room-temperature cure; 0.10 mm bondline; service to 120 °C.",
    },
    AdhesiveCandidate {
        name: "Loctite EA 9514",
        design_limit_C: 113.0,
        cure_C: 120.0,
        lap_shear_MPa: 45.0,
        role: "Alternative: more hot strength",
        note: "One-part toughened heat-cure epoxy; Tg 133 °C; cure 60 min at 120 °C (not 150 °C).",
    },
    AdhesiveCandidate {
        name: "3M Scotch-Weld 2214 Hi-Temp",
        design_limit_C: 177.0,
        cure_C: 121.0,
        lap_shear_MPa: 17.0,
        role: "Not recommended",
        note: "Rated to 177 °C but brittle (1 % elongation, 9 N/cm T-peel).",
    },
    AdhesiveCandidate {
        name: "3M Scotch-Weld DP460",
        design_limit_C: 60.0,
        cure_C: 23.0,
        lap_shear_MPa: 19.0,
        role: "Not recommended",
        note: "Room-temperature epoxy; T-peel collapses by 82 °C.",
    },
];

/// What a selector code outside 1..=4 selects (decision D3). Python indexes
/// `candidates[selected - 1]`, so 0 silently picks the LAST adhesive; the port
/// never picks another adhesive: the results are NaN and the name says so.
pub const NO_ADHESIVE: AdhesiveCandidate = AdhesiveCandidate {
    name: "#N/A",
    design_limit_C: f64::NAN,
    cure_C: f64::NAN,
    lap_shear_MPa: f64::NAN,
    role: "",
    note: "",
};

/// The adhesive a selector code selects.
pub fn selected_adhesive(code: i64) -> &'static AdhesiveCandidate {
    match code {
        1..=4 => &ADHESIVES[(code - 1) as usize],
        _ => &NO_ADHESIVE,
    }
}

inputs! {
    /// The adhesive choice (Temperature design!C75). The candidate table is [`ADHESIVES`].
    pub struct AdhesiveInputs {
        fields {
            selected: i64 = 1 => param("-", "Selected adhesive (code)",
                "1 = AA 326, 2 = EA 9514, 3 = 2214 Hi-Temp, 4 = DP460.", "Temperature design!C75")
                .choices(&[(1, "AA 326"), (2, "EA 9514"), (3, "2214 Hi-Temp"), (4, "DP460")]),
        }
    }
}

inputs! {
    /// Thermal mismatch inputs (Temperature design!C95:C103).
    pub struct MismatchInputs {
        fields {
            ndfeb_cte_per_C: f64 = -0.8e-6 => param("1/°C", "NdFeB expansion in the bond plane",
                "Across the magnetization.", "Temperature design!C95")
                .range(-3e-6, 6e-6, 1e-8),
            adhesive_shear_modulus_GPa: f64 = 0.107 => param("GPa", "Adhesive shear modulus",
                "Loctite AA 326 + SF 7649 (Henkel TDS, Aug-2020): tensile modulus 0.300 GPa, so G = E/(2(1+ν)) ≈ 0.107 GPa at ν = 0.4. Correction E1: the workbook's 0.55 GPa is the modulus of EA 9514.",
                "Temperature design!C96")
                .range(0.01, 3.0, 0.001)
                .log(), // > 0: square root in the Volkersen lambda
            ndfeb_modulus_GPa: f64 = 160.0 => param("GPa", "NdFeB elastic modulus",
                "", "Temperature design!C97")
                .range(100.0, 200.0, 1.0),
            recommended_bondline_mm: f64 = 0.1 => param("mm", "Recommended bondline",
                "", "Temperature design!C103")
                .range(0.01, 0.5, 0.005), // > 0: divides
        }
    }
}

inputs! {
    /// Slip loss inputs (Temperature design!C111:C133).
    pub struct SlipLossInputs {
        fields {
            sigma_316_S_m: f64 = 1.35e6 => param("S/m", "316L conductivity",
                "", "Temperature design!C111")
                .range(1e5, 1e7, 1000.0)
                .log(),
            sigma_ndfeb_S_m: f64 = 6.7e5 => param("S/m", "NdFeB conductivity",
                "", "Temperature design!C113")
                .range(1e5, 2e6, 1000.0)
                .log(),
            end_factor: f64 = 0.7 => param("-", "End factor for thin shells and the cap",
                "", "Temperature design!C114")
                .range(0.0, 1.0, 0.01),
            b_hub_T: f64 = 0.207 => param("T", "Opposite-ring field at hub steel (fundamental)",
                "3D, doubled at the steel surface.", "Temperature design!C116")
                .range(0.0, 1.0, 0.001),
            b_cup_T: f64 = 0.214 => param("T", "Opposite-ring field at cup steel (fundamental)",
                "3D.", "Temperature design!C117")
                .range(0.0, 1.0, 0.001),
            b_sleeve_T: f64 = 0.416 => param("T", "Opposite-ring field at the inner sleeve (fundamental)",
                "3D.", "Temperature design!C118")
                .range(0.0, 1.0, 0.001),
            b_liner_T: f64 = 0.419 => param("T", "Opposite-ring field at the outer liner (fundamental)",
                "3D.", "Temperature design!C119")
                .range(0.0, 1.0, 0.001),
            cap_integral_T2m4: f64 = 5.27e-10 => param("T²·m⁴", "Cap-face end field, ∫Bz² r² dA",
                "3D.", "Temperature design!C120")
                .range(1e-12, 1e-8, 1e-13)
                .log(),
            web_integral_T2m2: f64 = 4.14e-5 => param("T²·m²", "Rear-web end field, ∫B² dA",
                "3D, doubled at the steel surface (correction E5: the workbook's 1.035e-5 T²·m² is the free-space field, which made the web loss 4 times too low).",
                "Temperature design!C121")
                .range(1e-7, 1e-3, 1e-8)
                .log(), // the workbook's 1.035e-5 is inside too
            b_magnet_T: f64 = 0.19 => param("T", "Alternating radial field inside the blocks",
                "3D.", "Temperature design!C122")
                .range(0.0, 1.0, 0.001),
            high_multiplier: f64 = 3.0 => param("-", "High-case multiplier on the estimate",
                "Set to 1 once measured.", "Temperature design!C133")
                .range(1.0, 10.0, 0.1),
        }
    }
}

inputs! {
    /// Thermal network inputs (Temperature design!C138:C142).
    pub struct ThermalInputs {
        fields {
            c_ndfeb: f64 = 440.0 => param("J/(kg·K)", "Specific heat, NdFeB",
                "", "Temperature design!C138")
                .range(300.0, 600.0, 1.0),
            c_316: f64 = 500.0 => param("J/(kg·K)", "Specific heat, 316L",
                "", "Temperature design!C139")
                .range(300.0, 700.0, 1.0),
            c_aluminium: f64 = 900.0 => param("J/(kg·K)", "Specific heat, aluminium",
                "", "Temperature design!C140")
                .range(700.0, 1000.0, 1.0),
            conductance_W_K: f64 = 0.3 => param("W/K", "Thermal conductance to the housing and shafts",
                "Both 10 mm shafts plus convection in the sealed housing. Measure it.", "Temperature design!C142")
                .range(0.01, 5.0, 0.001)
                .log()
                .assumption(), // > 0: divides
        }
    }
}

inputs! {
    /// Adhesive life inputs (Temperature design!C194:C199).
    pub struct AdhesiveLifeInputs {
        fields {
            hot_strength_retained: f64 = 0.5 => param("-", "Share of lap-shear strength retained at the peak temperature",
                "Placeholder; not published for AA 326.", "Temperature design!C194")
                .range(0.05, 1.0, 0.01),
            fatigue_endurance: f64 = 0.2 => param("-", "Fatigue endurance at 10^8+ cycles, share of static strength",
                "", "Temperature design!C195")
                .range(0.02, 0.6, 0.005), // spans the E11 flip (0.106)
            service_years: f64 = 10.0 => param("years", "Service life",
                "Placeholder.", "Temperature design!C198")
                .range(1.0, 40.0, 0.5),
            daily_swing_C: f64 = 30.0 => param("°C", "Daily temperature swing at the coupling",
                "Placeholder.", "Temperature design!C199")
                .range(0.0, 100.0, 0.5),
        }
    }
}

inputs! {
    /// Every Temperature design input, grouped as the Python `TemperatureInputs`.
    pub struct TemperatureInputs {
        fields {}
        groups {
            duty: DutyInputs,
            demag: DemagInputs,
            adhesive: AdhesiveInputs,
            mismatch: MismatchInputs,
            slip_loss: SlipLossInputs,
            thermal: ThermalInputs,
            adhesive_life: AdhesiveLifeInputs,
        }
    }
}

/// Values the Temperature design sheet reads from other sheets (Python
/// `TemperatureLinks`), filled by `api::compute` exactly as Python's `compute_all`.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct TemperatureLinks {
    pub op_temp_C: f64,                // Calculator C10
    pub npole: i64,                    // Calculator C5
    pub br20_T: f64,                   // Calculator C21
    pub alpha_br: f64,                 // Calibration C22
    pub tmax_lib_C: NumOrText,         // Calculator C22 ("n/a" for manual magnets)
    pub mu0: f64,                      // Calculator C43
    pub pullout_op_Nm: f64,            // Calculator C93
    pub pullout_20C_Nm: f64,           // Calculator C94
    pub inner_back_apothem_mm: f64,    // Calculator C8
    pub inner_length_mm: f64,          // Calculator C18
    pub inner_width_mm: f64,           // Calculator C19
    pub inner_thickness_mm: f64,       // Calculator C20
    pub hub_wall_mm: f64,              // Calculator C38
    pub active_length_mm: f64,         // Calculator C33
    pub outer_back_apothem_mm: f64,    // Calculator C60
    pub mass_magnets_g: f64,           // Calculator C110
    pub mass_cup_g: f64,               // Calculator C111
    pub mass_hub_g: f64,               // Calculator C112
    pub mass_boss_g: f64,              // Calculator C113
    pub slip_rpm: f64,                 // Metal design C85
    pub slip_event_s: f64,             // Metal design C87
    pub life_events: f64,              // Metal design C88
    pub measured_drag_Nm: Option<f64>, // Metal design C90
    pub cold_high_Nm: f64,             // Metal design C10
    pub required_min_Nm: f64,          // Metal design C7
    pub variation: f64,                // Metal design C18
    pub min_temp_C: f64,               // Metal design C16
    pub magnetic_cycles: f64,          // Metal design C89
    pub bond_inner_mm: f64,            // Metal design C120
    pub bond_outer_mm: f64,            // Metal design C121
    pub sleeve_mm: f64,                // Metal design C25
    pub liner_mm: f64,                 // Metal design C26
    pub sleeve_id_mm: f64,             // Metal design C175
    pub sleeve_od_mm: f64,             // Metal design C176
    pub liner_od_mm: f64,              // Metal design C177
    pub liner_id_mm: f64,              // Metal design C178
    pub cap_face_mm: f64,              // Metal design C167
    pub hardware_g: f64,               // Metal design C128
    pub retainers_g: f64,              // Metal design C46
    pub cap_g: f64,                    // Metal design C180
    pub endplates_g: f64,              // Metal design C181
    pub steel_sigma_S_m: f64,          // Materials C14
    pub steel_mu_r: f64,               // Materials C15
    pub steel_c: f64,                  // Materials C16
    pub steel_cte: f64,                // Materials C17
    pub steel_E_GPa: f64,              // Materials C18
    pub al6061_sigma_S_m: f64,         // Materials C43
}

// =========================================================================== results
results! {
    /// The summary block (Temperature design!C6:C25, F12, F16).
    pub struct SummaryResults {
        fields {
            service_max_C: f64 => out("°C", "Service maximum magnet temperature", "", "Temperature design!C6"),
            onset_aligned_C: f64 => out("°C", "Demag onset, normal running (rings aligned)", "", "Temperature design!C7"),
            onset_pullout_C: f64 => out("°C", "Demag onset at pull-out", "", "Temperature design!C8"),
            onset_skipping_C: f64 => out("°C", "Demag onset while skipping (like poles facing)", "", "Temperature design!C9"),
            magnet_limit_C: f64 => out("°C", "Magnet design limit", "", "Temperature design!C10"),
            adhesive_limit_C: f64 => out("°C", "Adhesive design limit (selected adhesive)", "", "Temperature design!C11"),
            governing_limit_C: f64 => out("°C", "Governing temperature limit", "", "Temperature design!C12"),
            governing_note: String => out("", "Which limit governs", "", "Temperature design!F12"),
            margin_service_C: f64 => out("°C", "Margin above the service maximum", "", "Temperature design!C13"),
            hot_day_start_C: f64 => out("°C", "Hot-day starting magnet temperature", "", "Temperature design!C14"),
            margin_hot_day_C: f64 => out("°C", "Margin above the hot-day start", "", "Temperature design!C15"),
            torque_hot_day_Nm: f64 => out("N·m", "Pull-out torque at the hot-day start (reversible)", "", "Temperature design!C16"),
            torque_hot_day_note: String => out("", "Against the requirement", "", "Temperature design!F16"),
            steady_estimate_C: f64 => out("°C", "Continuous slip, steady magnet temp, estimate", "", "Temperature design!C17"),
            steady_high_C: f64 => out("°C", "Continuous slip, steady magnet temp, high case", "", "Temperature design!C18"),
            time_to_limit_high: NumOrText => out("s", "Unbroken slip time to the limit, high case", "", "Temperature design!C19"),
            peak_with_fault_C: f64 => out("°C", "Peak magnet and bond temperature with the slip fault", "", "Temperature design!C20"),
            life_rotations: f64 => out("rev", "Relative slip rotations over life", "", "Temperature design!C21"),
            avg_slip_heating_high_C: f64 => out("°C", "Average slip heating over life, high case", "", "Temperature design!C22"),
            critical_drag_Nm: f64 => out("N·m", "Slip drag torque that would reach the limit", "", "Temperature design!C23"),
            cure_margin_C: f64 => out("°C", "Cure margin below the single-ring demag onset", "", "Temperature design!C24"),
            verdict: String => out("", "Verdict", "", "Temperature design!C25"),
        }
    }
}

results! {
    /// Duty block (Temperature design!C28:C37).
    pub struct DutyResults {
        fields {
            slip_rpm: f64 => out("rpm", "Relative slip speed at the wheel", "", "Temperature design!C28"),
            slip_rad_s: f64 => out("rad/s", "Slip angular speed", "", "Temperature design!C29"),
            pole_pairs: f64 => out("-", "Pole pairs per ring", "", "Temperature design!C30"),
            field_freq_Hz: f64 => out("Hz", "Field frequency seen by the opposite ring", "", "Temperature design!C31"),
            field_omega_rad_s: f64 => out("rad/s", "Field angular frequency", "", "Temperature design!C32"),
            slip_event_s: f64 => out("s", "Slip duration per event", "", "Temperature design!C33"),
            hot_day_start_C: f64 => out("°C", "Hot-day starting magnet temperature", "", "Temperature design!C37"),
        }
    }
}

results! {
    /// Demagnetization block (Temperature design!C42:C62).
    pub struct DemagResults {
        fields {
            br20_T: f64 => out("T", "Remanence at 20 °C", "", "Temperature design!C42"),
            alpha_br: f64 => out("1/°C", "Br temperature coefficient", "", "Temperature design!C43"),
            // annotated `float` in Python, but "n/a" for a manual magnet
            tmax_lib_C: NumOrText => out("°C", "Library maximum operating temperature", "", "Temperature design!C47"),
            h_ref_kA_m: f64 => out("kA/m", "Reverse field of a magnet at permeance coefficient 1", "", "Temperature design!C48"),
            t_ref_model_C: f64 => out("°C", "Model onset for that reference magnet", "", "Temperature design!C49"),
            calibration_offset_C: f64 => out("°C", "Calibration offset (model minus library rating)", "", "Temperature design!C50"),
            onset_aligned_C: f64 => out("°C", "Onset, rings aligned", "", "Temperature design!C56"),
            onset_pullout_C: f64 => out("°C", "Onset at pull-out", "", "Temperature design!C57"),
            onset_skipping_C: f64 => out("°C", "Onset while skipping", "", "Temperature design!C58"),
            onset_single_ring_C: f64 => out("°C", "Onset, single ring during an adhesive cure", "", "Temperature design!C59"),
            magnet_limit_C: f64 => out("°C", "Magnet design limit", "", "Temperature design!C60"),
            torque_at_limit_Nm: f64 => out("N·m", "Pull-out torque at the magnet limit (reversible)", "", "Temperature design!C61"),
            torque_at_service_Nm: f64 => out("N·m", "Pull-out torque at the service maximum", "", "Temperature design!C62"),
        }
    }
}

results! {
    /// Adhesive selection and loads (Temperature design!C76:C91).
    pub struct AdhesiveResults {
        fields {
            selected_name: String => out_uncelled("", "Selected adhesive", ""),
            design_limit_C: f64 => out("°C", "Selected adhesive design limit", "", "Temperature design!C76"),
            cure_C: f64 => out("°C", "Selected adhesive cure (stress-free) temperature", "", "Temperature design!C77"),
            lap_shear_MPa: f64 => out("MPa", "Selected adhesive lap shear at 22 °C (TDS)", "", "Temperature design!C78"),
            bond_area_mm2: f64 => out("mm²", "Bond area per block (back face)", "", "Temperature design!C81"),
            block_mass_g: f64 => out("g", "Block mass", "", "Temperature design!C82"),
            cold_high_torque_Nm: f64 => out("N·m", "Highest pull-out torque (cold, +variation)", "", "Temperature design!C83"),
            inner_mid_radius_mm: f64 => out("mm", "Inner block mid radius", "", "Temperature design!C84"),
            tangential_force_N: f64 => out("N", "Tangential force per inner block at pull-out", "", "Temperature design!C85"),
            bond_shear_MPa: f64 => out("MPa", "Bond shear stress from magnetic torque", "", "Temperature design!C86"),
            centrifugal_force_N: f64 => out("N", "Centrifugal force per inner block at wheel speed", "", "Temperature design!C88"),
            static_ratio: f64 => out("-", "Static strength ratio at 22 °C", "", "Temperature design!C89"),
            shear_reversals: f64 => out("cycles", "Shear reversals over life while slipping", "", "Temperature design!C90"),
            fatigue_screen: String => out("", "Fatigue screen", "", "Temperature design!C91"),
        }
    }
}

results! {
    /// Thermal mismatch screen (Temperature design!C94:C106).
    pub struct MismatchResults {
        fields {
            steel_cte: f64 => out("1/°C", "Steel expansion coefficient (4140)", "", "Temperature design!C94"),
            steel_E_GPa: f64 => out("GPa", "Steel elastic modulus (4140)", "", "Temperature design!C98"),
            steel_thickness_mm: f64 => out("mm", "Steel thickness under the inner blocks", "", "Temperature design!C99"),
            cold_limit_C: f64 => out("°C", "Cold limit", "", "Temperature design!C100"),
            worst_swing_C: f64 => out("°C", "Worst swing from the stress-free (cure) temperature", "", "Temperature design!C101"),
            current_bondline_mm: f64 => out("mm", "Current bondline", "", "Temperature design!C102"),
            peak_shear_current_MPa: f64 => out("MPa", "Peak end shear, current bondline", "", "Temperature design!C104"),
            peak_shear_recommended_MPa: f64 => out("MPa", "Peak end shear, recommended bondline", "", "Temperature design!C105"),
            reading: String => out("", "Reading", "", "Temperature design!C106"),
        }
    }
}

results! {
    /// Slip losses (Temperature design!C109:C134).
    pub struct SlipLossResults {
        fields {
            steel_sigma_S_m: f64 => out("S/m", "Steel conductivity (4140)", "", "Temperature design!C109"),
            steel_mu_r: f64 => out("-", "Steel incremental relative permeability (4140)", "", "Temperature design!C110"),
            cap_sigma_S_m: f64 => out("S/m", "Aluminium cap conductivity (6061-T6)", "", "Temperature design!C112"),
            skin_depth_mm: f64 => out("mm", "Steel skin depth at the field frequency", "", "Temperature design!C115"),
            hub_W: f64 => out("W", "Hub surface (solid steel)", "", "Temperature design!C123"),
            cup_W: f64 => out("W", "Cup surface (solid steel)", "", "Temperature design!C124"),
            web_W: f64 => out("W", "Rear web (solid steel)", "", "Temperature design!C125"),
            sleeve_W: f64 => out("W", "Inner 316L sleeve", "", "Temperature design!C126"),
            liner_W: f64 => out("W", "Outer 316L liner", "", "Temperature design!C127"),
            cap_W: f64 => out("W", "Aluminium cap face", "", "Temperature design!C128"),
            magnets_W: f64 => out("W", "Magnet eddy currents (both rings)", "", "Temperature design!C129"),
            total_W: f64 => out("W", "Total estimated slip loss", "", "Temperature design!C130"),
            drag_Nm: f64 => out("N·m", "Equivalent mean drag torque", "", "Temperature design!C131"),
            used_W: f64 => out("W", "Loss used below", "Bench drag replaces the estimate once entered.", "Temperature design!C132"),
            high_W: f64 => out("W", "Loss, high case", "", "Temperature design!C134"),
        }
    }
}

results! {
    /// Thermal network (Temperature design!C137:C161).
    pub struct ThermalResults {
        fields {
            steel_c: f64 => out("J/(kg·K)", "Specific heat, steel (4140)", "", "Temperature design!C137"),
            heat_capacity_J_K: f64 => out("J/K", "Heat capacity of the rotating coupling", "", "Temperature design!C141"),
            time_constant_s: f64 => out("s", "Thermal time constant", "", "Temperature design!C143"),
            start_C: f64 => out("°C", "Starting magnet temperature", "", "Temperature design!C144"),
            rise_per_event_C: f64 => out("°C", "Temperature rise per slip event", "", "Temperature design!C145"),
            steady_rise_est_C: f64 => out("°C", "Continuous slip: steady rise, estimate", "", "Temperature design!C146"),
            steady_rise_high_C: f64 => out("°C", "Continuous slip: steady rise, high case", "", "Temperature design!C147"),
            steady_est_C: f64 => out("°C", "Continuous slip, steady magnet temp, estimate", "", "Temperature design!C148"),
            steady_high_C: f64 => out("°C", "Continuous slip, steady magnet temp, high case", "", "Temperature design!C149"),
            time_to_limit_high: NumOrText => out("s", "Continuous slip, time to the limit (high case)", "", "Temperature design!C150"),
            rotations_to_limit_high: NumOrText => out("rev", "Continuous slip, rotations to the limit (high case)", "", "Temperature design!C151"),
            time_to_limit_est: NumOrText => out("s", "Continuous slip, time to the limit (estimate)", "", "Temperature design!C152"),
            critical_drag_Nm: f64 => out("N·m", "Slip drag torque that would reach the limit", "", "Temperature design!C153"),
            heating_rate_est_C_s: f64 => out("°C/s", "Initial heating rate, estimate", "", "Temperature design!C154"),
            heating_rate_high_C_s: f64 => out("°C/s", "Initial heating rate, high case", "", "Temperature design!C155"),
            rev_per_C_est: f64 => out("rev/°C", "Relative rotations per °C at the start, estimate", "", "Temperature design!C156"),
            rev_per_C_high: f64 => out("rev/°C", "Relative rotations per °C at the start, high case", "", "Temperature design!C157"),
            rev_per_tau: f64 => out("rev", "Relative rotations per thermal time constant", "", "Temperature design!C158"),
            t95_s: f64 => out("s", "Time to 95 % of the steady rise", "", "Temperature design!C159"),
            rev95: f64 => out("rev", "Relative rotations to 95 % of the steady rise", "", "Temperature design!C160"),
            temp_at_fault_C: f64 => out("°C", "Magnet temperature at the fault trip time, high case", "", "Temperature design!C161"),
        }
    }
}

results! {
    /// Slip life (Temperature design!C164:C177).
    pub struct SlipLifeResults {
        fields {
            events: f64 => out("events", "Slip events over life", "", "Temperature design!C164"),
            rev_per_event: f64 => out("rev", "Relative rotations per event", "", "Temperature design!C165"),
            rotations: f64 => out("rev", "Relative rotations over life", "", "Temperature design!C166"),
            slip_hours: f64 => out("h", "Total slip time over life", "", "Temperature design!C167"),
            like_pole_passes: f64 => out("passes", "Like-pole passes per magnet over life", "", "Temperature design!C168"),
            heat_per_event_est_J: f64 => out("J", "Heat per event, estimate", "", "Temperature design!C169"),
            heat_per_event_high_J: f64 => out("J", "Heat per event, high case", "", "Temperature design!C170"),
            rise_per_event_est_C: f64 => out("°C", "Temperature rise per event, estimate", "", "Temperature design!C171"),
            rise_per_event_high_C: f64 => out("°C", "Temperature rise per event, high case", "", "Temperature design!C172"),
            life_heat_high_MJ: f64 => out("MJ", "Total slip heat over life, high case", "", "Temperature design!C173"),
            slip_duty: f64 => out("-", "Slip duty: share of operating time spent slipping", "", "Temperature design!C174"),
            avg_rise_est_C: f64 => out("°C", "Average temperature rise from slip, estimate", "", "Temperature design!C175"),
            avg_rise_high_C: f64 => out("°C", "Average temperature rise from slip, high case", "", "Temperature design!C176"),
            rise_per_pct_duty_C: f64 => out("°C", "Average rise per 1 % of time slipping, high case", "", "Temperature design!C177"),
        }
    }
}

results! {
    /// Magnet life (Temperature design!C180:C186).
    pub struct MagnetLifeResults {
        fields {
            peak_C: f64 => out("°C", "Peak magnet temperature: hot day plus fault-limited slip", "", "Temperature design!C180"),
            margin_onset_C: f64 => out("°C", "Margin to the skipping onset", "", "Temperature design!C181"),
            margin_limit_C: f64 => out("°C", "Margin to the magnet design limit", "", "Temperature design!C182"),
            torque_hot_day_Nm: f64 => out("N·m", "Pull-out torque at the hot-day start (reversible)", "", "Temperature design!C184"),
            torque_hot_day_check: String => out("", "Against the requirement", "", "Temperature design!C185"),
            torque_peak_Nm: f64 => out("N·m", "Pull-out torque at the peak temperature (reversible)", "", "Temperature design!C186"),
        }
    }
}

results! {
    /// Adhesive life (Temperature design!C189:C202).
    pub struct AdhesiveLifeResults {
        fields {
            peak_C: f64 => out("°C", "Peak bond temperature", "", "Temperature design!C189"),
            margin_C: f64 => out("°C", "Margin to the adhesive design limit", "", "Temperature design!C190"),
            reversals: f64 => out("cycles", "Shear reversals over life", "", "Temperature design!C191"),
            torque_peak_var_Nm: f64 => out("N·m", "Pull-out torque at the peak temperature, with +variation", "", "Temperature design!C192"),
            shear_amplitude_MPa: f64 => out("MPa", "Shear stress amplitude per reversal, hot", "", "Temperature design!C193"),
            hot_fatigue_margin: f64 => out("x", "Hot fatigue margin", "", "Temperature design!C196"),
            hot_fatigue_screen: String => out("", "Hot fatigue screen", "", "Temperature design!C197"),
            daily_cycles: f64 => out("cycles", "Daily thermal cycles over life", "", "Temperature design!C200"),
            daily_peak_shear_MPa: f64 => out("MPa", "Peak end shear per daily cycle, recommended bondline", "", "Temperature design!C201"),
            daily_screen: String => out("", "Daily-cycle screen", "", "Temperature design!C202"),
        }
    }
}

results! {
    /// Every Temperature design result, grouped as the Python `TemperatureResults`.
    pub struct TemperatureResults {
        fields {}
        groups {
            summary: SummaryResults,
            duty: DutyResults,
            demag: DemagResults,
            adhesive: AdhesiveResults,
            mismatch: MismatchResults,
            slip_loss: SlipLossResults,
            thermal: ThermalResults,
            slip_life: SlipLifeResults,
            magnet_life: MagnetLifeResults,
            adhesive_life: AdhesiveLifeResults,
        }
    }
}

// =========================================================================== model
/// The text of an unbroken-slip time that never reaches the limit.
pub const NEVER: &str = "never: steady state stays below the limit";
/// The text of the matching rotation count.
pub const NEVER_SHORT: &str = "never";

/// Temperature where the reverse field (scaling with Br) reaches the knee of the Hcj curve.
#[allow(non_snake_case)]
pub fn demag_onset_C(
    h_rev_kA_m: f64,
    hcj20: f64,
    beta: f64,
    knee: f64,
    alpha_br: f64,
    offset_C: f64,
) -> f64 {
    let hk = knee * hcj20;
    20.0 + (hk - h_rev_kA_m) / (hk * beta.abs() - h_rev_kA_m * alpha_br.abs()) - offset_C
}

/// Linear-elastic shear-lag peak shear at the ends of a bonded block from thermal mismatch.
#[allow(non_snake_case, clippy::too_many_arguments)] // Python names and signature
pub fn volkersen_peak_shear_MPa(
    G_GPa: f64,
    d_alpha: f64,
    dT: f64,
    bondline_mm: f64,
    magnet_E_GPa: f64,
    magnet_t_mm: f64,
    steel_E_GPa: f64,
    steel_t_mm: f64,
    bond_length_mm: f64,
) -> f64 {
    let G = G_GPa * 1e9;
    let lam = (G / (bondline_mm / 1000.0)
        * (1.0 / (magnet_E_GPa * 1e9 * magnet_t_mm / 1000.0)
            + 1.0 / (steel_E_GPa * 1e9 * steel_t_mm / 1000.0)))
        .sqrt();
    G * d_alpha * dT * (lam * bond_length_mm / 2000.0).tanh() / (lam * bondline_mm / 1000.0) / 1e6
}

#[allow(non_snake_case)] // Python names (T0, Ft, Fc, dT, L, C, G, Te, Th, Ee, Eh, ...)
pub fn compute(
    ti: &TemperatureInputs,
    k: &TemperatureLinks,
    _dev: Deviations,
) -> TemperatureResults {
    let npole = k.npole as f64; // integer overflow rule: arithmetic in f64
    // ---- duty
    let omega = k.slip_rpm * 2.0 * PI / 60.0;
    let pp = npole / 2.0;
    let f = pp * k.slip_rpm / 60.0;
    let we = 2.0 * PI * f;
    let T0 = ti.duty.hot_ambient_C + ti.duty.driving_rise_C;
    let duty = DutyResults {
        slip_rpm: k.slip_rpm,
        slip_rad_s: omega,
        pole_pairs: pp,
        field_freq_Hz: f,
        field_omega_rad_s: we,
        slip_event_s: k.slip_event_s,
        hot_day_start_C: T0,
    };

    // ---- demagnetization
    let d = &ti.demag;
    let h_ref = k.br20_T / (2.0 * k.mu0) / 1000.0;
    let t_ref = demag_onset_C(
        h_ref,
        d.hcj20_kA_m,
        d.beta_hcj_per_C,
        d.knee_fraction,
        k.alpha_br,
        0.0,
    );
    // the workbook errors out when the magnet is not in the library; Python leaves the onset uncalibrated
    let offset = match k.tmax_lib_C {
        NumOrText::Num(tmax) => t_ref - tmax,
        NumOrText::Text(_) => 0.0,
    };
    let on = |h: f64| {
        demag_onset_C(
            h,
            d.hcj20_kA_m,
            d.beta_hcj_per_C,
            d.knee_fraction,
            k.alpha_br,
            offset,
        )
    };
    let (on_al, on_po, on_lp, on_cu) = (
        on(d.h_rev_aligned_kA_m),
        on(d.h_rev_pullout_kA_m),
        on(d.h_rev_likepole_kA_m),
        on(d.h_rev_single_ring_kA_m),
    );
    let mag_lim = on_lp - d.design_margin_C;
    let thf = |T: f64| (1.0 + k.alpha_br * (T - 20.0)).powi(2);
    let demag = DemagResults {
        br20_T: k.br20_T,
        alpha_br: k.alpha_br,
        tmax_lib_C: k.tmax_lib_C,
        h_ref_kA_m: h_ref,
        t_ref_model_C: t_ref,
        calibration_offset_C: offset,
        onset_aligned_C: on_al,
        onset_pullout_C: on_po,
        onset_skipping_C: on_lp,
        onset_single_ring_C: on_cu,
        magnet_limit_C: mag_lim,
        torque_at_limit_Nm: k.pullout_20C_Nm * thf(mag_lim),
        torque_at_service_Nm: k.pullout_op_Nm,
    };

    // ---- adhesive selection and loads
    let sel = selected_adhesive(ti.adhesive.selected);
    let area = k.inner_length_mm * k.inner_width_mm;
    let m_block = k.inner_length_mm * k.inner_width_mm * k.inner_thickness_mm * 0.0075; // literal, as Python
    let r_mid = k.inner_back_apothem_mm + k.inner_thickness_mm / 2.0;
    let Ft = k.cold_high_Nm / (npole * r_mid / 1000.0);
    let tau_b = Ft / area;
    let Fc =
        m_block / 1000.0 * (ti.duty.wheel_rotor_rpm * 2.0 * PI / 60.0).powi(2) * r_mid / 1000.0;
    let fat = 0.2 * sel.lap_shear_MPa / tau_b;
    let adh = AdhesiveResults {
        selected_name: sel.name.to_owned(),
        design_limit_C: sel.design_limit_C,
        cure_C: sel.cure_C,
        lap_shear_MPa: sel.lap_shear_MPa,
        bond_area_mm2: area,
        block_mass_g: m_block,
        cold_high_torque_Nm: k.cold_high_Nm,
        inner_mid_radius_mm: r_mid,
        tangential_force_N: Ft,
        bond_shear_MPa: tau_b,
        centrifugal_force_N: Fc,
        static_ratio: sel.lap_shear_MPa / tau_b,
        shear_reversals: k.magnetic_cycles,
        fatigue_screen: if fat >= 4.0 {
            format!("OK: {}x margin", text0(fat))
        } else {
            "CHECK".to_owned()
        },
    };

    let gov = py_min(mag_lim, sel.design_limit_C);

    // ---- thermal mismatch screen
    let mm = &ti.mismatch;
    let dT = py_max(sel.cure_C - k.min_temp_C, gov - sel.cure_C);
    let d_alpha = k.steel_cte - mm.ndfeb_cte_per_C;
    let s1 = volkersen_peak_shear_MPa(
        mm.adhesive_shear_modulus_GPa,
        d_alpha,
        dT,
        k.bond_inner_mm,
        mm.ndfeb_modulus_GPa,
        k.inner_thickness_mm,
        k.steel_E_GPa,
        k.hub_wall_mm,
        k.inner_length_mm,
    );
    let s2 = volkersen_peak_shear_MPa(
        mm.adhesive_shear_modulus_GPa,
        d_alpha,
        dT,
        mm.recommended_bondline_mm,
        mm.ndfeb_modulus_GPa,
        k.inner_thickness_mm,
        k.steel_E_GPa,
        k.hub_wall_mm,
        k.inner_length_mm,
    );
    let mis = MismatchResults {
        steel_cte: k.steel_cte,
        steel_E_GPa: k.steel_E_GPa,
        steel_thickness_mm: k.hub_wall_mm,
        cold_limit_C: k.min_temp_C,
        worst_swing_C: dT,
        current_bondline_mm: k.bond_inner_mm,
        peak_shear_current_MPa: s1,
        peak_shear_recommended_MPa: s2,
        reading: if s1 > sel.lap_shear_MPa {
            "Above the lap-shear strength at the block ends"
        } else {
            "Below the lap-shear strength"
        }
        .to_owned(),
    };

    // ---- slip losses (estimates)
    let sl = &ti.slip_loss;
    let delta = (2.0 / (we * k.mu0 * k.steel_mu_r * k.steel_sigma_S_m)).sqrt() * 1000.0; // mm
    let L = k.active_length_mm / 1000.0;
    let r_hub = (k.inner_back_apothem_mm - k.bond_inner_mm) / 1000.0;
    let r_cup = (k.outer_back_apothem_mm + k.bond_outer_mm) / 1000.0;
    let surface = |B: f64, r: f64| {
        k.steel_sigma_S_m * we.powi(2) * B.powi(2) * (delta / 1000.0) / (4.0 * (pp / r).powi(2))
            * 2.0
            * PI
            * r
            * L
    };
    let p_hub = surface(sl.b_hub_T, r_hub);
    let p_cup = surface(sl.b_cup_T, r_cup);
    let p_web = k.steel_sigma_S_m * we.powi(2) * (delta / 1000.0) / 4.0
        * ((r_mid / 1000.0) / pp).powi(2)
        * sl.web_integral_T2m2;
    let r_s = (k.sleeve_id_mm + k.sleeve_od_mm) / 4.0 / 1000.0;
    let r_l = (k.liner_od_mm + k.liner_id_mm) / 4.0 / 1000.0;
    let shell = |t_mm: f64, r: f64, B: f64| {
        sl.end_factor * sl.sigma_316_S_m * (t_mm / 1000.0) * (omega * r).powi(2) * B.powi(2) / 2.0
            * 2.0
            * PI
            * r
            * L
    };
    let p_slv = shell(k.sleeve_mm, r_s, sl.b_sleeve_T);
    let p_lin = shell(k.liner_mm, r_l, sl.b_liner_T);
    let p_cap = sl.end_factor
        * k.al6061_sigma_S_m
        * (k.cap_face_mm / 1000.0)
        * omega.powi(2)
        * sl.cap_integral_T2m4;
    let p_mag = sl.sigma_ndfeb_S_m
        * we.powi(2)
        * sl.b_magnet_T.powi(2)
        * (k.inner_width_mm / 1000.0).powi(2)
        / 24.0
        * (k.inner_length_mm * k.inner_width_mm * k.inner_thickness_mm * 1e-9)
        * 2.0
        * npole;
    let p_tot = p_hub + p_cup + p_web + p_slv + p_lin + p_cap + p_mag;
    // Python: measured = isinstance(drag, (int, float)); p_use = drag * omega if measured else p_tot;
    // p_hi = p_use if measured else p_use * high_multiplier
    let (p_use, p_hi) = match k.measured_drag_Nm {
        Some(drag) => (drag * omega, drag * omega),
        None => (p_tot, p_tot * sl.high_multiplier),
    };
    let loss = SlipLossResults {
        steel_sigma_S_m: k.steel_sigma_S_m,
        steel_mu_r: k.steel_mu_r,
        cap_sigma_S_m: k.al6061_sigma_S_m,
        skin_depth_mm: delta,
        hub_W: p_hub,
        cup_W: p_cup,
        web_W: p_web,
        sleeve_W: p_slv,
        liner_W: p_lin,
        cap_W: p_cap,
        magnets_W: p_mag,
        total_W: p_tot,
        drag_Nm: p_tot / omega,
        used_W: p_use,
        high_W: p_hi,
    };

    // ---- thermal network
    let th = &ti.thermal;
    let C = (k.mass_magnets_g * th.c_ndfeb
        + (k.mass_cup_g + k.mass_hub_g + k.mass_boss_g + k.hardware_g) * k.steel_c
        + (k.retainers_g + k.endplates_g) * th.c_316
        + k.cap_g * th.c_aluminium)
        / 1000.0;
    let G = th.conductance_W_K;
    let tau_th = C / G;
    let (rise_e, rise_h) = (p_use / G, p_hi / G);
    let (Te, Th) = (T0 + rise_e, T0 + rise_h);
    let t_lim_h = if Th <= gov {
        NumOrText::Text(NEVER)
    } else {
        NumOrText::Num(-tau_th * (1.0 - (gov - T0) / rise_h).ln())
    };
    let t_lim_e = if Te <= gov {
        NumOrText::Text(NEVER)
    } else {
        NumOrText::Num(-tau_th * (1.0 - (gov - T0) / rise_e).ln())
    };
    let rev_s = k.slip_rpm / 60.0;
    let (ke, kh) = (p_use / C, p_hi / C);
    let T_fault = T0 + rise_h * (1.0 - (-ti.duty.fault_trip_s / tau_th).exp());
    let thermal = ThermalResults {
        steel_c: k.steel_c,
        heat_capacity_J_K: C,
        time_constant_s: tau_th,
        start_C: T0,
        rise_per_event_C: p_use * k.slip_event_s / C,
        steady_rise_est_C: rise_e,
        steady_rise_high_C: rise_h,
        steady_est_C: Te,
        steady_high_C: Th,
        time_to_limit_high: t_lim_h,
        rotations_to_limit_high: match t_lim_h {
            NumOrText::Num(t) => NumOrText::Num(t * rev_s),
            NumOrText::Text(_) => NumOrText::Text(NEVER_SHORT),
        },
        time_to_limit_est: t_lim_e,
        critical_drag_Nm: (gov - T0) * G / omega,
        heating_rate_est_C_s: ke,
        heating_rate_high_C_s: kh,
        rev_per_C_est: rev_s / ke,
        rev_per_C_high: rev_s / kh,
        rev_per_tau: tau_th * rev_s,
        t95_s: 3.0 * tau_th,
        rev95: 3.0 * tau_th * rev_s,
        temp_at_fault_C: T_fault,
    };

    // ---- slip life
    let rpe = rev_s * k.slip_event_s;
    let rot = k.life_events * rpe;
    let hrs = k.life_events * k.slip_event_s / 3600.0;
    let (Ee, Eh) = (p_use * k.slip_event_s, p_hi * k.slip_event_s);
    let (dTe, dTh) = (Ee / C, Eh / C);
    let duty_frac = hrs / ti.duty.life_hours;
    let life = SlipLifeResults {
        events: k.life_events,
        rev_per_event: rpe,
        rotations: rot,
        slip_hours: hrs,
        like_pole_passes: rot * pp,
        heat_per_event_est_J: Ee,
        heat_per_event_high_J: Eh,
        rise_per_event_est_C: dTe,
        rise_per_event_high_C: dTh,
        life_heat_high_MJ: k.life_events * Eh / 1e6,
        slip_duty: duty_frac,
        avg_rise_est_C: duty_frac * rise_e,
        avg_rise_high_C: duty_frac * rise_h,
        rise_per_pct_duty_C: 0.01 * rise_h,
    };

    // ---- magnet life
    let peak = py_max(T_fault, T0 + dTh) + duty_frac * rise_h;
    let tq_hot = k.pullout_20C_Nm * thf(T0);
    let tq_peak = k.pullout_20C_Nm * thf(peak);
    let mlife = MagnetLifeResults {
        peak_C: peak,
        margin_onset_C: on_lp - peak,
        margin_limit_C: mag_lim - peak,
        torque_hot_day_Nm: tq_hot,
        torque_hot_day_check: if tq_hot >= k.required_min_Nm {
            "Meets it nominally (no variation allowance)"
        } else {
            "Below it"
        }
        .to_owned(),
        torque_peak_Nm: tq_peak,
    };

    // ---- adhesive life
    let al = &ti.adhesive_life;
    let tq_var = tq_peak * (1.0 + k.variation);
    let amp = tq_var / (npole * r_mid / 1000.0) / area;
    let hot_fm = sel.lap_shear_MPa * al.hot_strength_retained * al.fatigue_endurance / amp;
    let daily = s2 * al.daily_swing_C / dT;
    let alife = AdhesiveLifeResults {
        peak_C: peak,
        margin_C: sel.design_limit_C - peak,
        reversals: rot * pp,
        torque_peak_var_Nm: tq_var,
        shear_amplitude_MPa: amp,
        hot_fatigue_margin: hot_fm,
        hot_fatigue_screen: if hot_fm >= 4.0 {
            "OK"
        } else {
            "CHECK: get hot fatigue data"
        }
        .to_owned(),
        daily_cycles: al.service_years * 365.0,
        daily_peak_shear_MPa: daily,
        daily_screen: if daily < sel.lap_shear_MPa * al.fatigue_endurance {
            "Below the fatigue endurance"
        } else {
            "Above the fatigue endurance: qualify by thermal cycling"
        }
        .to_owned(),
    };

    // ---- summary
    let cure_margin = on_cu - sel.cure_C;
    let margin_hot = gov - T0;
    let ok = margin_hot > 0.0
        && (mag_lim - peak) > 0.0
        && (sel.design_limit_C - peak) > 0.0
        && cure_margin >= 10.0;
    let summary = SummaryResults {
        service_max_C: k.op_temp_C,
        onset_aligned_C: on_al,
        onset_pullout_C: on_po,
        onset_skipping_C: on_lp,
        magnet_limit_C: mag_lim,
        adhesive_limit_C: sel.design_limit_C,
        governing_limit_C: gov,
        governing_note: if mag_lim <= sel.design_limit_C {
            "Magnets govern (skipping case)."
        } else {
            "Adhesive governs."
        }
        .to_owned(),
        margin_service_C: gov - k.op_temp_C,
        hot_day_start_C: T0,
        margin_hot_day_C: margin_hot,
        torque_hot_day_Nm: tq_hot,
        torque_hot_day_note: mlife.torque_hot_day_check.clone(),
        steady_estimate_C: Te,
        steady_high_C: Th,
        time_to_limit_high: t_lim_h,
        peak_with_fault_C: peak,
        life_rotations: rot,
        avg_slip_heating_high_C: duty_frac * rise_h,
        critical_drag_Nm: (gov - T0) * G / omega,
        cure_margin_C: cure_margin,
        verdict: if ok {
            "OK on temperature. Confirm drag torque and thermal cycling by test."
        } else {
            "CHECK: see the rows above."
        }
        .to_owned(),
    };
    TemperatureResults {
        summary,
        duty,
        demag,
        adhesive: adh,
        mismatch: mis,
        slip_loss: loss,
        thermal,
        slip_life: life,
        magnet_life: mlife,
        adhesive_life: alife,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_adhesive_covers_the_four_codes_only() {
        for (code, name) in [(1, "Loctite AA 326 + SF 7649"), (4, "3M Scotch-Weld DP460")] {
            assert_eq!(selected_adhesive(code).name, name);
        }
        for bad in [0, 5, -1, i64::MIN, i64::MAX] {
            let none = selected_adhesive(bad);
            assert_eq!(none.name, "#N/A", "{bad}");
            assert!(
                none.lap_shear_MPa.is_nan() && none.design_limit_C.is_nan(),
                "{bad}"
            );
        }
    }

    /// The default design's links (`api::compute` at the workbook defaults, long decimals
    /// rounded). Each equality test below changes only the fields it names.
    fn links() -> TemperatureLinks {
        TemperatureLinks {
            op_temp_C: 50.0,
            npole: 10,
            br20_T: 1.29,
            alpha_br: -0.0012,
            tmax_lib_C: NumOrText::Num(150.0),
            mu0: 1.256637e-6,
            pullout_op_Nm: 2.6473,
            pullout_20C_Nm: 2.8487,
            inner_back_apothem_mm: 10.15,
            inner_length_mm: 12.7,
            inner_width_mm: 6.35,
            inner_thickness_mm: 3.17,
            hub_wall_mm: 5.1,
            active_length_mm: 12.7,
            outer_back_apothem_mm: 17.89,
            mass_magnets_g: 38.347,
            mass_cup_g: 60.754,
            mass_hub_g: 25.81,
            mass_boss_g: 30.778,
            slip_rpm: 2000.0,
            slip_event_s: 0.1,
            life_events: 2e7,
            measured_drag_Nm: None,
            cold_high_Nm: 3.7647,
            required_min_Nm: 2.5,
            variation: 0.15,
            min_temp_C: -40.0,
            magnetic_cycles: 3.3333e8,
            bond_inner_mm: 0.05,
            bond_outer_mm: 0.05,
            sleeve_mm: 0.1,
            liner_mm: 0.2,
            sleeve_id_mm: 27.436,
            sleeve_od_mm: 27.636,
            liner_od_mm: 29.39,
            liner_id_mm: 28.99,
            cap_face_mm: 0.8,
            hardware_g: 6.0,
            retainers_g: 3.131,
            cap_g: 2.322,
            endplates_g: 6.653,
            steel_sigma_S_m: 4.5e6,
            steel_mu_r: 200.0,
            steel_c: 473.0,
            steel_cte: 12.3e-6,
            steel_E_GPa: 205.0,
            al6061_sigma_S_m: 2.5e7,
        }
    }

    fn run(ti: &TemperatureInputs, k: &TemperatureLinks) -> TemperatureResults {
        compute(ti, k, Deviations::NONE)
    }

    #[test]
    fn fatigue_screens_pass_at_a_margin_of_exactly_four() {
        // `fat >= 4` and `hot_fm >= 4`, with values that make every step exact:
        // npole * r_mid / 1000 = 10 * 100 / 1000 = 1 and a 1 mm² bond area.
        let mut k = links();
        k.inner_back_apothem_mm = 99.0;
        k.inner_thickness_mm = 2.0; // r_mid = 99 + 2 / 2 = 100 mm
        k.inner_length_mm = 1.0;
        k.inner_width_mm = 1.0;
        k.cold_high_Nm = 0.75; // tau_b = 0.75 MPa: fat = 0.2 * 15 / 0.75 = 4 (AA 326, 15 MPa)
        k.alpha_br = 0.0; // every temperature factor is exactly 1: amp = pullout_20C * (1 + variation)
        k.variation = 0.0;
        k.pullout_20C_Nm = 0.9375;
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 1;
        ti.adhesive_life.hot_strength_retained = 0.5;
        ti.adhesive_life.fatigue_endurance = 0.5; // hot_fm = 15 * 0.5 * 0.5 / 0.9375 = 4
        let r = run(&ti, &k);
        assert_eq!(
            (
                r.adhesive.bond_shear_MPa,
                r.adhesive_life.hot_fatigue_margin
            ),
            (0.75, 4.0)
        );
        assert_eq!(r.adhesive.fatigue_screen, "OK: 4x margin");
        assert_eq!(r.adhesive_life.hot_fatigue_screen, "OK");
        k.cold_high_Nm = 0.7500001; // both margins just under 4
        k.pullout_20C_Nm = 0.9375001;
        let r = run(&ti, &k);
        assert_eq!(r.adhesive.fatigue_screen, "CHECK");
        assert_eq!(
            r.adhesive_life.hot_fatigue_screen,
            "CHECK: get hot fatigue data"
        );
    }

    #[test]
    fn mismatch_reading_is_below_at_equal_shear() {
        // `s1 > lap` is strict. A 1 m bond length saturates tanh to exactly 1.0 (no
        // libm-dependent digits); the NdFeB CTE was found by search so that the peak
        // shear is exactly AA 326's 15 MPa lap shear (the first assert proves it).
        let mut k = links();
        k.inner_length_mm = 1000.0;
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 1;
        ti.mismatch.adhesive_shear_modulus_GPa = 0.55; // explicit: correction E1 changes the default
        ti.mismatch.ndfeb_cte_per_C = 8.83111640173109e-6;
        let r = run(&ti, &k);
        assert_eq!(r.mismatch.peak_shear_current_MPa, 15.0);
        assert_eq!(r.mismatch.reading, "Below the lap-shear strength");
    }

    #[test]
    fn limits_are_never_reached_when_the_steady_state_equals_the_limit() {
        // `Th <= gov` and `Te <= gov`. A bench drag of 0 puts both steady states at the
        // hot-day start (50 + 10 = 60 °C); DP460's 60 °C limit governs.
        let mut k = links();
        k.measured_drag_Nm = Some(0.0);
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 4;
        ti.duty.hot_ambient_C = 50.0;
        ti.duty.driving_rise_C = 10.0;
        let r = run(&ti, &k);
        assert_eq!(
            (
                r.thermal.steady_est_C,
                r.thermal.steady_high_C,
                r.summary.governing_limit_C
            ),
            (60.0, 60.0, 60.0)
        );
        assert_eq!(r.thermal.time_to_limit_high, NumOrText::Text(NEVER));
        assert_eq!(r.thermal.time_to_limit_est, NumOrText::Text(NEVER));
        assert_eq!(
            r.thermal.rotations_to_limit_high,
            NumOrText::Text(NEVER_SHORT)
        );
    }

    #[test]
    fn hot_day_torque_meets_the_minimum_at_equality() {
        // `tq_hot >= required_min`; the hot-day torque does not read the required minimum.
        let ti = TemperatureInputs::default();
        let mut k = links();
        k.required_min_Nm = run(&ti, &k).magnet_life.torque_hot_day_Nm;
        let r = run(&ti, &k);
        assert_eq!(r.magnet_life.torque_hot_day_Nm, k.required_min_Nm);
        assert_eq!(
            r.magnet_life.torque_hot_day_check,
            "Meets it nominally (no variation allowance)"
        );
        assert_eq!(
            r.summary.torque_hot_day_note,
            "Meets it nominally (no variation allowance)"
        );
    }

    #[test]
    fn magnets_govern_when_the_limits_are_equal() {
        // `mag_lim <= design_limit`. mag_lim = onset - margin; with margin = onset - 120 both
        // subtractions are exact (the onset, about 102.6 °C, lies within a factor 2 of 120),
        // so the magnet limit is exactly AA 326's 120 °C (the first assert proves it).
        let k = links();
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 1;
        ti.demag.design_margin_C = run(&ti, &k).demag.onset_skipping_C - 120.0;
        let r = run(&ti, &k);
        assert_eq!(r.summary.magnet_limit_C, 120.0);
        assert_eq!(r.summary.governing_note, "Magnets govern (skipping case).");
    }

    #[test]
    fn verdict_accepts_a_cure_margin_of_exactly_ten() {
        // `cure_margin >= 10`, the other three margins positive. EA 9514 cures at 120 °C; the
        // single-ring field was found by search so that its onset is exactly 130 °C (the first
        // assert proves it).
        let k = links();
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 2;
        ti.demag.h_rev_single_ring_kA_m = 666.3586306326292;
        let r = run(&ti, &k);
        assert_eq!(r.summary.cure_margin_C, 10.0);
        assert_eq!(
            r.summary.verdict,
            "OK on temperature. Confirm drag torque and thermal cycling by test."
        );
        ti.demag.h_rev_single_ring_kA_m = 667.0; // a lower onset: the margin drops under 10
        assert_eq!(run(&ti, &k).summary.verdict, "CHECK: see the rows above.");
    }

    // Equality edges of the daily screen and of each strict term of the verdict (controller
    // ruling C1). Each test asserts the equality first, checks that the other terms hold (so the
    // term under test alone decides the verdict), and steps just past the edge to see the verdict
    // flip. No constant is searched: each input is computed from a run that does not read it.
    const VERDICT_OK: &str = "OK on temperature. Confirm drag torque and thermal cycling by test.";
    const VERDICT_CHECK: &str = "CHECK: see the rows above.";

    #[test]
    fn daily_screen_is_above_at_equal_shear() {
        // `daily < lap * endurance` is strict. A 1 m block length saturates tanh to exactly 1.0
        // (no libm-dependent digits), so the run's peak shear and worst swing are exact inputs to
        // `daily = s2 * swing / dT`, which reads the daily swing and nothing else of the two: the
        // swing below gives exactly the limit (the first assert proves it).
        let mut k = links();
        k.inner_length_mm = 1000.0;
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 1;
        ti.mismatch.adhesive_shear_modulus_GPa = 0.55; // explicit: correction E1 changes the default
        ti.adhesive_life.fatigue_endurance = 0.2;
        let limit = ADHESIVES[0].lap_shear_MPa * 0.2;
        let base = run(&ti, &k);
        ti.adhesive_life.daily_swing_C =
            limit * base.mismatch.worst_swing_C / base.mismatch.peak_shear_recommended_MPa;
        let r = run(&ti, &k);
        assert_eq!(r.adhesive_life.daily_peak_shear_MPa, limit);
        assert_eq!(
            r.adhesive_life.daily_screen,
            "Above the fatigue endurance: qualify by thermal cycling"
        );
        ti.adhesive_life.daily_swing_C *= 0.99; // just below the limit
        assert_eq!(
            run(&ti, &k).adhesive_life.daily_screen,
            "Below the fatigue endurance"
        );
    }

    #[test]
    fn verdict_rejects_a_hot_day_margin_of_exactly_zero() {
        // `margin_hot > 0` is strict; margin_hot = gov - T0. A negative bench drag (Python takes
        // it) cools the coupling, so the peak stays below the hot-day start and the magnet and
        // adhesive margins stay positive. With margin = onset - T0 the magnet limit, which
        // governs, is exactly T0 (both subtractions are exact: the onset lies within a factor 2 of T0).
        let mut k = links();
        k.measured_drag_Nm = Some(-1.0);
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 1;
        let base = run(&ti, &k);
        ti.demag.design_margin_C = base.demag.onset_skipping_C - base.summary.hot_day_start_C;
        let r = run(&ti, &k);
        assert_eq!(r.summary.margin_hot_day_C, 0.0);
        assert!(
            r.magnet_life.margin_limit_C > 0.0
                && r.adhesive_life.margin_C > 0.0
                && r.summary.cure_margin_C >= 10.0
        );
        assert_eq!(r.summary.verdict, VERDICT_CHECK);
        ti.demag.design_margin_C -= 0.001; // the margin becomes 0.001 °C
        assert_eq!(run(&ti, &k).summary.verdict, VERDICT_OK);
    }

    #[test]
    fn verdict_rejects_a_magnet_margin_of_exactly_zero() {
        // `(mag_lim - peak) > 0` is strict. The peak does not read the design margin, so with
        // margin = onset - peak the magnet limit is exactly the peak (the onset lies within a
        // factor 2 of the peak, so both subtractions are exact). The magnets govern at that
        // limit, above the hot-day start, and the adhesive limit is higher.
        let k = links();
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 1;
        let base = run(&ti, &k);
        ti.demag.design_margin_C = base.demag.onset_skipping_C - base.magnet_life.peak_C;
        let r = run(&ti, &k);
        assert_eq!(r.magnet_life.peak_C, base.magnet_life.peak_C);
        assert_eq!(r.magnet_life.margin_limit_C, 0.0);
        assert!(
            r.summary.margin_hot_day_C > 0.0
                && r.adhesive_life.margin_C > 0.0
                && r.summary.cure_margin_C >= 10.0
        );
        assert_eq!(r.summary.verdict, VERDICT_CHECK);
        ti.demag.design_margin_C -= 0.001; // the margin becomes 0.001 °C
        assert_eq!(run(&ti, &k).summary.verdict, VERDICT_OK);
    }

    #[test]
    fn verdict_rejects_an_adhesive_margin_of_exactly_zero() {
        // `(design_limit - peak) > 0` is strict. Every step is made exact: masses that give a heat
        // capacity of 250 g * 400 J/(kg K) = 100 J/K, 1 s events, no slip duty (no life events)
        // and a fault trip short enough that the event rise, not the fault rise (which would
        // need exp), sets the peak: peak = T0 + 500 W * 1 s / 100 J/K = 55 + 5 = 60 °C, DP460's limit.
        // The drag is 500 W over the slip speed; the first assert proves it gives 500 W exactly.
        let mut k = links();
        k.mass_magnets_g = 250.0;
        k.mass_cup_g = 0.0;
        k.mass_hub_g = 0.0;
        k.mass_boss_g = 0.0;
        k.hardware_g = 0.0;
        k.retainers_g = 0.0;
        k.endplates_g = 0.0;
        k.cap_g = 0.0;
        k.slip_event_s = 1.0;
        k.life_events = 0.0;
        let mut ti = TemperatureInputs::default();
        ti.thermal.c_ndfeb = 400.0;
        ti.adhesive.selected = 4;
        ti.duty.hot_ambient_C = 50.0;
        ti.duty.driving_rise_C = 5.0;
        ti.duty.fault_trip_s = 0.1;
        let omega = run(&ti, &k).duty.slip_rad_s;
        let drag = 500.0 / omega;
        assert_eq!(drag * omega, 500.0);
        k.measured_drag_Nm = Some(drag);
        let r = run(&ti, &k);
        assert_eq!(
            (r.thermal.heat_capacity_J_K, r.summary.hot_day_start_C),
            (100.0, 55.0)
        );
        assert_eq!(r.magnet_life.peak_C, 60.0);
        assert_eq!(r.adhesive_life.margin_C, 0.0);
        // the other three terms hold: DP460 governs above the hot-day start, the magnets are far off
        assert!(
            r.summary.margin_hot_day_C > 0.0
                && r.magnet_life.margin_limit_C > 0.0
                && r.summary.cure_margin_C >= 10.0
        );
        assert_eq!(r.summary.verdict, VERDICT_CHECK);
        k.measured_drag_Nm = Some(drag * 0.99); // the peak drops to 59.95 °C
        assert_eq!(run(&ti, &k).summary.verdict, VERDICT_OK);
    }
}
