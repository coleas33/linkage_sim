//! Materials sheet ('Materials'): inputs only so far.
//!
//! Port of `reference/magcoupling-py/magcoupling/materials.py`. This module
//! holds the input structs: [`Steel4140`], [`ElectrolessNickel`],
//! [`ScrewClasses`] and their group [`MaterialsInputs`]. The Python
//! `aluminium` member has no metadata, so it is not an input (static data,
//! ported with the sheet's results).
//!
//! Planned deviations touching this sheet (see
//! [`crate::engine::deviations::REGISTRY`]): E9 (Materials!C22), E10 (Materials!C20 to C22).

use super::meta::{inputs, param};

inputs! {
    /// 4140 steel properties (Materials!C13:C19).
    pub struct Steel4140 {
        fields {
            bsat_T: f64 = 1.5 => param("T", "Design flux density for the back-iron check",
                "Annealed 4140. 1018 would be about 1.7 T; use about 1.4 T for pre-hardened stock.", "Materials!C13")
                .range(0.5, 2.2, 0.01)
                .assumption(),
            conductivity_S_m: f64 = 4.5e6 => param("S/m", "Electrical conductivity",
                "Resistivity about 0.22 µΩ·m.", "Materials!C14")
                .range(1e6, 1e7, 1e4)
                .log(),
            mu_r_incremental: f64 = 200.0 => param("-", "Incremental relative permeability (with the magnet bias)",
                "", "Materials!C15")
                .range(1.0, 2000.0, 1.0)
                .log(),
            specific_heat_J_kgK: f64 = 473.0 => param("J/(kg·K)", "Specific heat",
                "", "Materials!C16")
                .range(300.0, 1000.0, 1.0),
            cte_per_C: f64 = 12.3e-6 => param("1/°C", "Expansion coefficient", "", "Materials!C17")
                .range(5e-6, 25e-6, 1e-7),
            modulus_GPa: f64 = 205.0 => param("GPa", "Elastic modulus", "", "Materials!C18")
                .range(50.0, 250.0, 1.0),
            density_g_cm3: f64 = 7.85 => param("g/cm³", "Density",
                "Same as 1018; the mass model uses Metal design C132.", "Materials!C19")
                .range(7.0, 8.2, 0.01),
        }
    }
}

inputs! {
    /// Electroless nickel plating (Materials!C26).
    pub struct ElectrolessNickel {
        fields {
            thickness_mm: f64 = 0.015 => param("mm", "Plating thickness per surface",
                "High-phosphorus EN (10–12 % P): non-magnetic as plated. Typical 0.013–0.025 mm.", "Materials!C26")
                .range(0.0, 0.05, 0.001),
        }
    }
}

inputs! {
    /// Screw class strengths (Materials!C48:C50).
    pub struct ScrewClasses {
        fields {
            proof_12_9_MPa: f64 = 970.0 => param("MPa", "Class 12.9 proof stress",
                "ISO 898-1.", "Materials!C48")
                .range(500.0, 1200.0, 5.0),
            proof_10_9_MPa: f64 = 830.0 => param("MPa", "Class 10.9 proof stress",
                "", "Materials!C49")
                .range(400.0, 1100.0, 5.0),
            yield_A4_70_MPa: f64 = 450.0 => param("MPa", "Stainless A4-70 yield stress",
                "", "Materials!C50")
                .range(200.0, 800.0, 5.0),
        }
    }
}

inputs! {
    /// Every Materials input, grouped as the Python `MaterialsInputs`.
    pub struct MaterialsInputs {
        fields {}
        groups {
            steel: Steel4140,
            nickel: ElectrolessNickel,
            screws: ScrewClasses,
        }
    }
}
