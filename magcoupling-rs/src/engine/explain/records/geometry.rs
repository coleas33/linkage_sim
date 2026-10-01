//! Geometry callouts (plan A-3 batch 7): the numbers the M4 geometry view draws (spec M4
//! "Layout": face gap, corner gap and running clearance; spec A1: the overshoot per axis past
//! the space claim), with the derived dimensions and reserves the overshoots compare
//! (Metal design C134 to C139). Decision G1 of plan A-3 lists them.
//!
//! The face gap, the corner gap and the running clearance already have records (the torque
//! chain and the dashboard); this batch adds the clearance's two parts and the space claim.
//! Each formula is transcribed from `metal_design::compute` and `housing::compute`.

use crate::engine::explain::record::{Record, record};

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- The derived dimensions and their reserves (metal_design::compute) ---
    record("metal.rotating_od_mm", "D_{rot}", "max({model.cup_od_mm}, {metal.cap_od_mm})"),
    record("metal.axial_stack_mm", "L_{stack}",
        "{metal.cap_axial_mm} + {housing.cup_depth_mm} + {metal.web_mm} + {metal.boss_length_mm}"),
    record("metal.large_dia_stack_mm", "L_{large}", "{metal.cap_axial_mm} + {housing.cup_depth_mm} + {metal.web_mm}"),
    record("metal.diameter_reserve_mm", "ΔD", "{metal.max_diameter_mm} - {metal.rotating_od_mm}"),
    record("metal.axial_reserve_mm", "ΔL", "{metal.max_overall_axial_mm} - {metal.axial_stack_mm}"),
    record("metal.large_dia_reserve_mm", "ΔL_{bay}", "{metal.max_large_dia_axial_mm} - {metal.large_dia_stack_mm}"),

    // --- The space claim per axis (housing::compute): how far past the claim, 0 inside it ---
    record("housing.diameter_overshoot_mm", "o_D", "max(-{metal.diameter_reserve_mm}, 0)"),
    record("housing.length_overshoot_mm", "o_L", "max(-{metal.axial_reserve_mm}, 0)"),
    record("housing.bay_overshoot_mm", "o_{bay}", "max(-{metal.large_dia_reserve_mm}, 0)"),
];
