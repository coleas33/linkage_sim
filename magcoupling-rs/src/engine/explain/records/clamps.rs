//! Clamps (plan A-3 batch 5): the shaft clamp's preload and friction torque, per screw size
//! (the 'Clamp screw sizes' table, one family record per column over the five sizes, row 0 =
//! M2.5) and for the size recommended, and the adapter joint (Shaft clamps C14 to C69).
//!
//! A clamp holds by friction: each screw's preload F presses the jaws on the shaft, and the
//! torque one screw holds is μ F d k (friction, preload, shaft diameter, clamp factor). The
//! preload is a share of the screw's proof load, capped by stripping the aluminium thread.
//! Each formula is transcribed from `clamps::compute` with every approved correction on; a
//! screw size's standard data (diameter, stress area, hole, head) are read from the table.

use crate::engine::explain::record::{Family, Record, family, record};

/// The screw table's rows: M2.5, M3, M4, M5, M6.
const ROWS: &[u32] = &[0, 1, 2, 3, 4];

/// One record per screw size for each column the chain reads.
#[rustfmt::skip]
pub const FAMILIES: &[Family] = &[
    // Geometry: the screw axis sits half the bore, the ligament and half the clearance hole off
    // the shaft axis; the head seat, the grip and the far jaw's thread are chords of the boss.
    family(record("clamps.table[#].offset_mm", "e_{#}",
        r#"{clamps.shaft_mm} / 2 + {clamps.ligament_mm} + table("screw_sizes", n, "hole_mm") / 2"#), ROWS),
    family(record("clamps.table[#].wall_out_mm", "w_{out,#}",
        r#"{clamps.boss_radius_mm} - {clamps.table[#].offset_mm} - table("screw_sizes", n, "hole_mm") / 2"#), ROWS),
    family(record("clamps.table[#].head_fits", "fit_{head,#}",
        r#"cases({clamps.table[#].offset_mm} + table("screw_sizes", n, "head_mm") / 2 <= {clamps.boss_radius_mm} => 1; else => 0)"#), ROWS),
    family(record("clamps.table[#].grip_mm", "g_{grip,#}",
        r#"cases({clamps.table[#].head_fits} = 1
                   => sqrt({clamps.boss_radius_mm}^2 - ({clamps.table[#].offset_mm} + table("screw_sizes", n, "head_mm") / 2)^2) - {clamps.slit_mm} / 2;
               else => 0)"#), ROWS),
    family(record("clamps.table[#].thread_avail_mm", "L_{thr,#}",
        "cases({clamps.table[#].offset_mm} < {clamps.boss_radius_mm}
                 => sqrt({clamps.boss_radius_mm}^2 - {clamps.table[#].offset_mm}^2) - {clamps.slit_mm} / 2;
               else => 0)"), ROWS),
    family(record("clamps.table[#].engagement_req_mm", "L_{e,#}", r#"{clamps.engagement_x_d} * table("screw_sizes", n, "d_mm")"#), ROWS),
    family(record("clamps.table[#].geometry_ok", "ok_{geo,#}",
        "cases({clamps.table[#].wall_out_mm} >= {clamps.wall_out_mm} and {clamps.table[#].head_fits} = 1
                 and {clamps.table[#].grip_mm} >= {clamps.grip_min_mm} and {clamps.table[#].thread_avail_mm} >= {clamps.table[#].engagement_req_mm} => 1;
               else => 0)"), ROWS),
    // Strength: the preload from the screw's proof load, and the thread-stripping cap (shear
    // area about 0.6 π d L_e in the aluminium, with a safety factor); the smaller governs.
    family(record("clamps.table[#].preload_strength_N", "F_{b,#}",
        r#"{clamps.preload_fraction} * {clamps.screw_proof_MPa} * table("screw_sizes", n, "As_mm2")"#), ROWS),
    family(record("clamps.table[#].preload_strip_N", "F_{s,#}",
        r#"0.6 * π * table("screw_sizes", n, "d_mm") * {clamps.table[#].engagement_req_mm} * {clamps.al_shear_MPa} / {clamps.strip_sf}"#), ROWS),
    family(record("clamps.table[#].preload_N", "F_{#}", "min({clamps.table[#].preload_strength_N}, {clamps.table[#].preload_strip_N})"), ROWS),
    // Capacity: friction × preload × shaft diameter × clamp factor per screw.
    family(record("clamps.table[#].torque_per_screw_Nm", "T_{per,#}",
        "{clamps.friction} * {clamps.table[#].preload_N} * {clamps.shaft_mm|m} * {clamps.clamp_factor}"), ROWS),
    family(record("clamps.table[#].screws_needed", "n_{need,#}",
        "ceilto(frac({clamps.required_Nm}, {clamps.table[#].torque_per_screw_Nm}), 1)"), ROWS),
    // Fit: screws spaced a head diameter plus 1 mm along the clamp, inside the axial margins.
    family(record("clamps.table[#].screws_fit", "n_{fit,#}",
        r#"cases([s] >= 0 => floorto([s] / (table("screw_sizes", n, "head_mm") + 1), 1) + 1; else => 0)
           where [s] = {clamps.clamp_length_mm} - 2 * {clamps.axial_margin_mm} - (table("screw_sizes", n, "head_mm") + 0.5)"#), ROWS),
    family(record("clamps.table[#].works", "ok_{#}",
        "cases({clamps.table[#].geometry_ok} = 1 and {clamps.table[#].screws_needed} <= {clamps.table[#].screws_fit} => 1; else => 0)"), ROWS),
    family(record("clamps.table[#].clamp_torque_Nm", "T_{cl,#}",
        "{clamps.table[#].screws_needed} * {clamps.table[#].torque_per_screw_Nm}"), ROWS),
    family(record("clamps.table[#].sf_coupling", "S_{#}",
        "{clamps.table[#].screws_needed} * {clamps.table[#].torque_per_screw_Nm} / {clamps.max_torque_Nm}"), ROWS),
    family(record("clamps.table[#].tightening_Nm", "T_{tight,#}",
        r#"{clamps.nut_factor} * {clamps.table[#].preload_N} * table("screw_sizes", n, "d_mm") / 1000"#), ROWS),
];

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    record("clamps.shaft_mm", "d_{shaft}", "{coupling.bore_mm}"),
    record("clamps.boss_radius_mm", "R_{boss}", "{clamps.boss_od_mm} / 2"),
    record("clamps.max_torque_Nm", "T_{max}", "{metal.torque_cold_high_Nm}"),
    record("clamps.required_Nm", "T_{cl,req}", "{clamps.max_torque_Nm} * {clamps.safety_factor}"),
    record("clamps.clamp_factor", "k_{cl}",
        "cases({clamps.clamp_type} = 1 => {clamps.factor_one_piece}; else => {clamps.factor_two_piece})"),
    // ScrewClasses::proof: 12.9, 10.9 or A4-70 (the workbook's CHOOSE order).
    record("clamps.screw_proof_MPa", "σ_p",
        "cases({clamps.screw_class} = 1 => {materials.screws.proof_12_9_MPa}; {clamps.screw_class} = 2 => {materials.screws.proof_10_9_MPa};
               else => {materials.screws.yield_A4_70_MPa})"),
    record("clamps.al_shear_MPa", "τ_{Al,cl}",
        r#"cases({clamps.alloy} = 1 => table("aluminium", "7075-T6", "shear_MPa"); else => table("aluminium", "6061-T6", "shear_MPa"))"#),
    // The first size that works (0 when none does).
    record("clamps.index", "i_{size}",
        "cases({clamps.table[0].works} = 1 => 1; {clamps.table[1].works} = 1 => 2; {clamps.table[2].works} = 1 => 3;
               {clamps.table[3].works} = 1 => 4; {clamps.table[4].works} = 1 => 5; else => 0)"),
    record("clamps.screws", "n_{screws}",
        r#"cases({clamps.index} = 1 => {clamps.table[0].screws_needed}; {clamps.index} = 2 => {clamps.table[1].screws_needed};
               {clamps.index} = 3 => {clamps.table[2].screws_needed}; {clamps.index} = 4 => {clamps.table[3].screws_needed};
               {clamps.index} = 5 => {clamps.table[4].screws_needed}; else => "")"#),
    record("clamps.tightening_Nm", "T_{tight}",
        r#"cases({clamps.index} = 1 => {clamps.table[0].tightening_Nm}; {clamps.index} = 2 => {clamps.table[1].tightening_Nm};
               {clamps.index} = 3 => {clamps.table[2].tightening_Nm}; {clamps.index} = 4 => {clamps.table[3].tightening_Nm};
               {clamps.index} = 5 => {clamps.table[4].tightening_Nm}; else => "")"#),
    record("clamps.capacity_Nm", "T_{cap}",
        r#"cases({clamps.index} = 1 => {clamps.table[0].clamp_torque_Nm}; {clamps.index} = 2 => {clamps.table[1].clamp_torque_Nm};
               {clamps.index} = 3 => {clamps.table[2].clamp_torque_Nm}; {clamps.index} = 4 => {clamps.table[3].clamp_torque_Nm};
               {clamps.index} = 5 => {clamps.table[4].clamp_torque_Nm}; else => "")"#),
    record("clamps.sf_coupling", "S_{cl}",
        r#"cases({clamps.index} = 1 => {clamps.table[0].sf_coupling}; {clamps.index} = 2 => {clamps.table[1].sf_coupling};
               {clamps.index} = 3 => {clamps.table[2].sf_coupling}; {clamps.index} = 4 => {clamps.table[3].sf_coupling};
               {clamps.index} = 5 => {clamps.table[4].sf_coupling}; else => "")"#),
    // The adapter joint: M3 screws on a bolt circle, friction on the nickel-plated face.
    record("clamps.joint_preload_N", "F_{joint}", "{clamps.table[1].preload_N}"),
    record("clamps.joint_torque_Nm", "T_{joint}",
        "{clamps.joint_friction} * {clamps.joint_screws} * {clamps.joint_preload_N} * {clamps.joint_bolt_circle_mm} / 2000"),
    record("clamps.joint_sf", "S_{joint}", "frac({clamps.joint_torque_Nm}, {clamps.required_Nm})"),
];
