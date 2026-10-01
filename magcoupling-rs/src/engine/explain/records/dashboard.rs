//! Dashboard (plan A-3 batch 6): the headline paths outside the five chains (report section
//! 7.1): the gearbox input ripple, the cup OD, the rotating mass, the running clearance and
//! its check, the cup wall check and the recommended clamp screw, with what they need to reach
//! inputs: the back-iron wall rule (Calculator C36, C103, C104, E10), the wall suggestion
//! (decision 27), the sleeve-to-liner clearance and the screw length (E2).
//!
//! Two verdicts are formatted text; the markup states them with `concat`, `fmt` and `fmtnum`,
//! so they are proven like every other record (plan A-3). Each formula is transcribed
//! from the engine function named beside it, corrections on.

use crate::engine::deviations::DeviationId::{E2, E9, E10};
use crate::engine::explain::record::{Family, Record, family, record};

/// One record per screw size.
#[rustfmt::skip]
pub const FAMILIES: &[Family] = &[
    // clamps::compute, E2: the screw crosses the open slit before it reaches the far jaw; the
    // length is rounded up to an even millimetre.
    family(record("clamps.table[#].length_mm", "L_{scr,#}",
        "cases({clamps.table[#].head_fits} = 1
                 => ceilto({clamps.table[#].grip_mm} + {clamps.slit_mm} + {clamps.table[#].engagement_req_mm}, 2);
               else => 0)").corrected(&[E2]), &[0, 1, 2, 3, 4]),
];

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- Gearbox (model::compute) ---
    record("model.gearbox_input_ripple_Nm", "T_{ripple,in}",
        "frac({model.pullout_Nm}, {coupling.gear_ratio} * {coupling.gear_efficiency})"),

    // --- Rotating mass (model::mass_estimate) ---
    record("mass.total_g", "m_{tot}",
        "{mass.magnets_g} + {mass.cup_g} + {mass.hub_g} + {mass.boss_g} + {retainers.retainers_g} + {metal.hardware_g}
         + {retainers.cap_g} + {retainers.endplates_g}"),

    // --- Running clearance (metal_design::compute) ---
    record("metal.sleeve_liner_clearance_mm", "c_{nom}", "({retainers.liner_id_mm} - {retainers.sleeve_od_mm}) / 2"),
    record("metal.adverse_movement_mm", "Σ_{adv}",
        "{metal.shaft_displacement_mm} + {metal.runout_mm} + {metal.deflection_mm} + {metal.thermal_mm} + {metal.sleeve_form_mm}
         + {metal.magnet_position_mm}"),
    record("metal.min_running_clearance_mm", "c_{run}", "{metal.sleeve_liner_clearance_mm} - {metal.adverse_movement_mm}"),
    record("metal.clearance_check", "C_{run}",
        r#"cases({metal.min_running_clearance_mm} < {metal.residual_target_mm} => "Below target"; else => "Meets assumed target")"#),

    // --- The back-iron wall (model::compute, materials::compute) ---
    // The design flux density in effect (decision 20): a library back iron's own, else Materials C13.
    record("model.bsat_T", "B_{des}",
        r#"cases({materials.parts.back_iron} != 1 and table("back_iron", {materials.parts.back_iron}, "design_flux_density_T") != none
                   => table("back_iron", {materials.parts.back_iron}, "design_flux_density_T");
               else => {materials.steel.bsat_T})"#),
    // E10: in the series circuit each magnet contributes its own MMF, B_r t.
    record("model.gap_flux_density_T", "B_{gap}",
        "frac({model.br_inner_T_op} * {model.inner_thickness_mm} + {model.br_outer_T_op} * {model.outer_thickness_mm},
              {model.inner_thickness_mm} + {model.outer_thickness_mm} + {model.face_gap_mm})").corrected(&[E10]),
    // The flux of half a pole, B_gap τ_p / π per unit length, carried by the wall at B_des.
    record("model.backiron_needed_mm", "t_{bi}",
        "{model.gap_flux_density_T} * {model.pole_pitch_mm} / (π * {model.bsat_T})").corrected(&[E10]),
    // Decision 27: the rule's wall, rounded up to 0.1 mm (the number the check's advice quotes).
    record("materials.cup_wall_suggested_mm", "t_{wall,sug}",
        r#"cases({materials.circuit_backiron} = 0 => "n/a"; else => ceilto({model.backiron_needed_mm}, 0.1))"#).corrected(&[E9]),
    record("materials.cup_wall_check", "C_{wall}",
        r#"cases({materials.circuit_backiron} = 0 => "No back iron";
               {metal.cup_wall_corner_mm} >= {model.backiron_needed_mm} => "OK";
               else => concat("Too thin: raise Metal design C122 to at least ", fmt({materials.cup_wall_suggested_mm}, 1), " mm"))"#)
        .corrected(&[E9, E10]),

    // --- The recommended clamp screw (clamps::compute) ---
    record("clamps.recommended", "screw",
        r#"cases({clamps.index} = 0 => "None: enlarge the boss or the clamp length";
               else => concat("ISO 4762 ", table("screw_sizes", {clamps.index} - 1, "name"), " x ", fmtnum([L]), ", class ", [cls]))
           where [L] = cases({clamps.index} = 1 => {clamps.table[0].length_mm}; {clamps.index} = 2 => {clamps.table[1].length_mm};
                             {clamps.index} = 3 => {clamps.table[2].length_mm}; {clamps.index} = 4 => {clamps.table[3].length_mm};
                             else => {clamps.table[4].length_mm}),
                 [cls] = cases({clamps.screw_class} = 1 => "12.9"; {clamps.screw_class} = 2 => "10.9"; else => "A4-70")"#)
        .corrected(&[E2]),
];
