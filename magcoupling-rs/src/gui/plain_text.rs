//! The names and help the panel shows for the inputs and results (guidance slice 2, spec 3.5;
//! rulings R1, S2-4, S2-7): a GUI-side overlay over the engine's metadata, keyed by path.
//!
//! The engine's labels and help keep the workbook wording: `tests/python_schema.rs` holds every
//! ported one to the Python engine byte for byte, and the results table, the exports, design
//! files and share links keep them. On screen, [`label_for`] gives one name per quantity
//! (`LABELS`, spec 3.5's canonical names) and [`help_for`] the overlay's help (`HELP`)
//! where the engine's is empty, terse, stale or wrong; [`original_help`] keeps the original
//! spreadsheet's note for the spreadsheet-details toggle, and [`hint_for`] the short line under
//! a Key design input or a Size-tab control (`HINTS`). Every lookup falls back to the
//! metadata, so a path the overlay leaves out reads as the engine names it.
//!
//! The Size tab's Outside diameter is a control, not an input: it moves the ring radius through
//! `engine::size_controls`, so it has no row here. A fix on the ring radius names that control
//! (`guidance::SIZE_TAB_ALIASES`, which wins over `LABELS` in fix text only); the input row,
//! the equation terms and the tooltips call it Ring radius.
//!
//! Search matches the display label, the catalogue label, the path and the cell
//! (`format::search_haystack`). No egui.

use crate::engine::materials::CUP_WALL_TOO_THIN;
use crate::gui::dashboard::result_info;
use crate::gui::inputs::InputCatalogue;

/// The display label of the required torque (`metal.required_min_Nm`), which the Size tab's
/// slider and the plots' requirement line draw as well.
pub const REQUIRED_TORQUE: &str = "Required torque";

/// The display label of the nominal pull-out (`model.pullout_Nm`), which the plots' pull-out
/// line draws as well.
pub const PULL_OUT_NOMINAL: &str = "Pull-out torque (nominal)";

/// The display label of the hot-low torque (`metal.torque_hot_low_Nm`), which the Size tab's
/// readout draws as well.
pub const HOT_LOW_TORQUE: &str = "Hot-low torque";

/// The provenance line of an input or result the original spreadsheet does not have, drawn in
/// place of a cell with Show spreadsheet details on (map-locks J1, J3, J10).
pub const NO_SPREADSHEET_CELL: &str = "Not in the original spreadsheet";

/// The input whose wall the cup wall verdict asks to raise (ruling R9).
pub const CUP_WALL_INPUT: &str = "metal.cup_wall_corner_mm";

/// A text literal of an equation as the screen draws it (ruling R9, amendment A-9): the cup
/// wall verdict's engine prefix, [`CUP_WALL_TOO_THIN`], with the wall input's label in place of
/// the spreadsheet cell; any other literal as written. The record keeps the engine text, so it
/// still evaluates to the engine's verdict, and the exports keep it.
pub fn shown_literal(text: &str) -> std::borrow::Cow<'_, str> {
    if text == CUP_WALL_TOO_THIN {
        std::borrow::Cow::Owned(format!(
            "Too thin: raise {} to at least ",
            label_for(CUP_WALL_INPUT)
        ))
    } else {
        std::borrow::Cow::Borrowed(text)
    }
}

/// The display label of an input or a result, by path: spec 3.5's canonical names, and the
/// gearbox ripple row's (BL-059: the nominal slip swing, beside the gearbox check, which
/// compares cold-high). A row is never longer than the catalogue label it replaces and names one
/// quantity only (tests). A name other modules draw too is a constant above, so it is typed
/// once.
const LABELS: &[(&str, &str)] = &[
    ("coupling.inner_back_apothem_mm", "Ring radius"),
    ("coupling.npole", "Magnets per ring"),
    ("metal.face_gap_mm", "Magnet face gap"),
    ("metal.required_min_Nm", REQUIRED_TORQUE),
    ("metal.variation", "Production variation allowance"),
    ("model.gap_radius_mm", "Gap radius R_g"),
    ("model.pullout_Nm", PULL_OUT_NOMINAL),
    ("metal.torque_hot_low_Nm", HOT_LOW_TORQUE),
    ("metal.torque_cold_high_Nm", "Cold-high torque"),
    (
        "model.gearbox_input_ripple_Nm",
        "Nominal slip swing at the gearbox input",
    ),
];

// The caveats that end the help of every row in four clusters of `HELP` rows. Each is a macro,
// not a `const`, so that `concat!` can build the rows from it: each caveat is typed once.

/// The last sentence of every stored 3D field's help (amendment A-7): the value is the example
/// design's and stays fixed when the design changes. The engine's sentence, which its own
/// free-space fields end with too.
macro_rules! stored_3d_caveat {
    () => {
        crate::engine::temperature::stored_3d_sentence!()
    };
}

/// The last sentence of every bench-prototype input's help: what the calibration it feeds is for.
macro_rules! prototype_caveat {
    () => {
        "It sets the updated calibration coefficient, used only by no-back-iron designs with the prototype's magnet part and count."
    };
}

/// The last sentence of the help of every default nobody has measured.
macro_rules! guess_caveat {
    () => {
        "The default is a guess."
    };
}

/// The last sentence of every optional-adapter geometry input's help: what it cannot change.
macro_rules! adapter_caveat {
    () => {
        "It changes only the optional adapter variant's results, never the main design."
    };
}

/// The overlay help of an input or a result, by path: 1-2 sentences of at most 30 words, what
/// it physically is, what it moves and which way, and the trade-off (spec 3.5). Filled by the
/// slice-2 help batches; while a path has no row its engine help shows.
const HELP: &[(&str, &str)] = &[
    // Help batch A (slice 2, T7; G-3 review docs/analyses/2026-10-14-magcoupler-help-review.md).
    // Requirements and operating conditions.
    (
        "metal.required_min_Nm",
        "The torque the coupling must hold at the operating magnet temperature. The hot-low torque must reach it; when on, Auto sizes the design to meet it.",
    ),
    (
        "coupling.op_temp_C",
        "The magnet temperature in normal running. Magnets weaken as they warm, so the pull-out and hot-low torque fall as it rises; cold-high uses the minimum temperature instead.",
    ),
    (
        "metal.max_diameter_mm",
        "Largest diameter the coupling may take, over the cup or the front cap, whichever is larger. Past it, the space claim check fails.",
    ),
    (
        "metal.max_overall_axial_mm",
        "Longest the coupling may be, from the front cap to the end of the shaft boss. Past it, the space claim check fails.",
    ),
    (
        "metal.max_large_dia_axial_mm",
        "Longest the full-diameter part may be: front cap, magnet cup and its rear wall, without the shaft boss. Past it, the space claim check fails.",
    ),
    (
        "coupling.drive_torque_Nm",
        "Torque the wheel needs while driving. Times its safety factor it can raise the floor for the nominal pull-out in the results table and sweeps; the hot-low check ignores it.",
    ),
    (
        "coupling.drive_safety_factor",
        "Margin wanted on the driving torque. Driving torque times this factor sets the floor for the nominal pull-out when it exceeds the torque requirement.",
    ),
    (
        "coupling.gear_ratio",
        "Reduction ratio of the gearbox; the coupling sits on its wheel side. A higher ratio lowers the coupling torques referred to the gearbox input, such as the slip swing.",
    ),
    (
        "coupling.gear_efficiency",
        "Share of torque the gearbox passes. With the ratio it refers the coupling torques to the gearbox input; a higher efficiency lowers them there.",
    ),
    (
        "coupling.gearbox_input_rating_Nm",
        "The gearbox's input torque rating. The gearbox check compares the cold-high torque at the gearbox input with it.",
    ),
    (
        "temperature.duty.hot_ambient_C",
        "Air temperature around the coupling on the hottest day; with the driving rise it sets the hot-day magnet temperature. Higher lowers the hot-day pull-out and temperature margins, not hot-low torque.",
    ),
    (
        "temperature.adhesive_life.service_years",
        "Years the coupling must last. It sets only the count of daily temperature cycles reported for the adhesive; no check reads it.",
    ),
    (
        "temperature.adhesive_life.daily_swing_C",
        "How far the coupling's temperature swings each day. A larger swing raises the daily bond shear; above the adhesive's fatigue endurance, the screen asks for thermal-cycling tests.",
    ),
    (
        "metal.slip_rpm",
        "Speed of one ring relative to the other while the coupling slips. Faster slip raises the slip heat, the temperature rise per event and the slip frequency.",
    ),
    (
        "temperature.duty.fault_trip_s",
        "How long an unbroken slip can last before a fault trips the drive. Longer lets the magnets heat more and can raise the slip-fault peak toward the temperature limits.",
    ),
    // Magnets and rings.
    (
        "coupling.npole",
        "Magnet blocks per ring, always even; Auto can set it. At the same ring radius, fewer usually lose torque and, if flat, enlarge the cup; more may not fit.",
    ),
    (
        "coupling.magnets.part_outer",
        "Library part for the outer ring's blocks, matched by exact name. Blank or not in the library uses the manual outer block sizes, with the outer grade if picked.",
    ),
    (
        "coupling.inner_back_apothem_mm",
        "Shaft centre to the back face of the inner magnets. A larger ring adds lever arm, leaves more space between blocks and grows the cup; Auto can set it.",
    ),
    (
        "coupling.magnets.manual_inner_width_mm",
        "Width of one inner block around the ring, used only without a library part. Wider raises torque but must still fit, and on flat blocks takes running clearance.",
    ),
    (
        "coupling.magnets.manual_inner_thickness_mm",
        "Radial thickness of one inner block, used only without a library part. Thicker usually adds torque, with shrinking gains; it needs more steel wall and grows the cup.",
    ),
    (
        "coupling.magnets.manual_inner_br_T",
        "Magnet strength (remanence) of the inner manual blocks, used only with no library part and no grade picked. Stronger raises torque; shaft-joint and gearbox loads rise with it.",
    ),
    (
        "coupling.magnets.manual_outer_length_mm",
        "Axial length of one outer block, used only without a library part. Torque uses the shorter ring, so a longer outer ring alone adds no torque past the inner length.",
    ),
    (
        "coupling.magnets.manual_outer_width_mm",
        "Width of one outer block around the ring, used only without a library part. Wider raises torque but the blocks must still fit side by side.",
    ),
    (
        "coupling.magnets.manual_outer_thickness_mm",
        "Radial thickness of one outer block, used only without a library part. Thicker adds torque only up to a point, needs more steel wall and grows the cup.",
    ),
    (
        "coupling.magnets.manual_outer_br_T",
        "Magnet strength (remanence) of the outer manual blocks, used only with no library part and no grade picked. Stronger raises torque; shaft-joint and gearbox loads rise with it.",
    ),
    // Gap and clearances.
    (
        "metal.face_gap_mm",
        "Radial gap between the inner and outer magnet faces. Smaller raises torque but leaves less running clearance; at wide gaps the torque estimate runs optimistic.",
    ),
    (
        "metal.residual_target_mm",
        "The running clearance you want left after every allowance. The running-clearance check fails when the clearance left is below it.",
    ),
    (
        "metal.shaft_displacement_mm",
        "Allowance for the two shafts moving sideways relative to each other; the default is unconfirmed. It is one of six allowances subtracted from the running clearance.",
    ),
    (
        "metal.runout_mm",
        "Allowance for the assembled rotors running off-centre. It is one of six allowances subtracted from the running clearance.",
    ),
    (
        "metal.deflection_mm",
        "Allowance for the shafts deflecting or tilting under load. It is one of six allowances subtracted from the running clearance.",
    ),
    (
        "metal.thermal_mm",
        "Allowance for the two rotors growing by different amounts as they warm. It is one of six allowances subtracted from the running clearance.",
    ),
    (
        "metal.sleeve_form_mm",
        "Allowance for the sleeve's fit, thickness tolerance and out-of-roundness. It is one of six allowances subtracted from the running clearance.",
    ),
    (
        "metal.magnet_position_mm",
        "Allowance for magnet placement and size tolerances; the default is unconfirmed. It is one of six allowances subtracted from the running clearance.",
    ),
    (
        "metal.sleeve_bedding_mm",
        "Radial clearance between the inner magnets and the bore of the sleeve over them. More enlarges the sleeve and takes running clearance; the torque does not change.",
    ),
    (
        "metal.liner_bedding_mm",
        "Radial clearance between the outer magnet faces and the liner inside them. More shrinks the liner and takes running clearance; the torque does not change.",
    ),
    // Shaft and hub joint.
    (
        "coupling.bore_mm",
        "Diameter of the shaft in the hub bore. Larger thins the hub wall: the hub check can fail, and the wall past the keyway can break through.",
    ),
    (
        "coupling.keyway_depth_mm",
        "Depth of the key slot cut into the hub bore. Deeper leaves a thinner hub wall past the key, and too deep breaks through the hub.",
    ),
    (
        "clamps.key_width_mm",
        "Width of the shaft key. Only the drawings use it; the key pressure uses the contact height instead.",
    ),
    (
        "clamps.key_contact_mm",
        "Height of the key face bearing on the hub. Taller lowers the key bearing pressure, so the key alone holds the clamp's design torque with more margin.",
    ),
    (
        "clamps.clamp_length_mm",
        "Axial length of the clamp from its free end to the relief cut. Longer fits more screws in a row and lowers the key bearing pressure.",
    ),
    (
        "clamps.preload_fraction",
        "Screw preload as a share of proof load (yield for stainless), capped by aluminium thread stripping. Higher grips harder per screw, at a higher tightening torque.",
    ),
    (
        "clamps.slit_mm",
        "Width of the saw slit that lets the clamp close. Wider leaves less jaw and thread for each screw, so fewer sizes may fit, and can need a longer screw.",
    ),
    (
        "clamps.ligament_mm",
        "Least material between the shaft bore and the screw hole. More moves the screws outward, so fewer screw sizes may fit the boss.",
    ),
    (
        "clamps.wall_out_mm",
        "Least wall to leave outside the screw hole. A screw size with a thinner wall is rejected, so more means fewer sizes may fit.",
    ),
    (
        "clamps.grip_min_mm",
        "Least jaw to leave under the screw head. A screw size with less is rejected, so more means fewer sizes may fit.",
    ),
    (
        "clamps.axial_margin_mm",
        "Margin from each clamp end to the nearest counterbore edge. More shortens the row the screws sit in, so fewer may fit along the clamp.",
    ),
    (
        "clamps.relief_mm",
        "Width of the relief cut that leaves the clamp's hinge. Only the clamp layout and drawing show it; no check reads it.",
    ),
    (
        "clamps.factor_two_piece",
        "Share of the screw force that presses a two-piece clamp onto the shaft, used only for that clamp type. Higher means more torque per screw.",
    ),
    (
        "clamps.engagement_x_d",
        "Thread engaged in the aluminium, as a multiple of the screw diameter. More lifts the stripping limit on preload but needs longer screws and more room in the far jaw.",
    ),
    (
        "clamps.strip_sf",
        "Safety factor on stripping the aluminium thread. Higher lowers the stripping cap on screw preload, so more screws may be needed.",
    ),
    // Batch B (T8), empty engine help: Calibration and model.
    (
        "coupling.mu0",
        "Magnetic constant of free space, used in every torque calculation; a physical constant, not a design choice. Leave it unchanged: a larger value lowers every predicted torque.",
    ),
    (
        "calibration.mu0",
        concat!(
            "Magnetic constant for the prototype model; leave it unchanged. ",
            prototype_caveat!()
        ),
    ),
    (
        "calibration.apothem_mm",
        concat!(
            "The bench prototype's radius to its inner magnet backs, not your design's. ",
            prototype_caveat!()
        ),
    ),
    (
        "calibration.magnet_length_mm",
        concat!(
            "Axial length of the bench prototype's magnets, not your design's. ",
            prototype_caveat!()
        ),
    ),
    (
        "calibration.magnet_width_mm",
        concat!(
            "Width of the bench prototype's magnets around the ring, not your design's. ",
            prototype_caveat!()
        ),
    ),
    (
        "calibration.magnet_thickness_mm",
        concat!(
            "Radial thickness of the bench prototype's magnets, not your design's. ",
            prototype_caveat!()
        ),
    ),
    (
        "calibration.br_T",
        concat!(
            "Strength (remanence) of the prototype's magnets at 20 °C, not your design's. ",
            prototype_caveat!()
        ),
    ),
    (
        "calibration.c_end",
        concat!(
            "End-loss coefficient of the bench prototype's model, not your design's. ",
            prototype_caveat!()
        ),
    ),
    (
        "calibration.fea_torque2_Nm",
        "Stored 3D pull-out of the prototype at a 1.5 mm corner gap, interpolated with the 1.0 mm result to check the bench measurement. Your design's torque ignores it.",
    ),
    // Batch B (T8), empty engine help: Materials.
    (
        "materials.steel.mu_r_incremental",
        "Steel's relative permeability under the magnets' field; it sets how deep slip eddy currents reach, and higher means less estimated slip heat in steel. No material pick replaces it.",
    ),
    (
        "materials.steel.specific_heat_J_kgK",
        "Specific heat of the steel parts, including keys, screws and lock tabs; higher slows slip heating, not the steady temperature. Picking another magnetic back-iron material replaces it.",
    ),
    (
        "materials.steel.cte_per_C",
        "Thermal expansion of the steel hub; its difference from the magnets' expansion sets the bond shear from temperature swings, so higher raises it. Picking another back-iron material replaces it.",
    ),
    (
        "materials.steel.modulus_GPa",
        "Stiffness (Young's modulus) of the steel hub; a stiffer hub slightly raises the bond shear from temperature swings. Picking another back-iron material replaces it.",
    ),
    (
        "temperature.mismatch.ndfeb_modulus_GPa",
        "Stiffness (Young's modulus) of the magnets; a stiffer magnet slightly raises the bond shear from temperature swings. Used whatever the magnet grade.",
    ),
    (
        "temperature.slip_loss.sigma_ndfeb_S_m",
        "Electrical conductivity of the magnets: eddy-current heat inside the blocks during slip rises with it. Used whatever the magnet grade, which overstates the heat in ferrite.",
    ),
    (
        "temperature.thermal.c_ndfeb",
        "Specific heat of the magnets; higher slows slip heating, not the steady temperature. Used whatever the magnet grade.",
    ),
    (
        "metal.bond_inner_mm",
        "Adhesive thickness under the inner magnets. The ring radius includes it, so thicker thins the hub; it also lowers the bond shear from temperature swings.",
    ),
    (
        "metal.bond_outer_mm",
        "Adhesive thickness behind the outer magnets. Thicker pushes the cup wall outward, growing the cup diameter and its mass; the torque does not change.",
    ),
    (
        "temperature.mismatch.recommended_bondline_mm",
        "A second adhesive thickness at which the bond shear from temperature swings is also computed; the daily-cycle screen uses it. Thicker lowers that shear.",
    ),
    (
        "temperature.adhesive_life.fatigue_endurance",
        "Share of the adhesive's static strength it keeps for over a hundred million load cycles; the bond fatigue screens pass more easily when it is higher.",
    ),
    (
        "temperature.slip_loss.sigma_316_S_m",
        "Electrical conductivity of the sleeve and liner: their eddy-current heat during slip rises with it. Picking another sleeve and liner material replaces it.",
    ),
    (
        "temperature.thermal.c_316",
        "Specific heat of the sleeve, liner and endplates; higher slows slip heating, not the steady temperature. Picking another sleeve and liner material replaces it.",
    ),
    (
        "temperature.thermal.c_aluminium",
        "Specific heat of the aluminium cap, and of the hub, cup and boss with no back iron; higher slows slip heating. Picking another cap material replaces it for the cap.",
    ),
    (
        "metal.steel_density_g_mm3",
        "Density of the steel hub, cup and boss, which sets their mass and heat capacity. Unused with no back iron; picking another back-iron material replaces it.",
    ),
    (
        "metal.sleeve_density_g_mm3",
        "Density of the sleeve, liner and both endplates, which sets their mass and heat capacity. Picking another sleeve and liner material replaces it.",
    ),
    (
        "metal.al_density_g_mm3",
        "Density of the aluminium cap, the optional adapter and, with no back iron, the hub, cup and boss: their masses. Picking another cap material replaces it for the cap.",
    ),
    (
        "metal.hardware_g",
        "Mass of keys, screws and lock tabs, added to the rotating mass and, as steel, to the heat capacity.",
    ),
    (
        "materials.screws.proof_10_9_MPa",
        "Proof stress of class 10.9 screws, used only for that screw class. Preload is a share of it unless the aluminium thread limits first; more preload grips the shaft harder.",
    ),
    (
        "materials.screws.yield_A4_70_MPa",
        "Yield stress of stainless A4-70 screws, used only for that screw class in place of a proof stress. Preload is a share of it unless the aluminium thread limits first.",
    ),
    // Batch B (T8), empty engine help: Housing and retainers.
    (
        "metal.hub_length_mm",
        "Axial length of the hub under the inner magnets: it sets the hub mass, not the overall length. With an axial magnet length set, it follows the inner ring.",
    ),
    (
        "metal.cup_depth_mm",
        "Depth of the cup's magnet pocket: it adds to the overall length, the large-diameter length and the cup mass. With an axial magnet length set, it follows the outer ring.",
    ),
    (
        "metal.web_mm",
        "Thickness of the cup's closed rear wall: it adds to the overall and large-diameter lengths and the cup mass. With no back iron it also adds slip heat.",
    ),
    (
        "metal.boss_length_mm",
        "Axial length of the shaft boss behind the rear wall: it adds to the overall length and the boss mass, not to the large-diameter length.",
    ),
    (
        "metal.boss_od_mm",
        "Outside diameter of the drawn shaft boss: it sets the boss mass and the outside diameter when larger than the cup and cap.",
    ),
    (
        "metal.cap_axial_mm",
        "Thickness of the front cap at the open end: it adds to the overall and large-diameter lengths, the cap mass and the cap's slip heat.",
    ),
    (
        "metal.cap_od_mm",
        "Front cap diameter; when larger than the cup and boss, it sets the outside diameter. Size-tab diameter changes and Auto on the ring radius move it with the cup.",
    ),
    (
        "metal.cap_thread_engagement_mm",
        "Length of the cap's thread engagement. Only the cap mass reads it: no check covers the thread's strength.",
    ),
    (
        "metal.front_endplate_mm",
        "Thickness of the endplate at the front (open) end of the inner rotor; only the endplate mass reads it, not the overall length.",
    ),
    (
        "metal.rear_endplate_mm",
        "Thickness of the endplate at the rear end of the inner rotor; only the endplate mass reads it, not the overall length.",
    ),
    (
        "metal.retainer_span_mm",
        "Axial length of the sleeve and liner that retain the magnets; only their mass reads it. With an axial magnet length set, it follows the longer ring.",
    ),
    (
        "metal.rear_endplate_hole_mm",
        "Diameter of the screw clearance hole in the rear endplate; a larger hole only lowers the endplate mass.",
    ),
    (
        "metal.adapter_flange_dia_mm",
        concat!(
            "Flange diameter of the optional aluminium adapter that replaces the integral steel boss. ",
            adapter_caveat!()
        ),
    ),
    (
        "metal.adapter_flange_mm",
        concat!(
            "Flange thickness of the optional aluminium adapter; it adds to the adapter variant's length. ",
            adapter_caveat!()
        ),
    ),
    (
        "metal.adapter_pilot_dia_mm",
        concat!(
            "Pilot diameter of the optional aluminium adapter; a larger pilot bores more out of the rear wall. ",
            adapter_caveat!()
        ),
    ),
    (
        "metal.adapter_pilot_mm",
        concat!(
            "Pilot length of the optional aluminium adapter. ",
            adapter_caveat!()
        ),
    ),
    (
        "metal.adapter_boss_mm",
        concat!(
            "Axial boss extension of the optional aluminium adapter; it adds to the adapter variant's length. ",
            adapter_caveat!()
        ),
    ),
    (
        "metal.adapter_hardware_g",
        concat!(
            "Extra screw and dowel mass for the optional adapter's joint. ",
            adapter_caveat!()
        ),
    ),
    (
        "clamps.joint_bolt_circle_mm",
        "Bolt-circle diameter of the optional adapter's joint screws; a larger circle raises the joint's slip torque and its margin over the clamp design torque.",
    ),
    (
        "clamps.joint_friction",
        "Friction between the nickel-plated faces of the optional adapter's joint; the joint's slip torque rises in proportion. It is not the clamp's shaft-to-bore friction.",
    ),
    // Batch B (T8), empty engine help: Thermal and demagnetization.
    (
        "temperature.demag.knee_fraction",
        "Reverse field where a magnet starts losing strength for good, as a share of its coercivity (Hcj). Higher raises the magnet design limit and lowers ferrite's cold magnet limit.",
    ),
    (
        "temperature.demag.design_margin_C",
        "Degrees kept between the temperature where demagnetization starts and the magnet design limit; larger lowers that limit and raises ferrite's cold magnet limit.",
    ),
    (
        "temperature.slip_loss.end_factor",
        "End correction on the slip heat of thin parts: the sleeve, liner and cap, and with no back iron the aluminium hub, cup and web. Higher means more heat.",
    ),
    // Batch B (T8), overrides: the grade-mode density, rating and coefficient helps.
    (
        "calibration.alpha_br_per_C",
        "Change in magnet strength (Br) per °C; more negative means torque falls faster above 20 °C and rises faster below. A ring with a picked grade uses its grade's value.",
    ),
    (
        "mass.magnets_g",
        "Mass of every magnet in both rings: NdFeB density for library parts and ungraded magnets, the grade's own density for a ring with a picked grade.",
    ),
    (
        "model.inner_temp_check",
        "Compares the operating magnet temperature with the inner ring's rating: its library part's or its picked grade's. Unknown for a manual magnet with no grade.",
    ),
    (
        "model.outer_temp_check",
        "Compares the operating magnet temperature with the outer ring's rating: its library part's or its picked grade's. Unknown for a manual magnet with no grade.",
    ),
    // Batch B (T8), overrides: the stored 3D fields ("3D." and the reverse fields).
    (
        "temperature.slip_loss.b_hub_T",
        concat!(
            "Outer-ring field at the hub's steel surface, doubled by the steel; hub slip heat grows with its square. ",
            stored_3d_caveat!()
        ),
    ),
    (
        "temperature.slip_loss.b_cup_T",
        concat!(
            "Inner-ring field at the cup's steel surface, doubled by the steel; cup slip heat grows with its square. ",
            stored_3d_caveat!()
        ),
    ),
    (
        "temperature.slip_loss.b_sleeve_T",
        concat!(
            "Outer-ring field at the inner sleeve; the sleeve's slip heat grows with its square. ",
            stored_3d_caveat!()
        ),
    ),
    (
        "temperature.slip_loss.b_liner_T",
        concat!(
            "Inner-ring field at the outer liner; the liner's slip heat grows with its square. ",
            stored_3d_caveat!()
        ),
    ),
    (
        "temperature.slip_loss.cap_integral_T2m4",
        concat!(
            "Field integral over the cap face; the cap's slip heat grows in proportion. ",
            stored_3d_caveat!()
        ),
    ),
    (
        "temperature.slip_loss.web_integral_T2m2",
        concat!(
            "Squared-field integral over the rear wall, field doubled by the steel; the wall's slip heat grows in proportion. ",
            stored_3d_caveat!()
        ),
    ),
    (
        "temperature.slip_loss.b_magnet_T",
        concat!(
            "Alternating field inside the magnet blocks during slip; their eddy-current heat grows with its square. ",
            stored_3d_caveat!()
        ),
    ),
    (
        "temperature.demag.h_rev_aligned_kA_m",
        concat!(
            "Worst reverse field in the magnets, rings aligned; it sets only that case's onset, which no check uses. ",
            stored_3d_caveat!()
        ),
    ),
    (
        "temperature.demag.h_rev_pullout_kA_m",
        concat!(
            "Worst reverse field in the magnets at pull-out; it sets only that case's onset, which no check uses. ",
            stored_3d_caveat!()
        ),
    ),
    (
        "temperature.demag.h_rev_likepole_kA_m",
        concat!(
            "Worst reverse field as like poles pass; higher lowers the magnet design limit (raises ferrite's cold limit). ",
            stored_3d_caveat!()
        ),
    ),
    (
        "temperature.demag.h_rev_single_ring_kA_m",
        concat!(
            "Worst reverse field on a single ring during adhesive cure; higher lowers the cure margin. ",
            stored_3d_caveat!()
        ),
    ),
    // Batch B (T8), overrides: "Placeholder", "ASSUMED" and "Illustrative".
    (
        "temperature.duty.driving_rise_C",
        concat!(
            "Rise of the coupling above outside air while driving, from housing air, sun and gearbox heat; it raises the hot-day starting magnet temperature. ",
            guess_caveat!()
        ),
    ),
    (
        "temperature.duty.life_hours",
        concat!(
            "Hours the coupling runs over its life; slip time divided by it gives the average slip duty, which adds to the peak temperature. ",
            guess_caveat!()
        ),
    ),
    (
        "temperature.adhesive_life.hot_strength_retained",
        concat!(
            "Share of the adhesive's lap-shear strength left at the peak temperature; the hot fatigue screen scales with it. ",
            guess_caveat!()
        ),
    ),
    (
        "calibration.test_temp_C",
        concat!(
            "Magnet temperature during the prototype's bench test; enter the recorded value. ",
            prototype_caveat!()
        ),
    ),
    (
        "metal.slip_event_s",
        "How long one slip event lasts; heat per event, slip duty and life cycles grow with it. The default is a guess: enter the recorded value.",
    ),
    // Batch B (T8), overrides: developer wording, jargon, stale or wrong facts.
    (
        "metal.measured_drag_Nm",
        "Mean drag torque measured on the bench while slipping; leave blank if not measured. Once entered, it replaces the estimated slip heat in every temperature check.",
    ),
    (
        "materials.steel.density_g_cm3",
        "For reference only: no result reads it. The steel parts' mass uses the steel density listed with the other part densities.",
    ),
    (
        "temperature.demag.hcj20_kA_m",
        "Coercivity (Hcj) at 20 °C; higher raises the magnet design limit. Used only for ungraded manual magnets, or when the coercivity choice selects these inputs.",
    ),
    (
        "temperature.demag.beta_hcj_per_C",
        "How fast coercivity changes per °C (negative for NdFeB). Used only for ungraded manual magnets or when the coercivity choice selects these inputs; ferrite needs the Y30 grade instead.",
    ),
    (
        "temperature.mismatch.adhesive_shear_modulus_GPa",
        "Shear stiffness of the cured adhesive; stiffer raises the bond shear from temperature swings. It does not follow the selected adhesive: enter your adhesive's value.",
    ),
    (
        "metal.cup_wall_corner_mm",
        "Cup wall at the magnet pocket corners, its thinnest point. Thinner than the back iron needs fails the cup wall check; thicker grows the cup diameter by twice the change.",
    ),
    (
        "materials.steel.bsat_T",
        "Flux density the back-iron wall rule designs the steel to; lower asks for thicker cup and hub walls. Picking another back-iron material with its own value replaces it.",
    ),
    (
        "materials.steel.conductivity_S_m",
        "Electrical conductivity of the back-iron steel: slip heat in a steel hub, cup and web rises with it. Picking another back-iron material replaces it.",
    ),
    (
        "temperature.mismatch.ndfeb_cte_per_C",
        "Thermal expansion of the magnets in the bonded face; the further it is from the hub's, the higher the bond shear from temperature swings. Used whatever the magnet grade.",
    ),
    (
        "temperature.slip_loss.high_multiplier",
        "Multiplies the estimated slip heat for the high case, which the temperature checks use. Ignored once a measured drag is entered.",
    ),
    (
        "clamps.joint_screws",
        "Number of M3 screws in the optional adapter's joint, beside its pilot and dowel; the joint's slip torque rises in proportion.",
    ),
    // BL-059: the ripple row, read beside the gearbox check (G-3 review
    // docs/analyses/2026-10-14-magcoupler-help-review.md, BL-059 section).
    (
        "model.gearbox_input_ripple_Nm",
        "Slip swing: the pull-out at the gearbox output at operating temperature, without the production allowance, divided by ratio and efficiency. The gearbox check compares cold-high at the gearbox input instead.",
    ),
];

/// The hint key of the Size tab's Minimum OD control: it edits no input, so its hint line is
/// keyed by name (spec 3.5).
pub const MINIMUM_OD_HINT_KEY: &str = "size_tab.minimum_od";

/// The hint key of the Size tab's Outside diameter control: it edits the ring radius, but its
/// hint says what the diameter is, so it does not use that input's path.
pub const DIAMETER_HINT_KEY: &str = "size_tab.outside_diameter";

/// The hint key of the Size tab's Overall length control: it edits the axial magnet length,
/// whose Key design row's hint means the magnet length itself, so it does not use that path.
pub const LENGTH_HINT_KEY: &str = "size_tab.overall_length";

/// The [`HINTS`] keys that are not input paths: the three Size-tab controls above. The Size
/// tab's Magnet face gap and Magnets per ring draw their inputs' own hints.
pub const SIZE_TAB_HINT_KEYS: [&str; 3] = [MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY];

/// The hint line under a Key design input or a Size-tab control, by input path or Size-tab hint
/// key ([`SIZE_TAB_HINT_KEYS`]): at most 12 words (spec 3.5). Filled by the slice-2 help batches.
const HINTS: &[(&str, &str)] = &[
    // The Key design group (inputs::KEY_DESIGN), in its order. The Size tab's Magnet face gap and
    // Magnets per ring draw the first two.
    (
        "metal.face_gap_mm",
        "Smaller gap: more torque, less running clearance.",
    ),
    (
        "coupling.npole",
        "Fewer blocks usually lose torque; more may not fit.",
    ),
    (
        "coupling.magnets.part_inner",
        "Sets the inner blocks' size, strength and temperature rating.",
    ),
    (
        "coupling.magnets.part_outer",
        "Sets the outer blocks' size, strength and temperature rating.",
    ),
    (
        "coupling.magnets.axial_length_mm",
        "Blank keeps ring lengths; longer adds torque, length, shaft-joint and gearbox loads.",
    ),
    (
        "coupling.op_temp_C",
        "Hotter magnets are weaker: pull-out and hot-low torque fall.",
    ),
    (
        "coupling.backiron",
        "Magnetic steel behind the magnets adds torque if thick enough.",
    ),
    (
        "metal.cup_wall_corner_mm",
        "Thicker passes the wall check more easily but grows the cup.",
    ),
    (
        "temperature.thermal.conductance_W_K",
        "Higher lowers the slip temperatures; best measured on a bench.",
    ),
    (
        "metal.measured_drag_Nm",
        "Blank uses the model's estimate; a bench value replaces it.",
    ),
    // The Size tab's other three controls (SIZE_TAB_HINT_KEYS).
    (
        MINIMUM_OD_HINT_KEY,
        "Sets the diameter slider's lower end; your design does not change.",
    ),
    (
        DIAMETER_HINT_KEY,
        "Largest of cup, cap and boss; it moves the magnet rings.",
    ),
    (
        LENGTH_HINT_KEY,
        "Sets both rings' magnet length; longer adds torque, shaft-joint and gearbox loads.",
    ),
];

/// The text of `path`'s row in `table`, if it has one.
fn row(table: &[(&'static str, &'static str)], path: &str) -> Option<&'static str> {
    table
        .iter()
        .find(|(key, _)| *key == path)
        .map(|(_, text)| *text)
}

/// The metadata label and help of an input or a result path; `None` for neither.
fn metadata(path: &str) -> Option<(&'static str, &'static str)> {
    result_info(path)
        .map(|info| (info.meta.label, info.meta.help))
        .or_else(|| {
            InputCatalogue::get()
                .entry(path)
                .map(|entry| (entry.meta.label, entry.meta.help))
        })
}

/// The display label of the input or result at `path`: its `LABELS` row, else its metadata
/// label; `""` for a path that is neither, never the path itself.
pub fn label_for(path: &str) -> &'static str {
    row(LABELS, path).unwrap_or_else(|| metadata(path).map_or("", |(label, _)| label))
}

/// [`label_for`] for a caller that holds the path's metadata label `catalogue`: the `LABELS`
/// row, else `catalogue`. The input catalogue builds its search text with it while it is being
/// built, when [`label_for`] cannot ask it; C-3 compares every input's with it.
pub(crate) fn display_label(path: &str, catalogue: &'static str) -> &'static str {
    row(LABELS, path).unwrap_or(catalogue)
}

/// The help of the input or result at `path`: its `HELP` row, else its metadata help; `""`
/// for none.
pub fn help_for(path: &str) -> &'static str {
    help_in(HELP, path)
}

/// The original spreadsheet's note of `path` where `HELP` replaces the engine's help, shown
/// under the spreadsheet-details toggle as "Original spreadsheet note: ...": the engine's help,
/// or the workbook's own text where a correction reworded it; `None` where that note is empty
/// or the overlay has no row.
pub fn original_help(path: &str) -> Option<&'static str> {
    original_in(HELP, path)
}

/// The hint line under a Key design input or a Size-tab control, if `HINTS` has one: `path` is
/// the input's path or one of the [`SIZE_TAB_HINT_KEYS`].
pub fn hint_for(path: &str) -> Option<&'static str> {
    row(HINTS, path)
}

/// [`help_for`] over `table`.
fn help_in(table: &[(&'static str, &'static str)], path: &str) -> &'static str {
    row(table, path).unwrap_or_else(|| metadata(path).map_or("", |(_, help)| help))
}

/// [`original_help`] over `table`: the original spreadsheet's note of the help a `table` row
/// replaces. That is the engine's help, except where a correction reworded it: then the
/// workbook's own text, which the deviation registry keeps (`spreadsheet_help`). `None` when
/// that note is empty or `table` has no row for `path`.
fn original_in(table: &[(&'static str, &'static str)], path: &str) -> Option<&'static str> {
    row(table, path)?;
    spreadsheet_help(path)
        .or_else(|| metadata(path).map(|(_, help)| help))
        .filter(|help| !help.is_empty())
}

/// The workbook's text of a help a correction reworded, as the deviation registry records it
/// (`Deviation::workbook_help`, possibly empty); `None` for a help the port kept as the
/// workbook wrote it.
fn spreadsheet_help(path: &str) -> Option<&'static str> {
    crate::engine::deviations::REGISTRY
        .iter()
        .flat_map(|deviation| deviation.workbook_help)
        .find(|(key, _)| *key == path)
        .map(|(_, text)| *text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::explain::plain::{
        mentions_gearbox_protection, plain_text_violations, sentence_count, word_count,
    };
    use crate::gui::results_table::table_entries;
    use crate::gui::size_controls::{GAP, MAGNETS};
    use crate::gui::test_support::assert_glyphs;
    use std::collections::BTreeSet;

    /// Every row of the three tables, tagged with its table.
    fn every_row() -> Vec<(&'static str, &'static str, &'static str)> {
        [("label", LABELS), ("help", HELP), ("hint", HINTS)]
            .into_iter()
            .flat_map(|(table, rows)| rows.iter().map(move |&(path, text)| (table, path, text)))
            .collect()
    }

    /// Every input's path, then every results-table row's: each path a label and a help are
    /// shown for.
    fn every_path() -> impl Iterator<Item = &'static str> {
        let inputs = InputCatalogue::get().all().map(|entry| entry.path.as_str());
        let results = table_entries().iter().map(|entry| entry.path.as_str());
        inputs.chain(results)
    }

    fn is_input(path: &str) -> bool {
        InputCatalogue::get().entry(path).is_some()
    }

    fn is_result(path: &str) -> bool {
        result_info(path).is_some()
    }

    #[test]
    fn every_overlay_row_names_a_real_path_once() {
        // Spec 3.5 coverage test, its always-on half: no stray key, no key twice, no empty
        // text, and no help row that only repeats the engine's help. A hint sits under an
        // input row or a Size-tab control, so its key is an input or one of the Size tab's
        // own hint keys (SIZE_TAB_HINT_KEYS: those controls are not inputs).
        for (table, rows, inputs_only) in [
            ("label", LABELS, false),
            ("help", HELP, false),
            ("hint", HINTS, true),
        ] {
            let mut seen = BTreeSet::new();
            for &(path, text) in rows {
                assert!(seen.insert(path), "{table}: {path} twice");
                assert!(
                    is_input(path)
                        || (!inputs_only && is_result(path))
                        || (table == "hint" && SIZE_TAB_HINT_KEYS.contains(&path)),
                    "{table}: {path} is no input{}",
                    if inputs_only {
                        " or Size-tab hint key"
                    } else {
                        " or result"
                    }
                );
                assert!(!text.trim().is_empty(), "{table}: {path} is empty");
            }
        }
        for &(path, text) in HELP {
            assert_ne!(
                Some(text),
                metadata(path).map(|(_, help)| help),
                "{path}: the row repeats the engine's help"
            );
        }
    }

    #[test]
    fn the_canonical_names_are_the_spec_s() {
        // Spec 3.5's table: one name per quantity. The spec writes the ring radius's meaning
        // in parentheses ("the shaft centre to the back of the inner magnets"); that is help
        // (T7), the name is "Ring radius". BL-059 adds the gearbox ripple row's: the nominal
        // slip swing, read beside the gearbox check, which compares cold-high instead.
        let want = [
            ("coupling.inner_back_apothem_mm", "Ring radius"),
            ("coupling.npole", "Magnets per ring"),
            ("metal.face_gap_mm", "Magnet face gap"),
            ("metal.required_min_Nm", "Required torque"),
            ("metal.variation", "Production variation allowance"),
            ("model.gap_radius_mm", "Gap radius R_g"),
            ("model.pullout_Nm", "Pull-out torque (nominal)"),
            ("metal.torque_hot_low_Nm", "Hot-low torque"),
            ("metal.torque_cold_high_Nm", "Cold-high torque"),
            (
                "model.gearbox_input_ripple_Nm",
                "Nominal slip swing at the gearbox input",
            ),
        ];
        for (path, label) in want {
            assert_eq!(label_for(path), label, "{path}");
        }
        assert_eq!(LABELS.len(), want.len(), "no row beyond these");
        // The Size tab names its gap and magnet-count controls as the input rows do.
        assert_eq!(GAP, label_for("metal.face_gap_mm"));
        assert_eq!(MAGNETS, label_for("coupling.npole"));
    }

    #[test]
    fn label_for_falls_back_to_the_metadata_and_never_to_the_path() {
        // An input and a result the overlay leaves out read as the engine names them.
        let drive = InputCatalogue::get()
            .entry("coupling.drive_torque_Nm")
            .unwrap();
        assert_eq!(label_for("coupling.drive_torque_Nm"), drive.meta.label);
        let f_end = result_info("model.f_end").unwrap();
        assert_eq!(label_for("model.f_end"), f_end.meta.label);
        // A path that is neither has no name on screen: "", never the path.
        assert_eq!(label_for("no.such"), "");
        assert_eq!(label_for(""), "");
        // Every overlay label replaces its catalogue label and is never longer (the inputs
        // side is 320 points wide, audit UI-34); display_label agrees with label_for.
        for &(path, label) in LABELS {
            let (catalogue, _) = metadata(path).unwrap();
            assert_ne!(label, catalogue, "{path}");
            assert!(
                label.chars().count() <= catalogue.chars().count(),
                "{path}: {label:?} is longer than {catalogue:?}"
            );
            assert_eq!(display_label(path, catalogue), label, "{path}");
        }
        assert_eq!(
            display_label("coupling.drive_torque_Nm", drive.meta.label),
            drive.meta.label
        );
    }

    /// A stand-in overlay: the mechanism is tested on it, apart from what `HELP` holds. One row
    /// over an input with engine help, one over an input without, one over a result, and two
    /// over helps a correction reworded: an input whose workbook text the deviation registry
    /// keeps, and a result whose workbook text was empty.
    const TEST_HELP: &[(&str, &str)] = &[
        (
            "metal.face_gap_mm",
            "Stand-in help over a non-empty engine help.",
        ),
        (
            "metal.runout_mm",
            "Stand-in help over an empty engine help.",
        ),
        ("model.pullout_Nm", "Stand-in help over a result's help."),
        (
            "clamps.boss_od_mm",
            "Stand-in help over a help a correction reworded.",
        ),
        (
            "temperature.slip_loss.hub_W",
            "Stand-in help over a reworded help the workbook left empty.",
        ),
    ];

    #[test]
    fn help_reads_the_overlay_then_the_engine_and_keeps_the_original() {
        let engine = |path: &str| metadata(path).unwrap().1;
        // The stand-in rows sit over what the test says they sit over.
        assert!(!engine("metal.face_gap_mm").is_empty());
        assert!(engine("metal.runout_mm").is_empty());
        assert!(!engine("model.pullout_Nm").is_empty());
        for &(path, text) in TEST_HELP {
            assert_eq!(help_in(TEST_HELP, path), text, "{path}");
        }
        // Over a help no correction reworded, the original is kept only where the overlay
        // replaces a non-empty engine help.
        assert_eq!(spreadsheet_help("metal.face_gap_mm"), None);
        assert_eq!(
            original_in(TEST_HELP, "metal.face_gap_mm"),
            Some(engine("metal.face_gap_mm"))
        );
        assert_eq!(original_in(TEST_HELP, "metal.runout_mm"), None);
        assert_eq!(
            original_in(TEST_HELP, "model.pullout_Nm"),
            Some(engine("model.pullout_Nm"))
        );
        // Where a correction reworded the help, the original is the workbook's own text, which
        // the registry keeps, not the corrected engine help; none where the workbook had none.
        let boss = original_in(TEST_HELP, "clamps.boss_od_mm").expect("the workbook's text");
        assert_eq!(Some(boss), spreadsheet_help("clamps.boss_od_mm"));
        assert_ne!(boss, engine("clamps.boss_od_mm"));
        assert_eq!(spreadsheet_help("temperature.slip_loss.hub_W"), Some(""));
        assert!(!engine("temperature.slip_loss.hub_W").is_empty());
        assert_eq!(original_in(TEST_HELP, "temperature.slip_loss.hub_W"), None);
        // A path the overlay leaves out shows the engine's help and has no original.
        assert_eq!(
            help_in(TEST_HELP, "coupling.npole"),
            engine("coupling.npole")
        );
        assert_eq!(original_in(TEST_HELP, "coupling.npole"), None);
        // A path that is neither: no help, no original, no hint.
        assert_eq!(help_in(TEST_HELP, "no.such"), "");
        assert_eq!(original_in(TEST_HELP, "no.such"), None);
        assert_eq!(hint_for("no.such"), None);
        // The public accessors read HELP over each input's own engine help: its row where it
        // has one (the spreadsheet's note kept as the original only where it is not empty: the
        // registry's workbook text for a reworded help, else the engine's), else the engine's
        // help and no original.
        for entry in InputCatalogue::get().all() {
            let path = entry.path.as_str();
            let engine = entry.meta.help;
            match row(HELP, path) {
                Some(text) => {
                    let note = spreadsheet_help(path).unwrap_or(engine);
                    assert_eq!(help_for(path), text, "{path}");
                    assert_eq!(
                        original_help(path),
                        Some(note).filter(|help| !help.is_empty()),
                        "{path}"
                    );
                }
                None => {
                    assert_eq!(help_for(path), engine, "{path}");
                    assert_eq!(original_help(path), None, "{path}");
                }
            }
        }
    }

    #[test]
    fn every_overlay_text_is_plain_words_with_glyphs_and_within_its_caps() {
        // Spec 3.5 and 6 test 9: every label, help and hint passes the plain-words guard and
        // the glyph test, calls no torque guaranteed (slice-1 spec 8), and keeps the caps (help
        // <= 30 words in 1-2 sentences, a hint <= 12 words in one, a label a name, not a
        // sentence). The counters are plain.rs's (A-16).
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        for (table, path, text) in every_row() {
            let what = format!("{table} of {path}");
            assert_eq!(
                plain_text_violations(text),
                Vec::<&str>::new(),
                "{what}: {text:?}"
            );
            assert_glyphs(&ctx, text, &what);
            for c in ['\u{3d1}', '\u{207b}', '\u{221d}', '\u{2192}', '\u{2264}'] {
                assert!(!text.contains(c), "{what}: {c:?}; write -> and <=");
            }
            assert!(
                !text.contains('\n') && text.trim() == text,
                "{what}: one line, no outer blanks"
            );
            // Overlay text is drawn as written: a brace would read as an unfilled live value.
            assert!(!text.contains(['{', '}']), "{what}: a brace");
            assert!(
                !text.to_lowercase().contains("guarantee"),
                "{what}: no torque is called guaranteed"
            );
            let words = word_count(text);
            let sentences = sentence_count(text);
            match table {
                "label" => assert!(!text.ends_with('.'), "{what}: a name, not a sentence"),
                "help" => {
                    assert!(words <= 30, "{what}: {words} words");
                    assert!(text.ends_with('.'), "{what}: a sentence");
                    assert!(
                        (1..=2).contains(&sentences),
                        "{what}: {sentences} sentences"
                    );
                }
                _ => {
                    assert!(words <= 12, "{what}: {words} words");
                    assert_eq!(sentences, 1, "{what}: {sentences} sentences");
                }
            }
        }
    }

    #[test]
    fn no_label_help_or_hint_says_the_coupling_protects_the_gearbox() {
        // BL-059 and slice-2 spec section 8: a pass of the gearbox check means only that the
        // modelled cold-high peak is within the gearbox input rating. Every overlay row, and
        // every engine label and help shown where the overlay has none (the gearbox check's
        // own among them), keeps to the rule the notes and the guides keep.
        let shown = every_path().flat_map(|path| [label_for(path), help_for(path)]);
        let overlay = every_row().into_iter().map(|(_, _, text)| text);
        for text in overlay.chain(shown) {
            assert!(!mentions_gearbox_protection(text), "{text:?}");
        }
    }

    #[test]
    fn each_display_label_names_one_quantity() {
        // No overlay label is any other input's or result's name on screen, ignoring case.
        let all: Vec<&str> = every_path().collect();
        for &(path, label) in LABELS {
            for &other in all.iter().filter(|&&other| other != path) {
                assert!(
                    !label_for(other).eq_ignore_ascii_case(label),
                    "{path} and {other} both read {label:?}"
                );
            }
        }
    }

    #[test]
    fn every_input_has_help_from_the_engine_or_the_overlay() {
        // Spec 3.5 coverage test, its other half: 87 inputs have no help at b0166ad.
        let missing: Vec<&str> = InputCatalogue::get()
            .all()
            .map(|entry| entry.path.as_str())
            .filter(|path| help_for(path).is_empty())
            .collect();
        assert!(
            missing.is_empty(),
            "{} inputs without help: {missing:?}",
            missing.len()
        );
    }

    #[test]
    fn rust_only_inputs_keep_their_engine_help() {
        // Ruling R1: a Rust-only input has no spreadsheet note, and its plain help is written in
        // the engine (T9); the overlay never covers one, so no "Original spreadsheet note" shows.
        let rust_only: Vec<&crate::gui::inputs::InputEntry> = InputCatalogue::get()
            .all()
            .filter(|entry| entry.meta.rust_only)
            .collect();
        assert!(!rust_only.is_empty());
        for entry in rust_only {
            assert_eq!(help_for(&entry.path), entry.meta.help, "{}", entry.path);
            assert_eq!(original_help(&entry.path), None, "{}", entry.path);
        }
    }

    /// Prints every overlay row beside the engine text it replaces, for the G-3 physics review
    /// of each batch. Run from the repo root:
    /// `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib overlay_strings_for_review -- --ignored --nocapture`.
    #[test]
    #[ignore = "prints the overlay for the physics review; run by hand"]
    fn overlay_strings_for_review() {
        let rows = every_row();
        for &(table, path, text) in &rows {
            let (label, help) = metadata(path).unwrap_or(("", ""));
            let was = match table {
                "label" => label,
                "help" => help,
                _ => "",
            };
            println!("[{table}] {path}\n  now: {text}\n  was: {was}");
        }
        println!("TOTAL ROWS: {}", rows.len());
    }
}

/// Help batch A (slice 2, T7) and the hint lines: batch A's coverage, the hint keys, no number
/// and no typed label in them, and the G-3 review sheet. Every row's key, caps, plain words and
/// glyphs are the `tests` module's checks; the coverage of every input too.
#[cfg(test)]
mod help_tests {
    use super::*;
    use crate::gui::inputs::{KEY_DESIGN, WORKFLOW};

    /// The workflow groups of help batch A: their inputs' help is this batch's.
    const BATCH_A: [&str; 4] = ["operating", "magnets", "gap", "shaft"];

    /// The inputs of the workflow group `id`, in its sections' order.
    fn group_paths(id: &str) -> Vec<&'static str> {
        WORKFLOW
            .iter()
            .find(|group| group.id == id)
            .unwrap_or_else(|| panic!("no workflow group {id}"))
            .sections
            .iter()
            .flat_map(|section| section.paths.iter().copied())
            .collect()
    }

    /// The inputs of batch A, every group in turn.
    fn batch_a_paths() -> Vec<&'static str> {
        BATCH_A
            .iter()
            .flat_map(|group| group_paths(group))
            .collect()
    }

    #[test]
    fn batch_a_gives_every_input_of_its_groups_help() {
        let missing: Vec<&str> = batch_a_paths()
            .into_iter()
            .filter(|path| help_for(path).trim().is_empty())
            .collect();
        assert!(
            missing.is_empty(),
            "{} inputs without help: {missing:?}",
            missing.len()
        );
    }

    #[test]
    fn hints_cover_the_key_design_inputs_and_the_size_tab_controls_only() {
        let mut wanted: Vec<&str> = KEY_DESIGN.to_vec();
        wanted.extend(SIZE_TAB_HINT_KEYS);
        let missing: Vec<&str> = wanted
            .iter()
            .copied()
            .filter(|key| hint_for(key).is_none())
            .collect();
        assert!(missing.is_empty(), "no hint for {missing:?}");
        for &(key, _) in HINTS {
            assert!(wanted.contains(&key), "{key}: a hint no control draws");
        }
        // The Size tab's own keys are not input paths: no input row can draw them by mistake.
        let catalogue = InputCatalogue::get();
        for key in SIZE_TAB_HINT_KEYS {
            assert!(catalogue.entry(key).is_none(), "{key} names an input");
        }
    }

    #[test]
    fn batch_a_help_and_every_hint_type_no_number() {
        // Spec 3.5: no default values (the tooltip prints the default), cells or audit ids.
        let batch = batch_a_paths();
        let rows = HELP
            .iter()
            .filter(|&&(key, _)| batch.contains(&key))
            .chain(HINTS);
        for &(key, text) in rows {
            assert!(!text.chars().any(|c| c.is_ascii_digit()), "{key}: {text}");
        }
    }

    #[test]
    fn no_hint_types_an_input_label() {
        // Contract: an input's label comes from label_for at display time, never typed.
        for &(key, text) in HINTS {
            let hint = text.to_lowercase();
            for entry in InputCatalogue::get().all() {
                for label in [entry.meta.label, label_for(&entry.path)] {
                    assert!(
                        label.is_empty() || !hint.contains(&label.to_lowercase()),
                        "{key}: {text:?} types the label {label:?}"
                    );
                }
            }
        }
    }

    /// The G-3 review sheet of `groups` (workflow group ids): each input's display label and
    /// path, the help shown and, for an overlay row, the engine help it replaces; then every
    /// hint line.
    fn review_sheet(groups: &[&str]) -> String {
        use std::fmt::Write as _;
        let catalogue = InputCatalogue::get();
        let mut sheet = String::new();
        for &group in groups {
            let workflow = WORKFLOW
                .iter()
                .find(|g| g.id == group)
                .unwrap_or_else(|| panic!("no workflow group {group}"));
            writeln!(sheet, "# {}\n", workflow.label).unwrap();
            for section in workflow.sections {
                for &path in section.paths {
                    let engine = catalogue
                        .entry(path)
                        .expect("a workflow path is an input")
                        .meta
                        .help;
                    writeln!(sheet, "## {} ({path})", label_for(path)).unwrap();
                    writeln!(sheet, "help: {}", help_for(path)).unwrap();
                    if HELP.iter().any(|&(key, _)| key == path) {
                        let replaced = if engine.is_empty() {
                            "(no help)"
                        } else {
                            engine
                        };
                        writeln!(sheet, "replaces: {replaced}").unwrap();
                    }
                    writeln!(sheet).unwrap();
                }
            }
        }
        writeln!(sheet, "# Hint lines\n").unwrap();
        for &(key, hint) in HINTS {
            writeln!(sheet, "{key}: {hint}").unwrap();
        }
        sheet
    }

    #[test]
    fn the_review_sheet_lists_every_input_of_its_groups_and_every_hint() {
        let sheet = review_sheet(&BATCH_A);
        for path in batch_a_paths() {
            let entry = format!("({path})\nhelp: {}\n", help_for(path));
            assert!(sheet.contains(&entry), "{path}");
        }
        for &(key, hint) in HINTS {
            assert!(sheet.contains(&format!("{key}: {hint}\n")), "{key}");
        }
    }

    /// Writes the review sheet for the G-3 physics review (spec section 4): the groups in
    /// `HELP_REVIEW_GROUPS` (comma-separated workflow group ids; batch A's by default) to
    /// `HELP_REVIEW_OUT` (`target/help-for-review.txt` by default).
    #[test]
    #[ignore = "writes the G-3 review sheet (HELP_REVIEW_GROUPS, HELP_REVIEW_OUT)"]
    fn help_for_review() {
        let groups: Vec<String> = match std::env::var("HELP_REVIEW_GROUPS") {
            Ok(list) => list.split(',').map(|id| id.trim().to_owned()).collect(),
            Err(_) => BATCH_A.iter().map(|id| (*id).to_owned()).collect(),
        };
        let groups: Vec<&str> = groups.iter().map(String::as_str).collect();
        let out = std::env::var("HELP_REVIEW_OUT")
            .unwrap_or_else(|_| "target/help-for-review.txt".to_owned());
        let sheet = review_sheet(&groups);
        std::fs::write(&out, &sheet).expect("the review sheet is written");
        println!("{sheet}");
    }
}

/// Help batch B (T8, spec 3.5): the empty helps of the Calibration and model, Materials,
/// Housing and retainers, and Thermal and demagnetization groups, and the overrides batch A
/// leaves. Every direction a batch-B help states is checked against the engine here.
#[cfg(test)]
mod batch_b_tests {
    use super::*;
    use crate::engine::meta::{InputSet, ResultSet, Value};
    use crate::engine::size_controls::{dimensions, resize_diameter, with_ring_radius};
    use crate::gui::inputs::WORKFLOW;
    use crate::{DesignInputs, compute_all};
    use std::collections::BTreeSet;

    /// Batch B's share of the 87 empty engine helps (map-help section 5), by workflow group:
    /// Calibration and model, Materials, Housing and retainers, Thermal and demagnetization.
    const BATCH_B_EMPTY: [&str; 52] = [
        "coupling.mu0",
        "calibration.mu0",
        "calibration.apothem_mm",
        "calibration.magnet_length_mm",
        "calibration.magnet_width_mm",
        "calibration.magnet_thickness_mm",
        "calibration.br_T",
        "calibration.c_end",
        "calibration.fea_torque2_Nm",
        "materials.steel.mu_r_incremental",
        "materials.steel.specific_heat_J_kgK",
        "materials.steel.cte_per_C",
        "materials.steel.modulus_GPa",
        "temperature.mismatch.ndfeb_modulus_GPa",
        "temperature.slip_loss.sigma_ndfeb_S_m",
        "temperature.thermal.c_ndfeb",
        "metal.bond_inner_mm",
        "metal.bond_outer_mm",
        "temperature.mismatch.recommended_bondline_mm",
        "temperature.adhesive_life.fatigue_endurance",
        "temperature.slip_loss.sigma_316_S_m",
        "temperature.thermal.c_316",
        "temperature.thermal.c_aluminium",
        "metal.steel_density_g_mm3",
        "metal.sleeve_density_g_mm3",
        "metal.al_density_g_mm3",
        "metal.hardware_g",
        "materials.screws.proof_10_9_MPa",
        "materials.screws.yield_A4_70_MPa",
        "metal.hub_length_mm",
        "metal.cup_depth_mm",
        "metal.web_mm",
        "metal.boss_length_mm",
        "metal.boss_od_mm",
        "metal.cap_axial_mm",
        "metal.cap_od_mm",
        "metal.cap_thread_engagement_mm",
        "metal.front_endplate_mm",
        "metal.rear_endplate_mm",
        "metal.retainer_span_mm",
        "metal.rear_endplate_hole_mm",
        "metal.adapter_flange_dia_mm",
        "metal.adapter_flange_mm",
        "metal.adapter_pilot_dia_mm",
        "metal.adapter_pilot_mm",
        "metal.adapter_boss_mm",
        "metal.adapter_hardware_g",
        "clamps.joint_bolt_circle_mm",
        "clamps.joint_friction",
        "temperature.demag.knee_fraction",
        "temperature.demag.design_margin_C",
        "temperature.slip_loss.end_factor",
    ];

    /// The non-empty engine helps batch B replaces (spec 3.5): the grade-mode helps, the stored
    /// 3D fields, the "User supplied", "Placeholder" and "UNCONFIRMED" family, and helps with
    /// jargon or a stale or wrong fact.
    const BATCH_B_OVERRIDES: [&str; 31] = [
        "calibration.alpha_br_per_C",
        "mass.magnets_g",
        "model.inner_temp_check",
        "model.outer_temp_check",
        "temperature.slip_loss.b_hub_T",
        "temperature.slip_loss.b_cup_T",
        "temperature.slip_loss.b_sleeve_T",
        "temperature.slip_loss.b_liner_T",
        "temperature.slip_loss.cap_integral_T2m4",
        "temperature.slip_loss.web_integral_T2m2",
        "temperature.slip_loss.b_magnet_T",
        "temperature.demag.h_rev_aligned_kA_m",
        "temperature.demag.h_rev_pullout_kA_m",
        "temperature.demag.h_rev_likepole_kA_m",
        "temperature.demag.h_rev_single_ring_kA_m",
        "temperature.duty.driving_rise_C",
        "temperature.duty.life_hours",
        "temperature.adhesive_life.hot_strength_retained",
        "calibration.test_temp_C",
        "metal.slip_event_s",
        "metal.measured_drag_Nm",
        "materials.steel.density_g_cm3",
        "temperature.demag.hcj20_kA_m",
        "temperature.demag.beta_hcj_per_C",
        "temperature.mismatch.adhesive_shear_modulus_GPa",
        "metal.cup_wall_corner_mm",
        "materials.steel.bsat_T",
        "materials.steel.conductivity_S_m",
        "temperature.mismatch.ndfeb_cte_per_C",
        "temperature.slip_loss.high_multiplier",
        "clamps.joint_screws",
    ];

    /// The stored 3D fields the overlay words: the dashboard's stored 3D inputs but the Rust-only
    /// free-space ones, which the engine words. Each help ends with the caveat that they do not
    /// follow the design.
    fn stored_3d() -> Vec<&'static str> {
        crate::gui::dashboard::STORED_3D_INPUTS
            .into_iter()
            .filter(|path| !InputCatalogue::get().entry(path).unwrap().meta.rust_only)
            .collect()
    }

    /// The bench-prototype inputs: each help ends with what the calibration coefficient is for.
    const PROTOTYPE: [&str; 8] = [
        "calibration.mu0",
        "calibration.apothem_mm",
        "calibration.magnet_length_mm",
        "calibration.magnet_width_mm",
        "calibration.magnet_thickness_mm",
        "calibration.br_T",
        "calibration.c_end",
        "calibration.test_temp_C",
    ];

    /// Defaults nobody has measured: each help ends with the same warning.
    const GUESSES: [&str; 3] = [
        "temperature.duty.driving_rise_C",
        "temperature.duty.life_hours",
        "temperature.adhesive_life.hot_strength_retained",
    ];

    /// The optional adapter's geometry: each help ends with what it cannot change.
    const ADAPTER: [&str; 6] = [
        "metal.adapter_flange_dia_mm",
        "metal.adapter_flange_mm",
        "metal.adapter_pilot_dia_mm",
        "metal.adapter_pilot_mm",
        "metal.adapter_boss_mm",
        "metal.adapter_hardware_g",
    ];

    /// Which way a help says a result moves when its input rises one slider step.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Moves {
        Up,
        Down,
        Stays,
    }
    use Moves::{Down, Stays, Up};

    /// The design a claim is checked on: the example design, or it with one setting changed.
    fn claim_design(name: &str) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        let mut set = |path: &str, value: Value| {
            inputs
                .set(path, value)
                .unwrap_or_else(|e| panic!("{name}: {path}: {e:?}"));
        };
        match name {
            "example" => {}
            "no back iron" => set("coupling.backiron", Value::Int(0)),
            "1018 back iron" => set("materials.parts.back_iron", Value::Int(2)),
            "304 back iron" => set("materials.parts.back_iron", Value::Int(7)),
            "titanium sleeve" => set("materials.parts.sleeve_liner", Value::Int(2)),
            "7075 cap" => set("materials.parts.cap_housing", Value::Int(2)),
            "class 10.9" => set("clamps.screw_class", Value::Int(2)),
            "class A4-70" => set("clamps.screw_class", Value::Int(3)),
            "coercivity inputs" => set("temperature.demag.coercivity_source", Value::Int(0)),
            "measured drag" => set("metal.measured_drag_Nm", Value::Num(0.05)),
            "manual, no grade" => {
                set("coupling.magnets.part_inner", Value::Text(String::new()));
                set("coupling.magnets.part_outer", Value::Text(String::new()));
            }
            "Y30 grades" => {
                set("coupling.magnets.part_inner", Value::Text(String::new()));
                set("coupling.magnets.part_outer", Value::Text(String::new()));
                set(
                    "coupling.magnets.grade_inner",
                    Value::Text("Y30".to_owned()),
                );
                set(
                    "coupling.magnets.grade_outer",
                    Value::Text("Y30".to_owned()),
                );
            }
            other => panic!("no claim design {other:?}"),
        }
        inputs
    }

    /// The numeric result at `path` on `inputs`; any other value fails the test.
    fn number(inputs: &DesignInputs, path: &str) -> f64 {
        match compute_all(inputs).get(path) {
            Some(Value::Num(x)) => x,
            other => panic!("{path}: {other:?}"),
        }
    }

    /// Every direction a batch-B help states, as (input, design, result, how the result moves
    /// when the input rises one slider step on that design). `Stays` pins "only", "not",
    /// "unused" and "replaces it" claims bit for bit.
    #[rustfmt::skip]
    const CLAIMS: &[(&str, &str, &str, Moves)] = &[
        // Calibration and model
        ("coupling.mu0", "example", "model.pullout_Nm", Down),
        ("calibration.mu0", "example", "model.pullout_Nm", Stays),
        ("calibration.mu0", "no back iron", "calibration.f_cal_updated", Up),
        ("calibration.apothem_mm", "example", "model.pullout_Nm", Stays),
        ("calibration.apothem_mm", "no back iron", "model.pullout_Nm", Down),
        ("calibration.magnet_length_mm", "example", "model.pullout_Nm", Stays),
        ("calibration.magnet_length_mm", "no back iron", "calibration.f_cal_updated", Down),
        ("calibration.magnet_width_mm", "no back iron", "calibration.f_cal_updated", Down),
        ("calibration.magnet_width_mm", "example", "model.pullout_Nm", Stays),
        ("calibration.magnet_thickness_mm", "no back iron", "calibration.f_cal_updated", Down),
        ("calibration.magnet_thickness_mm", "example", "model.pullout_Nm", Stays),
        ("calibration.br_T", "no back iron", "calibration.f_cal_updated", Down),
        ("calibration.br_T", "example", "model.pullout_Nm", Stays),
        ("calibration.c_end", "example", "model.pullout_Nm", Stays),
        ("calibration.c_end", "no back iron", "model.pullout_Nm", Up),
        ("calibration.fea_torque2_Nm", "example", "calibration.fea_interp_Nm", Up),
        ("calibration.fea_torque2_Nm", "no back iron", "model.pullout_Nm", Stays),
        // Materials
        ("materials.steel.mu_r_incremental", "example", "temperature.slip_loss.total_W", Down),
        ("materials.steel.mu_r_incremental", "1018 back iron", "temperature.slip_loss.total_W", Down),
        ("materials.steel.mu_r_incremental", "no back iron", "temperature.slip_loss.total_W", Stays),
        ("materials.steel.specific_heat_J_kgK", "example", "temperature.thermal.heat_capacity_J_K", Up),
        ("materials.steel.specific_heat_J_kgK", "example", "temperature.thermal.steady_high_C", Stays),
        ("materials.steel.specific_heat_J_kgK", "1018 back iron", "temperature.thermal.heat_capacity_J_K", Stays),
        ("materials.steel.specific_heat_J_kgK", "304 back iron", "temperature.thermal.heat_capacity_J_K", Up),
        ("materials.steel.cte_per_C", "example", "temperature.mismatch.peak_shear_current_MPa", Up),
        ("materials.steel.cte_per_C", "1018 back iron", "temperature.mismatch.peak_shear_current_MPa", Stays),
        ("materials.steel.cte_per_C", "no back iron", "temperature.mismatch.peak_shear_current_MPa", Stays),
        ("materials.steel.modulus_GPa", "example", "temperature.mismatch.peak_shear_current_MPa", Up),
        ("materials.steel.modulus_GPa", "1018 back iron", "temperature.mismatch.peak_shear_current_MPa", Stays),
        ("temperature.mismatch.ndfeb_modulus_GPa", "example", "temperature.mismatch.peak_shear_current_MPa", Up),
        ("temperature.mismatch.ndfeb_modulus_GPa", "Y30 grades", "temperature.mismatch.peak_shear_current_MPa", Up),
        ("temperature.slip_loss.sigma_ndfeb_S_m", "example", "temperature.slip_loss.magnets_W", Up),
        ("temperature.slip_loss.sigma_ndfeb_S_m", "Y30 grades", "temperature.slip_loss.magnets_W", Up),
        ("temperature.thermal.c_ndfeb", "Y30 grades", "temperature.thermal.heat_capacity_J_K", Up),
        ("temperature.thermal.c_ndfeb", "example", "temperature.thermal.steady_high_C", Stays),
        ("temperature.thermal.c_ndfeb", "example", "temperature.thermal.heat_capacity_J_K", Up),
        ("metal.bond_inner_mm", "example", "model.hub_wall_mm", Down),
        ("metal.bond_inner_mm", "example", "temperature.mismatch.peak_shear_current_MPa", Down),
        ("metal.bond_inner_mm", "example", "model.pullout_Nm", Stays),
        ("metal.bond_outer_mm", "example", "model.cup_od_mm", Up),
        ("metal.bond_outer_mm", "example", "mass.cup_g", Up),
        ("metal.bond_outer_mm", "example", "model.pullout_Nm", Stays),
        ("temperature.mismatch.recommended_bondline_mm", "example", "temperature.adhesive_life.daily_peak_shear_MPa", Down),
        ("temperature.mismatch.recommended_bondline_mm", "example", "temperature.mismatch.peak_shear_current_MPa", Stays),
        ("temperature.adhesive_life.fatigue_endurance", "example", "temperature.adhesive_life.hot_fatigue_margin", Up),
        ("temperature.slip_loss.sigma_316_S_m", "example", "temperature.slip_loss.sleeve_W", Up),
        ("temperature.slip_loss.sigma_316_S_m", "titanium sleeve", "temperature.slip_loss.total_W", Stays),
        ("temperature.slip_loss.sigma_316_S_m", "example", "temperature.slip_loss.liner_W", Up),
        ("temperature.thermal.c_316", "example", "temperature.thermal.heat_capacity_J_K", Up),
        ("temperature.thermal.c_316", "titanium sleeve", "temperature.thermal.heat_capacity_J_K", Stays),
        ("temperature.thermal.c_aluminium", "example", "temperature.thermal.heat_capacity_J_K", Up),
        ("temperature.thermal.c_aluminium", "7075 cap", "temperature.thermal.heat_capacity_J_K", Stays),
        ("temperature.thermal.c_aluminium", "no back iron", "temperature.thermal.heat_capacity_J_K", Up),
        ("metal.steel_density_g_mm3", "example", "mass.cup_g", Up),
        ("metal.steel_density_g_mm3", "1018 back iron", "mass.total_g", Stays),
        ("metal.steel_density_g_mm3", "no back iron", "mass.total_g", Stays),
        ("metal.sleeve_density_g_mm3", "example", "retainers.endplates_g", Up),
        ("metal.sleeve_density_g_mm3", "titanium sleeve", "mass.total_g", Stays),
        ("metal.al_density_g_mm3", "example", "retainers.cap_g", Up),
        ("metal.al_density_g_mm3", "7075 cap", "retainers.cap_g", Stays),
        ("metal.al_density_g_mm3", "7075 cap", "metal.adapter_g", Up),
        ("metal.al_density_g_mm3", "no back iron", "mass.hub_g", Up),
        ("metal.al_density_g_mm3", "example", "metal.adapter_g", Up),
        ("metal.hardware_g", "example", "mass.total_g", Up),
        ("metal.hardware_g", "example", "temperature.thermal.heat_capacity_J_K", Up),
        ("materials.screws.proof_10_9_MPa", "example", "clamps.joint_preload_N", Stays),
        ("materials.screws.proof_10_9_MPa", "class 10.9", "clamps.joint_preload_N", Up),
        ("materials.screws.yield_A4_70_MPa", "example", "clamps.joint_preload_N", Stays),
        ("materials.screws.yield_A4_70_MPa", "class A4-70", "clamps.joint_preload_N", Up),
        // Housing and retainers
        ("metal.hub_length_mm", "example", "mass.hub_g", Up),
        ("metal.hub_length_mm", "example", "metal.axial_stack_mm", Stays),
        ("metal.cup_depth_mm", "example", "metal.axial_stack_mm", Up),
        ("metal.cup_depth_mm", "example", "metal.large_dia_stack_mm", Up),
        ("metal.cup_depth_mm", "example", "mass.cup_g", Up),
        ("metal.cup_depth_mm", "example", "metal.hybrid_length_mm", Up),
        ("metal.web_mm", "example", "metal.axial_stack_mm", Up),
        ("metal.web_mm", "example", "metal.large_dia_stack_mm", Up),
        ("metal.web_mm", "example", "temperature.slip_loss.web_W", Stays),
        ("metal.web_mm", "no back iron", "temperature.slip_loss.web_W", Up),
        ("metal.web_mm", "example", "mass.cup_g", Up),
        ("metal.boss_length_mm", "example", "metal.axial_stack_mm", Up),
        ("metal.boss_length_mm", "example", "metal.large_dia_stack_mm", Stays),
        ("metal.boss_length_mm", "example", "mass.boss_g", Up),
        ("metal.boss_od_mm", "example", "mass.boss_g", Up),
        ("metal.boss_od_mm", "example", "metal.adapter_g", Up),
        ("metal.cap_axial_mm", "example", "metal.axial_stack_mm", Up),
        ("metal.cap_axial_mm", "example", "metal.large_dia_stack_mm", Up),
        ("metal.cap_axial_mm", "example", "temperature.slip_loss.cap_W", Up),
        ("metal.cap_axial_mm", "example", "retainers.cap_g", Up),
        ("metal.cap_od_mm", "example", "metal.rotating_od_mm", Up),
        ("metal.cap_od_mm", "example", "retainers.cap_g", Up),
        ("metal.cap_thread_engagement_mm", "example", "retainers.cap_g", Up),
        ("metal.cap_thread_engagement_mm", "example", "metal.axial_stack_mm", Stays),
        ("metal.front_endplate_mm", "example", "retainers.endplates_g", Up),
        ("metal.front_endplate_mm", "example", "metal.axial_stack_mm", Stays),
        ("metal.rear_endplate_mm", "example", "retainers.endplates_g", Up),
        ("metal.rear_endplate_mm", "example", "metal.axial_stack_mm", Stays),
        ("metal.retainer_span_mm", "example", "retainers.retainers_g", Up),
        ("metal.retainer_span_mm", "example", "temperature.slip_loss.sleeve_W", Stays),
        ("metal.rear_endplate_hole_mm", "example", "retainers.endplates_g", Down),
        ("metal.adapter_flange_dia_mm", "example", "metal.adapter_variant_mass_g", Up),
        ("metal.adapter_flange_dia_mm", "example", "mass.total_g", Stays),
        ("metal.adapter_flange_mm", "example", "metal.hybrid_length_mm", Up),
        ("metal.adapter_flange_mm", "example", "mass.total_g", Stays),
        ("metal.adapter_pilot_dia_mm", "example", "metal.adapter_steel_removed_g", Up),
        ("metal.adapter_pilot_dia_mm", "example", "mass.total_g", Stays),
        ("metal.adapter_pilot_mm", "example", "metal.adapter_g", Up),
        ("metal.adapter_pilot_mm", "example", "mass.total_g", Stays),
        ("metal.adapter_boss_mm", "example", "metal.hybrid_length_mm", Up),
        ("metal.adapter_boss_mm", "example", "mass.total_g", Stays),
        ("metal.adapter_hardware_g", "example", "metal.adapter_variant_mass_g", Up),
        ("metal.adapter_hardware_g", "example", "mass.total_g", Stays),
        ("clamps.joint_bolt_circle_mm", "example", "clamps.joint_torque_Nm", Up),
        ("clamps.joint_bolt_circle_mm", "example", "clamps.joint_sf", Up),
        ("clamps.joint_friction", "example", "clamps.joint_torque_Nm", Up),
        // Thermal and demagnetization
        ("temperature.demag.knee_fraction", "example", "temperature.summary.magnet_limit_C", Up),
        ("temperature.demag.knee_fraction", "Y30 grades", "temperature.demag.cold_limit_C", Down),
        ("temperature.demag.design_margin_C", "example", "temperature.summary.magnet_limit_C", Down),
        ("temperature.demag.design_margin_C", "Y30 grades", "temperature.demag.cold_limit_C", Up),
        ("temperature.slip_loss.end_factor", "example", "temperature.slip_loss.cap_W", Up),
        ("temperature.slip_loss.end_factor", "example", "temperature.slip_loss.cup_W", Stays),
        ("temperature.slip_loss.end_factor", "no back iron", "temperature.slip_loss.cup_W", Up),
        ("temperature.slip_loss.end_factor", "example", "temperature.slip_loss.sleeve_W", Up),
        // Overrides: grade mode
        ("calibration.alpha_br_per_C", "example", "metal.torque_hot_low_Nm", Up),
        ("calibration.alpha_br_per_C", "example", "metal.torque_cold_Nm", Down),
        ("calibration.alpha_br_per_C", "Y30 grades", "metal.torque_hot_low_Nm", Stays),
        // Overrides: stored 3D fields
        ("temperature.slip_loss.b_hub_T", "example", "temperature.slip_loss.hub_W", Up),
        ("temperature.slip_loss.b_hub_T", "no back iron", "temperature.slip_loss.hub_W", Stays),
        ("temperature.slip_loss.b_cup_T", "example", "temperature.slip_loss.cup_W", Up),
        ("temperature.slip_loss.b_cup_T", "no back iron", "temperature.slip_loss.cup_W", Stays),
        ("temperature.slip_loss.b_sleeve_T", "example", "temperature.slip_loss.sleeve_W", Up),
        ("temperature.slip_loss.b_liner_T", "example", "temperature.slip_loss.liner_W", Up),
        ("temperature.slip_loss.cap_integral_T2m4", "example", "temperature.slip_loss.cap_W", Up),
        ("temperature.slip_loss.web_integral_T2m2", "example", "temperature.slip_loss.web_W", Up),
        ("temperature.slip_loss.web_integral_T2m2", "no back iron", "temperature.slip_loss.web_W", Stays),
        ("temperature.slip_loss.b_magnet_T", "example", "temperature.slip_loss.magnets_W", Up),
        ("temperature.demag.h_rev_aligned_kA_m", "example", "temperature.demag.onset_aligned_C", Down),
        ("temperature.demag.h_rev_aligned_kA_m", "example", "temperature.summary.magnet_limit_C", Stays),
        ("temperature.demag.h_rev_pullout_kA_m", "example", "temperature.demag.onset_pullout_C", Down),
        ("temperature.demag.h_rev_pullout_kA_m", "example", "temperature.summary.magnet_limit_C", Stays),
        ("temperature.demag.h_rev_likepole_kA_m", "example", "temperature.summary.magnet_limit_C", Down),
        ("temperature.demag.h_rev_likepole_kA_m", "Y30 grades", "temperature.demag.cold_limit_C", Up),
        ("temperature.demag.h_rev_single_ring_kA_m", "example", "temperature.summary.cure_margin_C", Down),
        ("temperature.demag.h_rev_single_ring_kA_m", "example", "temperature.summary.magnet_limit_C", Stays),
        // Overrides: placeholders and assumed values
        ("temperature.duty.driving_rise_C", "example", "temperature.thermal.start_C", Up),
        ("temperature.duty.driving_rise_C", "example", "temperature.summary.hot_day_start_C", Up),
        ("temperature.duty.life_hours", "example", "temperature.slip_life.slip_duty", Down),
        ("temperature.duty.life_hours", "example", "temperature.magnet_life.peak_C", Down),
        ("temperature.adhesive_life.hot_strength_retained", "example", "temperature.adhesive_life.hot_fatigue_margin", Up),
        ("calibration.test_temp_C", "no back iron", "calibration.f_cal_updated", Up),
        ("calibration.test_temp_C", "example", "model.pullout_Nm", Stays),
        ("metal.slip_event_s", "example", "temperature.slip_life.slip_duty", Up),
        ("metal.slip_event_s", "example", "metal.magnetic_cycles", Up),
        // Overrides: bench drag, jargon and stale facts
        ("materials.steel.density_g_cm3", "example", "mass.total_g", Stays),
        ("temperature.demag.hcj20_kA_m", "example", "temperature.summary.magnet_limit_C", Stays),
        ("temperature.demag.hcj20_kA_m", "coercivity inputs", "temperature.summary.magnet_limit_C", Up),
        ("temperature.demag.hcj20_kA_m", "manual, no grade", "temperature.summary.magnet_limit_C", Up),
        ("temperature.demag.beta_hcj_per_C", "example", "temperature.summary.magnet_limit_C", Stays),
        ("temperature.demag.beta_hcj_per_C", "coercivity inputs", "temperature.summary.magnet_limit_C", Down),
        ("temperature.demag.beta_hcj_per_C", "manual, no grade", "temperature.summary.magnet_limit_C", Up),
        ("temperature.mismatch.adhesive_shear_modulus_GPa", "example", "temperature.mismatch.peak_shear_current_MPa", Up),
        ("metal.cup_wall_corner_mm", "example", "model.cup_od_mm", Up),
        ("materials.steel.bsat_T", "example", "materials.backiron_thickness_needed_mm", Down),
        ("materials.steel.bsat_T", "1018 back iron", "materials.backiron_thickness_needed_mm", Stays),
        ("materials.steel.conductivity_S_m", "example", "temperature.slip_loss.total_W", Up),
        ("materials.steel.conductivity_S_m", "1018 back iron", "temperature.slip_loss.total_W", Stays),
        ("temperature.mismatch.ndfeb_cte_per_C", "example", "temperature.mismatch.peak_shear_current_MPa", Down),
        ("temperature.mismatch.ndfeb_cte_per_C", "Y30 grades", "temperature.mismatch.peak_shear_current_MPa", Down),
        ("temperature.slip_loss.high_multiplier", "example", "temperature.thermal.steady_high_C", Up),
        ("temperature.slip_loss.high_multiplier", "measured drag", "temperature.thermal.steady_high_C", Stays),
        ("temperature.slip_loss.high_multiplier", "example", "temperature.slip_loss.total_W", Stays),
        ("clamps.joint_screws", "example", "clamps.joint_torque_Nm", Up),
    ];

    #[test]
    fn batch_b_fills_every_empty_help_of_its_four_groups() {
        // The split with batch A (T7) is by workflow group: these four groups are batch B's.
        let catalogue = InputCatalogue::get();
        let empty_in_batch_b: BTreeSet<&str> = WORKFLOW
            .iter()
            .filter(|group| ["calibration", "materials", "housing", "thermal"].contains(&group.id))
            .flat_map(|group| group.sections.iter())
            .flat_map(|section| section.paths.iter().copied())
            .filter(|&path| {
                catalogue
                    .entry(path)
                    .is_some_and(|e| e.meta.help.is_empty())
            })
            .collect();
        assert_eq!(empty_in_batch_b, BTreeSet::from(BATCH_B_EMPTY));
        for path in BATCH_B_EMPTY {
            assert!(row(HELP, path).is_some(), "{path}: no help row");
            assert_eq!(original_help(path), None, "{path}: nothing to keep");
        }
    }

    #[test]
    fn batch_b_overrides_replace_a_non_empty_engine_help() {
        for path in BATCH_B_OVERRIDES {
            let (_, engine) = metadata(path).unwrap_or_else(|| panic!("{path}: no such path"));
            assert!(!engine.is_empty(), "{path}: the engine help is empty");
            assert!(row(HELP, path).is_some(), "{path}: no help row");
            // A Rust-only input's help is rewritten in the engine (T9), never overlaid.
            if let Some(entry) = InputCatalogue::get().entry(path) {
                assert!(!entry.meta.rust_only, "{path} is Rust-only");
            }
            assert!(!BATCH_B_EMPTY.contains(&path), "{path} is in both lists");
        }
    }

    #[test]
    fn the_original_note_is_the_spreadsheet_s_own_text() {
        // Under the toggle, "Original spreadsheet note" shows what the spreadsheet said: the
        // engine help, except where a correction reworded it, when the registry keeps the
        // workbook's text (and an empty one shows no note).
        let reworded: Vec<&str> = BATCH_B_OVERRIDES
            .into_iter()
            .filter(|&path| spreadsheet_help(path).is_some())
            .collect();
        assert_eq!(
            reworded,
            [
                "temperature.slip_loss.web_integral_T2m2",
                "temperature.demag.hcj20_kA_m",
                "temperature.demag.beta_hcj_per_C",
                "temperature.mismatch.adhesive_shear_modulus_GPa",
            ]
        );
        for path in BATCH_B_OVERRIDES {
            let engine = metadata(path).map(|(_, help)| help);
            let spreadsheet = spreadsheet_help(path).or(engine).filter(|t| !t.is_empty());
            assert_eq!(original_help(path), spreadsheet, "{path}");
            if reworded.contains(&path) {
                assert_ne!(original_help(path), engine, "{path}: the corrected text");
            }
        }
    }

    #[test]
    fn each_cluster_ends_with_one_shared_caveat() {
        // A cluster is exactly the rows whose last sentence is its caveat. The stored-3D caveat
        // is the engine's sentence (amendment A-7), typed once.
        let stored_3d = stored_3d();
        for (cluster, caveat) in [
            (&stored_3d[..], stored_3d_caveat!()),
            (&PROTOTYPE[..], prototype_caveat!()),
            (&GUESSES[..], guess_caveat!()),
            (&ADAPTER[..], adapter_caveat!()),
        ] {
            let ending = format!(". {caveat}");
            let ending_with_it: BTreeSet<&str> = HELP
                .iter()
                .filter(|(_, text)| text.ends_with(&ending))
                .map(|&(path, _)| path)
                .collect();
            assert_eq!(
                ending_with_it,
                BTreeSet::from_iter(cluster.iter().copied()),
                "{caveat}"
            );
        }
    }

    #[test]
    fn no_help_row_repeats_its_input_s_default() {
        // Spec 3.5: no default numbers in help; the tooltip already prints "Default: ...".
        for &(path, text) in HELP {
            let Some(entry) = InputCatalogue::get().entry(path) else {
                continue;
            };
            let default = match entry.default {
                Value::Num(x) => x,
                Value::Int(n) => n as f64,
                _ => continue,
            };
            let repeats = text
                .split(|c: char| c.is_whitespace() || ",;:()".contains(c))
                .filter_map(|token| token.trim_end_matches('.').parse::<f64>().ok())
                .any(|number| number == default);
            assert!(!repeats, "{path}: {text:?} repeats the default");
        }
    }

    #[test]
    fn every_direction_a_batch_b_help_states_holds_one_slider_step_up() {
        let catalogue = InputCatalogue::get();
        for &(input, design, result, moves) in CLAIMS {
            assert!(
                BATCH_B_EMPTY.contains(&input) || BATCH_B_OVERRIDES.contains(&input),
                "{input} is not batch B's"
            );
            let before = claim_design(design);
            let step = catalogue
                .entry(input)
                .and_then(|entry| entry.meta.range)
                .unwrap_or_else(|| panic!("{input}: no slider"))
                .step;
            let raised = match before.get(input) {
                Some(Value::Num(x)) => Value::Num(x + step),
                Some(Value::Int(n)) => Value::Int(n + step as i64),
                other => panic!("{input}: {other:?}"),
            };
            let mut after = before.clone();
            after
                .set(input, raised)
                .unwrap_or_else(|e| panic!("{input}: {e:?}"));
            let (was, now) = (number(&before, result), number(&after, result));
            let seen = if now > was {
                Up
            } else if now < was {
                Down
            } else {
                Stays
            };
            assert_eq!(
                seen, moves,
                "{input} one step up on the {design} design: {result} {was} -> {now}"
            );
        }
        // Every batch-B input with a slider and a value has its help's claim checked here; the
        // blank measured drag and the three results are checked in the next test.
        for path in BATCH_B_EMPTY.iter().chain(&BATCH_B_OVERRIDES) {
            let Some(entry) = catalogue.entry(path) else {
                continue;
            };
            if entry.default != Value::None {
                assert!(
                    CLAIMS.iter().any(|claim| claim.0 == *path),
                    "{path}: no claim"
                );
            }
        }
    }

    /// Ruling 11 (3A): the texts that named the clamp as a cost now name what rises in its
    /// place, the shaft-joint and gearbox loads: the cold-high torque, which whatever holds the
    /// hub on the shaft and the gearbox output both carry, and its value at the gearbox input.
    /// Each claim holds one slider step up on a design where the input is free. The bore help's
    /// two limits hold at the bore's slider top: the hub check fails on the 8-magnet probe
    /// design, keyed or keyless, and the wall past the keyway breaks through on the smallest ring.
    #[test]
    fn the_joint_neutral_texts_claim_what_the_engine_does() {
        use crate::engine::explain::notes::Step;
        use crate::engine::explain::probes::{probe_designs, step_input};
        use crate::engine::size_controls::dimensions;
        use crate::gui::dashboard::{Level, check_level};
        let raised = |design: &DesignInputs, input: &str| {
            step_input(design, input, Step::Raise).unwrap_or_else(|e| panic!("{input}: {e:?}"))
        };
        let torque_and_loads = [
            "model.pullout_Nm",
            "metal.torque_cold_high_Nm",
            "metal.cold_high_input_Nm",
        ];
        let rise = |before: &DesignInputs, after: &DesignInputs, input: &str| {
            for result in torque_and_loads {
                assert!(
                    number(after, result) > number(before, result),
                    "{input} up: {result}"
                );
            }
        };
        // Stronger manual blocks (no library part, no grade picked): both helps.
        let mut manual = DesignInputs::default();
        manual.coupling.magnets.part_inner.clear();
        manual.coupling.magnets.part_outer.clear();
        for input in [
            "coupling.magnets.manual_inner_br_T",
            "coupling.magnets.manual_outer_br_T",
        ] {
            rise(&manual, &raised(&manual, input), input);
        }
        // Longer magnets (the Key design row's hint and the Size tab's overall length): the
        // torque, both loads and the overall length.
        let example = DesignInputs::default();
        let length = "coupling.magnets.axial_length_mm";
        let longer = raised(&example, length);
        rise(&example, &longer, length);
        let overall = |inputs: &DesignInputs| dimensions(inputs).expect("a size").overall_length_mm;
        assert!(overall(&longer) > overall(&example));
        // A larger bore thins the hub wall past the keyway; at its slider top it breaks through
        // the smallest ring's hub, the keyway at its default.
        let wall = "model.hub_wall_past_key_mm";
        let wider = raised(&example, "coupling.bore_mm");
        assert!(number(&wider, wall) < number(&example, wall));
        let range = |path: &str| {
            InputCatalogue::get()
                .entry(path)
                .and_then(|entry| entry.meta.range)
                .expect("a slider")
        };
        let mut smallest = example.clone();
        smallest.coupling.inner_back_apothem_mm = range("coupling.inner_back_apothem_mm").min;
        assert!(
            number(&smallest, wall) > 0.0,
            "the smallest ring holds the example bore"
        );
        smallest.coupling.bore_mm = range("coupling.bore_mm").max;
        assert!(
            number(&smallest, wall) <= 0.0,
            "the bore alone breaks through"
        );
        // A larger bore also thins the hub wall under the magnets, which the hub check reads: the
        // help names the check by the label the results table draws it under. At the 8-magnet
        // probe design the bore's slider top fails it, with the design's keyway and keyless
        // alike, while the wall past the keyway still stands: the check fails first there.
        let hub_check = "model.hub_check";
        let bore_help = help_for("coupling.bore_mm").to_lowercase();
        assert!(
            bore_help.contains(&label_for(hub_check).to_lowercase()),
            "the bore help names the {hub_check} row: {bore_help}"
        );
        let hub_wall = "model.hub_wall_mm";
        let (_, eight) = probe_designs()
            .into_iter()
            .find(|(name, _)| *name == "8 magnets per ring")
            .expect("the 8-magnet probe design");
        let keyless = range("coupling.keyway_depth_mm").min;
        assert_ne!(eight.coupling.keyway_depth_mm, keyless, "two keyways");
        let level = |inputs: &DesignInputs| check_level(&compute_all(inputs), hub_check);
        for keyway in [eight.coupling.keyway_depth_mm, keyless] {
            let mut own_bore = eight.clone();
            own_bore.coupling.keyway_depth_mm = keyway;
            let mut widest = own_bore.clone();
            widest.coupling.bore_mm = range("coupling.bore_mm").max;
            assert_eq!(
                level(&own_bore),
                Some(Level::Good),
                "keyway {keyway} mm, the design's bore"
            );
            assert_eq!(
                level(&widest),
                Some(Level::Bad),
                "keyway {keyway} mm, the widest bore"
            );
            assert!(
                number(&widest, hub_wall) < number(&own_bore, hub_wall),
                "keyway {keyway} mm: {hub_wall}"
            );
            assert!(
                number(&widest, wall) > 0.0,
                "keyway {keyway} mm: {wall} at the widest bore"
            );
        }
    }

    /// BL-059: the gearbox rating's help names the check that now reads the rating, and the
    /// ripple row's label and help name the nominal slip swing, apart from the cold-high torque
    /// the check compares; the gear ratio's help calls it the slip swing too (one name on
    /// screen); the check row's help names both sides of the gearbox and that fatigue is not
    /// checked (G-3 review M-2, M-3). The rating's type is open (owner decision 2A), so none of
    /// these texts calls it a peak (the spreadsheet's own note, shown only with Show spreadsheet
    /// details on, keeps its "peak": plan ruling R-11); the operating-temperature slider reaches
    /// below the minimum, so no text calls the swing warm. Each claim holds on the engine: at
    /// the five probe designs the swing is the nominal pull-out divided by the ratio and
    /// efficiency, one slider step of the operating temperature moves it and one of the minimum
    /// temperature or the variation does not; and with the rating halfway between the swing and
    /// cold-high at the gearbox input, the check follows cold-high on either side of the swing.
    #[test]
    fn the_gearbox_texts_claim_what_the_engine_does() {
        use crate::engine::explain::notes::Step;
        use crate::engine::explain::probes::{probe_designs, step_input};
        use crate::engine::gearbox::{ABOVE_THE_GEARBOX_RATING, WITHIN_THE_GEARBOX_RATING};
        const RATING: &str = "coupling.gearbox_input_rating_Nm";
        const RIPPLE: &str = "model.gearbox_input_ripple_Nm";
        // The check row's help is the engine's: the overlay has no row for it.
        const CHECK: &str = "gearbox.cold_high_check";
        // The words: what each text must name, then what none may say ("no check" was the
        // rating's help until the check landed).
        let names = |path: &str, text: &str, words: &[&str]| {
            let lower = text.to_lowercase();
            for word in words {
                assert!(lower.contains(word), "{path}: {word:?} in {text:?}");
            }
        };
        names(
            RATING,
            help_for(RATING),
            &["gearbox check", "cold-high", "gearbox input"],
        );
        names(
            RIPPLE,
            help_for(RIPPLE),
            &[
                "operating temperature",
                "without the production allowance",
                "ratio and efficiency",
                "cold-high at the gearbox input",
                "pull-out at the gearbox output",
            ],
        );
        names(
            CHECK,
            help_for(CHECK),
            &[
                "gearbox output",
                "gearbox input",
                "fatigue from repeated slips is not checked",
            ],
        );
        // The gear ratio's help names the ripple row's quantity by the row's own name (slice 2's
        // one-name rule; BL-059 G-3 round 1 S-3): "slip swing", as the row reads.
        names(RIPPLE, label_for(RIPPLE), &["slip swing"]);
        names(
            "coupling.gear_ratio",
            help_for("coupling.gear_ratio"),
            &["the slip swing"],
        );
        for text in [
            help_for(RATING),
            label_for(RIPPLE),
            help_for(RIPPLE),
            help_for(CHECK),
        ] {
            let lower = text.to_lowercase();
            for word in ["no check", "peak", "warm"] {
                assert!(!lower.contains(word), "{word:?} in {text:?}");
            }
        }
        // Under the spreadsheet-details toggle the rating keeps the spreadsheet's own note; the
        // ripple row's engine help is empty, so it shows none.
        assert!(original_help(RATING).is_some());
        assert_eq!(
            original_help(RATING),
            metadata(RATING).map(|(_, help)| help)
        );
        assert_eq!(original_help(RIPPLE), None);
        // The swing, on the engine.
        for (name, design) in probe_designs() {
            let ripple = number(&design, RIPPLE);
            let ratio_and_efficiency = design.coupling.gear_ratio * design.coupling.gear_efficiency;
            let through = number(&design, "model.pullout_Nm") / ratio_and_efficiency;
            assert!(
                (ripple - through).abs() <= 1e-12 * through.abs(),
                "{name}: {ripple} against {through}"
            );
            for (input, moves) in [
                ("coupling.op_temp_C", true),
                ("metal.min_temp_C", false),
                ("metal.variation", false),
            ] {
                let stepped = step_input(&design, input, Step::Raise)
                    .unwrap_or_else(|e| panic!("{name}: {input}: {e:?}"));
                assert_eq!(
                    number(&stepped, RIPPLE) != ripple,
                    moves,
                    "{name}: {input} one step up"
                );
            }
        }
        // The check compares cold-high at the gearbox input, not the swing: with the rating
        // halfway between them it follows cold-high, above the swing at the defaults and below
        // it with the operating temperature under the minimum (-40 against 20 °C, variation
        // 0.1: both temperatures on their sliders).
        let mut colder_running = DesignInputs::default();
        colder_running.coupling.op_temp_C = -40.0;
        colder_running.metal.min_temp_C = 20.0;
        colder_running.metal.variation = 0.1;
        for (design, passes) in [(DesignInputs::default(), false), (colder_running, true)] {
            let results = compute_all(&design);
            let ripple = results.model.gearbox_input_ripple_Nm;
            let cold_high = results.metal.cold_high_input_Nm;
            assert_eq!(cold_high < ripple, passes, "{cold_high} against {ripple}");
            let mut between = design.clone();
            between.coupling.gearbox_input_rating_Nm = (ripple + cold_high) / 2.0;
            let verdict = compute_all(&between).gearbox.cold_high_check;
            if passes {
                assert_eq!(verdict, WITHIN_THE_GEARBOX_RATING);
            } else {
                assert!(verdict.starts_with(ABOVE_THE_GEARBOX_RATING), "{verdict}");
            }
        }
    }

    #[test]
    fn the_batch_b_claims_beyond_one_result_hold() {
        let example = DesignInputs::default();
        // metal.boss_od_mm: the boss sets the outside diameter only when it is the largest part.
        let outside = dimensions(&example).expect("a size").diameter_mm;
        assert!(example.metal.boss_od_mm < outside);
        let mut big_boss = example.clone();
        big_boss.metal.boss_od_mm = outside + 1.0;
        assert_eq!(
            dimensions(&big_boss).expect("a size").diameter_mm,
            outside + 1.0
        );
        // metal.cap_od_mm: the cap sets the outside diameter only when it is the largest part.
        let cap_step = 0.1;
        assert_eq!(example.metal.cap_od_mm, outside);
        let mut wider_cap = example.clone();
        wider_cap.metal.cap_od_mm += cap_step;
        assert_eq!(
            dimensions(&wider_cap).expect("a size").diameter_mm,
            outside + cap_step
        );
        let mut wider_cap_under_boss = big_boss.clone();
        wider_cap_under_boss.metal.cap_od_mm += cap_step;
        assert_eq!(
            dimensions(&wider_cap_under_boss)
                .expect("a size")
                .diameter_mm,
            outside + 1.0
        );
        // metal.cap_od_mm: Size-tab diameter changes and Auto on the ring radius move the cap.
        let wider = resize_diameter(&example, outside + 1.0).expect("a wider design");
        assert!(wider.metal.cap_od_mm > example.metal.cap_od_mm);
        let radius = example.coupling.inner_back_apothem_mm + 0.5;
        let larger_ring = with_ring_radius(&example, radius).expect("a larger ring");
        assert!(larger_ring.metal.cap_od_mm > example.metal.cap_od_mm);
        // metal.cup_wall_corner_mm: a thicker wall grows the cup diameter by twice the change.
        let change = 0.1;
        let mut thicker = example.clone();
        thicker.metal.cup_wall_corner_mm += change;
        let grown = number(&thicker, "model.cup_od_mm") - number(&example, "model.cup_od_mm");
        assert!((grown - 2.0 * change).abs() < 1e-9, "{grown}");
        // metal.measured_drag_Nm: a measured drag replaces the estimate in both cases.
        let measured = claim_design("measured drag");
        let heat = measured.metal.measured_drag_Nm.expect("a drag")
            * number(&measured, "temperature.duty.slip_rad_s");
        assert_eq!(number(&measured, "temperature.slip_loss.used_W"), heat);
        assert_eq!(number(&measured, "temperature.slip_loss.high_W"), heat);
        // mass.magnets_g: a picked grade brings its density; an ungraded magnet keeps NdFeB's.
        let (manual, graded) = (claim_design("manual, no grade"), claim_design("Y30 grades"));
        assert_ne!(
            number(&manual, "mass.magnets_g"),
            number(&graded, "mass.magnets_g")
        );
        assert_eq!(
            number(&manual, "model.inner_magnet_density_g_mm3"),
            number(&example, "model.inner_magnet_density_g_mm3")
        );
        // model.inner_temp_check, model.outer_temp_check: no rating without a part or a grade.
        let unknown = Some(Value::Text("unknown".to_owned()));
        for path in ["model.inner_temp_check", "model.outer_temp_check"] {
            assert_eq!(compute_all(&manual).get(path), unknown, "{path}");
            assert_ne!(compute_all(&graded).get(path), unknown, "{path}");
            assert_ne!(compute_all(&example).get(path), unknown, "{path}");
        }
    }
}
