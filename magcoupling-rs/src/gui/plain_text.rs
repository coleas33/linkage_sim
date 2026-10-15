//! The names and help the panel shows for the inputs and results (guidance slice 2, spec 3.5;
//! rulings R1, S2-4, S2-7): a GUI-side overlay over the engine's metadata, keyed by path.
//!
//! The engine's labels and help keep the workbook wording: `tests/python_schema.rs` holds every
//! ported one to the Python engine byte for byte, and the results table, the exports, design
//! files and share links keep them. On screen, [`label_for`] gives one name per quantity
//! (`LABELS`, spec 3.5's canonical names) and [`help_for`] the overlay's help (`HELP`)
//! where the engine's is empty, terse, stale or wrong; [`original_help`] keeps the engine's
//! text for the spreadsheet-details toggle, and [`hint_for`] the short line under a Key design
//! input or a Size-tab control (`HINTS`). Every lookup falls back to the metadata, so a path
//! the overlay leaves out reads as the engine names it.
//!
//! The Size tab's Outside diameter is a control, not an input: it moves the ring radius through
//! `engine::size_controls`, so it has no row here. A fix on the ring radius names that control
//! (`guidance::SIZE_TAB_ALIASES`, which wins over `LABELS` in fix text only); the input row,
//! the equation terms and the tooltips call it Ring radius.
//!
//! Search matches the display label, the catalogue label, the path and the cell
//! (`format::search_haystack`). No egui.

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

/// The display label of an input or a result, by path: spec 3.5's canonical names. A row is
/// never longer than the catalogue label it replaces and names one quantity only (tests). A
/// name other modules draw too is a constant above, so it is typed once.
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
];

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
        "Reduction ratio of the gearbox; the coupling sits on its wheel side. A higher ratio lowers the coupling torques referred to the gearbox input, such as the slip ripple.",
    ),
    (
        "coupling.gear_efficiency",
        "Share of torque the gearbox passes. With the ratio it refers the coupling torques to the gearbox input; a higher efficiency lowers them there.",
    ),
    (
        "coupling.gearbox_input_rating_Nm",
        "Peak torque the gearbox input is rated for. Only an informational reference uses it: no check compares the coupling's torques with it.",
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
        "Magnet strength (remanence) of the inner manual blocks, used only with no library part and no grade picked. Stronger raises torque, and the cold-high torque the clamp must hold.",
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
        "Magnet strength (remanence) of the outer manual blocks, used only with no library part and no grade picked. Stronger raises torque, and the cold-high torque the clamp must hold.",
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
    // Shaft, key and clamps.
    (
        "coupling.bore_mm",
        "The shaft diameter under the hub and clamp. Larger lets each screw carry more torque but moves the screws outward, so fewer sizes may fit, and thins the hub wall.",
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
        "Blank keeps each ring's length; longer adds torque, length and clamp demand.",
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
        "Sets both rings' magnet length; longer adds torque and clamp demand.",
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

/// The engine's help of `path` where `HELP` replaces a non-empty one, shown under the
/// spreadsheet-details toggle as the original note; `None` where the overlay fills an empty
/// help or has no row.
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

/// [`original_help`] over `table`.
fn original_in(table: &[(&'static str, &'static str)], path: &str) -> Option<&'static str> {
    row(table, path)?;
    metadata(path)
        .map(|(_, help)| help)
        .filter(|help| !help.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::explain::plain::{plain_text_violations, sentence_count, word_count};
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
        // (T7), the name is "Ring radius".
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
        ];
        for (path, label) in want {
            assert_eq!(label_for(path), label, "{path}");
        }
        assert_eq!(LABELS.len(), want.len(), "no row beyond the spec's table");
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

    /// A stand-in overlay: the mechanism is tested on it while `HELP` fills (T7, T8). One row
    /// over an input with engine help, one over an input without, one over a result.
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
        // The original is kept only where the overlay replaces a non-empty engine help.
        assert_eq!(
            original_in(TEST_HELP, "metal.face_gap_mm"),
            Some(engine("metal.face_gap_mm"))
        );
        assert_eq!(original_in(TEST_HELP, "metal.runout_mm"), None);
        assert_eq!(
            original_in(TEST_HELP, "model.pullout_Nm"),
            Some(engine("model.pullout_Nm"))
        );
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
        // has one (the engine's kept as the original only where it is not empty), else the
        // engine's help and no original.
        for entry in InputCatalogue::get().all() {
            let path = entry.path.as_str();
            let engine = entry.meta.help;
            match row(HELP, path) {
                Some(text) => {
                    assert_eq!(help_for(path), text, "{path}");
                    assert_eq!(
                        original_help(path),
                        Some(engine).filter(|help| !help.is_empty()),
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
    fn each_display_label_names_one_quantity() {
        // No overlay label is any other input's or result's name on screen, ignoring case.
        let inputs = InputCatalogue::get().all().map(|e| e.path.as_str());
        let results = table_entries().iter().map(|e| e.path.as_str());
        let all: Vec<&str> = inputs.chain(results).collect();
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
    #[ignore = "un-ignored in T8, when the overlay fills every input whose engine help is empty"]
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
/// glyphs are the `tests` module's checks; the coverage of every input (ignored until T8) too.
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
