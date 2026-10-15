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

/// The display label of an input or a result, by path: spec 3.5's canonical names. A row is
/// never longer than the catalogue label it replaces and names one quantity only (tests).
const LABELS: &[(&str, &str)] = &[
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

/// The overlay help of an input or a result, by path: 1-2 sentences of at most 30 words, what
/// it physically is, what it moves and which way, and the trade-off (spec 3.5). Filled by the
/// slice-2 help batches; while a path has no row its engine help shows.
const HELP: &[(&str, &str)] = &[];

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
const HINTS: &[(&str, &str)] = &[];

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

/// The hint line under the input at `path`, if `HINTS` has one.
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
        // No torque is ever called guaranteed (slice-1 spec 8).
        for &(path, label) in LABELS {
            assert!(!label.to_lowercase().contains("guarantee"), "{path}");
        }
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
        // The public accessors read HELP itself, for every input.
        for entry in InputCatalogue::get().all() {
            assert_eq!(help_for(&entry.path), help_in(HELP, &entry.path));
            assert_eq!(original_help(&entry.path), original_in(HELP, &entry.path));
        }
    }

    #[test]
    fn every_overlay_text_is_plain_words_with_glyphs_and_within_its_caps() {
        // Spec 3.5 and 6 test 9: every label, help and hint passes the plain-words guard and
        // the glyph test, and keeps the caps (help <= 30 words in 1-2 sentences, a hint <= 12
        // words in one, a label a name, not a sentence). The counters are plain.rs's (A-16).
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
