//! Direct controls for the complete envelope, alongside the existing magnet count and gap.
//! The engine adapter links housing dimensions; every torque still comes from compute_all.
//!
//! Under each control, its what-moves-what lines (spec 3.3: [`Consequences`], computed by the
//! guidance layer). The panel's centre view draws the torque readouts (spec 3.2) with the
//! texts here: [`REQUIRED`], [`HOT_LOW`] with [`torque_gap_text`], [`pullout_label`], and
//! [`NO_PREDICTION`] or [`NO_TORQUE`] where the model makes none ([`predicts`]).

use crate::engine::meta::{InputSet, SliderRange, Value};
use crate::engine::size_controls::{
    dimensions, dimensions_with_minimum_diameter, resize_diameter_with_minimum, resize_length,
};
use crate::gui::dashboard::end_effect_out_of_range;
use crate::gui::format::{fixed, signed_percent};
use crate::gui::guidance::Consequences;
use crate::gui::input_ui::slider;
use crate::gui::inputs::InputCatalogue;
use crate::{DesignInputs, DesignResults};

pub const HEADING: &str = "Size and torque";
pub const MINIMUM: &str = "Minimum OD";
pub const DIAMETER: &str = "Outside diameter";
pub const LENGTH: &str = "Overall length";
pub const GAP: &str = "Magnet face gap";
pub const MAGNETS: &str = "Magnets per ring";
pub const LINKED_HOUSING: &str = "Diameter and length edits resize the associated housing, keeping its allowances. Gap and magnet count remain your choices.";

/// The label of the requirement's slider (`metal.required_min_Nm`).
pub const REQUIRED: &str = "Required";

/// The label of the hot-low torque (`metal.torque_hot_low_Nm`), its gap to the requirement
/// beside it ([`torque_gap_text`]).
pub const HOT_LOW: &str = "Hot-low";

/// What a torque readout shows, the gap left out, where the end effect is outside the model's
/// range ([`predicts`]).
pub const NO_PREDICTION: &str =
    "no prediction (magnets too short for the end-effect model at this pole pitch)";

/// What a torque readout shows, the gap left out, for a torque that is not finite while the end
/// effect is in range ([`predicts`]).
pub const NO_TORQUE: &str = "no prediction (the torque is not finite for this design)";

/// The label of the nominal pull-out (`model.pullout_Nm`) at the operating temperature
/// `op_temp_C` [°C] (`metal.op_temp_C`), in its shortest decimal form: "50", or "37.5" at the
/// input's 0.5 °C step (-0 reads 0).
#[allow(non_snake_case)] // unit suffix
pub fn pullout_label(op_temp_C: f64) -> String {
    let op_temp_C = if op_temp_C == 0.0 { 0.0 } else { op_temp_C };
    format!("Pull-out torque (nominal, at {op_temp_C} °C)")
}

/// The hot-low torque's gap to the requirement, and whether it meets it: `hot_low_Nm >=
/// required_Nm`, as the engine's hot minimum check (which fails `hot_low < required`). Two
/// decimals of N·m and a whole percent of the requirement, each signed: "short 0.29 N·m
/// (-12 %)", "margin +0.40 N·m (+16 %)". The percent is left out when the requirement is not a
/// positive torque (the slider stops at its metadata minimum; a design file need not).
#[allow(non_snake_case)] // unit suffix
pub fn torque_gap_text(hot_low_Nm: f64, required_Nm: f64) -> (String, bool) {
    let meets = hot_low_Nm >= required_Nm;
    let gap_Nm = (hot_low_Nm - required_Nm).abs();
    let (word, torque) = if meets {
        ("margin", format!("+{}", fixed(gap_Nm, 2)))
    } else {
        ("short", fixed(gap_Nm, 2))
    };
    let text = if required_Nm > 0.0 {
        // `signed_percent` writes a change that rounds to zero as "+0 %". A shortfall keeps its
        // sign ("-0 %"), so a design a hair under its requirement never reads like a pass.
        let percent = signed_percent(gap_Nm / required_Nm);
        let percent = if meets {
            percent
        } else {
            percent.replacen('+', "-", 1)
        };
        format!("{word} {torque} N·m ({percent})")
    } else {
        format!("{word} {torque} N·m")
    };
    (text, meets)
}

/// Whether the Size tab shows `torque_Nm` as a number: the end effect is inside the model's
/// range (no banner) and the torque is finite. Otherwise the readout reads [`NO_PREDICTION`] or
/// [`NO_TORQUE`] (spec 3.2 and 5).
#[allow(non_snake_case)] // unit suffix
pub(crate) fn predicts(results: &DesignResults, torque_Nm: f64) -> bool {
    end_effect_out_of_range(results).is_none() && torque_Nm.is_finite()
}

pub(crate) struct ControlsOutput {
    pub inputs: DesignInputs,
    pub minimum_diameter_mm: Option<f64>,
    pub widgets: Vec<egui::Id>,
    pub error: Option<String>,
}

/// Draw before computing the panel's results, so the readouts reflect this frame's edits.
/// `consequences` are `inputs`' what-moves-what lines (spec 3.3), each drawn under its control:
/// they show the design before this frame's edit, and the panel asks for the next frame.
pub(crate) fn controls_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    minimum_diameter_mm: Option<f64>,
    consequences: &Consequences,
) -> ControlsOutput {
    let mut output = ControlsOutput {
        inputs: inputs.clone(),
        minimum_diameter_mm,
        widgets: Vec::new(),
        error: None,
    };
    let catalogue = InputCatalogue::get();
    ui.push_id("direct_size_controls", |ui| {
        let mut minimum_widget = None;
        if let Some(legacy) = dimensions(inputs) {
            let entry = catalogue
                .entry("coupling.inner_back_apothem_mm")
                .expect("the diameter has input metadata");
            let range = legacy.minimum_diameter_range;
            let mut minimum = minimum_diameter_mm
                .unwrap_or(legacy.diameter_range.min.min(legacy.diameter_mm))
                .clamp(range.min, range.max);
            ui.label(MINIMUM).on_hover_text(
                "Sets the lower end of the diameter slider without changing your current size.",
            );
            let response = ui
                .push_id("minimum_diameter_mm", |ui| {
                    ui.add(slider(&mut minimum, entry.meta, range))
                })
                .inner;
            minimum_widget = Some(response.id);
            if response.changed() {
                output.minimum_diameter_mm = Some(minimum);
            } else if minimum_diameter_mm.is_none()
                && legacy.diameter_mm < legacy.diameter_range.min
            {
                // Pin a loaded design's starting floor, so increasing its diameter later
                // does not move the floor upward on every idle frame.
                output.minimum_diameter_mm = Some(minimum);
            }
            // A loaded smaller design starts with its actual diameter exposed. The range
            // preference itself is local to this view, like the input ordering preference.
            let size = dimensions_with_minimum_diameter(inputs, minimum)
                .expect("a bounded minimum defines a diameter range");
            for (label, path, mut value, range, line) in [
                (
                    DIAMETER,
                    "coupling.inner_back_apothem_mm",
                    size.diameter_mm,
                    size.diameter_range,
                    consequences.diameter_line(),
                ),
                (
                    LENGTH,
                    "coupling.magnets.axial_length_mm",
                    size.overall_length_mm,
                    size.length_range,
                    consequences.length_line(),
                ),
            ] {
                ui.label(label);
                let entry = catalogue
                    .entry(path)
                    .expect("a size control has input metadata");
                let response = ui
                    .push_id(path, |ui| ui.add(slider(&mut value, entry.meta, range)))
                    .inner;
                output.widgets.push(response.id);
                if response.changed() {
                    let resized = if path == "coupling.inner_back_apothem_mm" {
                        resize_diameter_with_minimum(&output.inputs, value, minimum)
                    } else {
                        resize_length(&output.inputs, value)
                    };
                    match resized {
                        Ok(resized) => output.inputs = resized,
                        Err(error) => output.error = Some(error),
                    }
                }
                if let Some(line) = line {
                    consequence_line(ui, &line);
                }
            }
            ui.weak(format!(
                "Shaft and housing range floor: {:.1} mm",
                range.min
            ));
        } else {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "Size controls unavailable for these dimensions. Check the detailed inputs.",
            );
        }
        for (label, path, lines) in [
            (GAP, "metal.face_gap_mm", Vec::new()),
            (
                MAGNETS,
                "coupling.npole",
                consequences.magnet_lines(inputs.coupling.npole),
            ),
        ] {
            let entry = catalogue
                .entry(path)
                .expect("a size control has input metadata");
            let current = output
                .inputs
                .get(path)
                .expect("a size control has an input");
            let mut value = match current {
                Value::Num(value) => value,
                Value::Int(value) => value as f64,
                _ => unreachable!("the gap and magnet count are numbers"),
            };
            ui.label(label).on_hover_text(entry.meta.help);
            let range: SliderRange = entry.meta.range.expect("a size control has a slider");
            let response = ui
                .push_id(path, |ui| ui.add(slider(&mut value, entry.meta, range)))
                .inner;
            output.widgets.push(response.id);
            if response.changed() {
                let value = if path == "coupling.npole" {
                    Value::Int(value as i64)
                } else {
                    Value::Num(value)
                };
                if let Err(error) = output.inputs.set(path, value) {
                    output.error = Some(format!("{error}"));
                }
            }
            for line in &lines {
                consequence_line(ui, line);
            }
        }
        if let Some(widget) = minimum_widget {
            output.widgets.push(widget);
        }
        ui.label("Minimum OD changes the slider limit; your current diameter stays unchanged.");
        ui.label(LINKED_HOUSING);
    });
    output
}

/// One what-moves-what line under its control: weak text, wrapped to the side's width.
fn consequence_line(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(egui::RichText::new(text).weak()).wrap());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::gui::guidance::plain_text_violations;
    use crate::gui::test_support::{SCREEN, assert_glyphs, short_magnets, sized_frame};

    #[test]
    fn the_pull_out_label_names_the_live_operating_temperature() {
        assert_eq!(pullout_label(50.0), "Pull-out torque (nominal, at 50 °C)");
        assert_eq!(pullout_label(37.5), "Pull-out torque (nominal, at 37.5 °C)");
        assert_eq!(pullout_label(-40.0), "Pull-out torque (nominal, at -40 °C)");
        assert_eq!(pullout_label(-0.0), "Pull-out torque (nominal, at 0 °C)");
    }

    #[test]
    fn the_gap_reads_short_or_a_margin_against_the_requirement() {
        assert_eq!(
            torque_gap_text(2.21, 2.5),
            ("short 0.29 N·m (-12 %)".to_owned(), false)
        );
        assert_eq!(
            torque_gap_text(2.9, 2.5),
            ("margin +0.40 N·m (+16 %)".to_owned(), true)
        );
        // The engine's hot minimum check passes at equality (it fails `hot_low < required`).
        assert_eq!(
            torque_gap_text(2.5, 2.5),
            ("margin +0.00 N·m (+0 %)".to_owned(), true)
        );
        // A shortfall too small to show still reads short, with its sign.
        assert_eq!(
            torque_gap_text(2.499, 2.5),
            ("short 0.00 N·m (-0 %)".to_owned(), false)
        );
    }

    #[test]
    fn a_requirement_that_is_no_positive_torque_drops_the_percent() {
        // Spec 5 "Requirement <= 0": a design file can hold one (the slider stops at its
        // metadata minimum). The gap still reads, without a share of a non-positive torque.
        assert_eq!(
            torque_gap_text(2.0, 0.0),
            ("margin +2.00 N·m".to_owned(), true)
        );
        assert_eq!(
            torque_gap_text(2.0, -1.0),
            ("margin +3.00 N·m".to_owned(), true)
        );
        assert_eq!(
            torque_gap_text(-0.5, 0.0),
            ("short 0.50 N·m".to_owned(), false)
        );
    }

    #[test]
    fn a_torque_is_shown_only_inside_the_end_effect_range_and_finite() {
        let results = compute_all(&DesignInputs::default());
        assert!(predicts(&results, results.metal.torque_hot_low_Nm));
        assert!(predicts(&results, results.model.pullout_Nm));
        // Spec 5: a torque that is not finite reads no prediction, banner or not.
        for torque in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!predicts(&results, torque), "{torque}");
        }
        // With the end-effect banner even a finite torque is no prediction.
        let short = compute_all(&short_magnets());
        assert!(crate::gui::dashboard::end_effect_banner(&short).is_some());
        assert!(short.metal.torque_hot_low_Nm.is_finite());
        assert!(!predicts(&short, short.metal.torque_hot_low_Nm));
        assert!(!predicts(&short, short.model.pullout_Nm));
    }

    #[test]
    fn the_readout_texts_have_glyphs_and_plain_words() {
        let ctx = egui::Context::default();
        // The fonts load on the context's first frame.
        sized_frame(&ctx, SCREEN, Vec::new(), |_| {});
        let texts = [
            REQUIRED.to_owned(),
            HOT_LOW.to_owned(),
            NO_PREDICTION.to_owned(),
            NO_TORQUE.to_owned(),
            pullout_label(50.0),
            pullout_label(-37.5),
            torque_gap_text(2.21, 2.5).0,
            torque_gap_text(2.9, 2.5).0,
            torque_gap_text(2.0, 0.0).0,
        ];
        for text in &texts {
            assert_glyphs(&ctx, text, "a Size tab readout");
            let violations = plain_text_violations(text);
            assert!(violations.is_empty(), "{text:?}: {violations:?}");
        }
    }
}
