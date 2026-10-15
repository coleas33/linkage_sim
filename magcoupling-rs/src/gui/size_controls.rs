//! Direct controls for the complete envelope, alongside the existing magnet count and gap.
//! The engine adapter links housing dimensions; every torque still comes from compute_all.
//! Auto (spec guidance 3.5) runs the inverse sizing from here: it locks the slider of the size it
//! sets ([`variable_slider`]) at the value shown and names the change ([`change_text`]).
//!
//! Under each control, its what-moves-what lines (spec 3.3: [`Consequences`], computed by the
//! guidance layer). The panel's centre view draws the torque readouts (spec 3.2) with the
//! texts here: [`REQUIRED`], [`HOT_LOW`] with [`torque_gap_text`], [`pullout_label`], and
//! [`NO_PREDICTION`] or [`NO_TORQUE`] where the model makes none ([`predicts`]).

use crate::engine::meta::{InputMeta, InputSet, SliderRange, Value};
use crate::engine::size_controls::{
    dimensions, dimensions_with_minimum_diameter, resize_diameter_with_minimum, resize_length,
};
use crate::engine::sizing::FreeVariable;
use crate::gui::dashboard::end_effect_out_of_range;
use crate::gui::format::{fixed, signed_percent};
use crate::gui::guidance::Consequences;
use crate::gui::input_ui::slider;
use crate::gui::inputs::InputCatalogue;
use crate::gui::plain_text::{HOT_LOW_TORQUE, REQUIRED_TORQUE, help_for};
use crate::gui::sizing::{SizingMode, SizingState, auto_choice, variable_label};
use crate::{DesignInputs, DesignResults, compute_all};

pub const HEADING: &str = "Size and torque";
pub const MINIMUM: &str = "Minimum OD";
pub const DIAMETER: &str = "Outside diameter";
pub const LENGTH: &str = "Overall length";
pub const GAP: &str = "Magnet face gap";
pub const MAGNETS: &str = "Magnets per ring";
/// The Size tab's footnote (O-7): what the diameter and length sliders move
/// ([`crate::engine::size_controls`]).
pub const LINKED_HOUSING: &str = "Outside diameter moves the ring radius; the cap and its thread follow the cup. Overall length sets both rings' magnet length, and the housing follows. Gap and magnet count stay as you set them.";

/// The hover text of the Outside diameter control (R-3): what it is (the largest of the cup,
/// the cap and the shaft boss) and what moving it changes (the ring radius; the cap and its
/// thread by the cup's change).
pub const DIAMETER_HOVER: &str = "The largest of the cup, cap and shaft boss diameters. Moving it changes the ring radius; the cap and its thread follow the cup.";

/// The hover text of the Overall length control: what it is and what moving it changes (the
/// axial magnet length override of both rings; the housing follows, `engine::housing`).
pub const LENGTH_HOVER: &str = "The axial length of the coupling: the magnets and the housing around them. Moving it sets both rings' magnet length, and the housing follows.";

/// The Auto row (spec guidance 3.5): "Auto: smallest [size] that meets the required torque".
pub const AUTO: &str = "Auto: smallest";
pub const AUTO_SUFFIX: &str = "that meets the required torque";
/// The hover text of the Auto toggle.
pub const AUTO_HOVER: &str = "On: the solver sets the chosen size to the smallest value whose hot-low torque meets the required torque, skipping layouts whose magnets do not fit, whose keyway breaks through the hub or that the end-effect model cannot predict; its slider locks. Off: the design keeps the value shown.";
/// Under the slider Auto sets, and its hover text.
pub const SET_BY_AUTO: &str = "Set by Auto; turn Auto off to edit";
/// The start of the Key design group's line while Auto is on ([`auto_status`]).
pub const AUTO_STATUS_START: &str = "Auto: ";
/// The Key design group's button to the Size tab, beside [`auto_status`].
pub const OPEN_SIZE_TAB: &str = "Open the Size tab";

/// The Size-tab slider a free variable sets, by its label: Auto locks it.
pub fn variable_slider(variable: FreeVariable) -> &'static str {
    match variable {
        FreeVariable::AxialLength => LENGTH,
        FreeVariable::RingRadius => DIAMETER,
        FreeVariable::MagnetsPerRing => MAGNETS,
    }
}

/// The Auto outcome line of a solved design: the size Auto set, its value in `from` (the
/// inputs) and in `to` (the design shown), e.g. "Magnet length 12.7 -> 13.8 mm". Ring size reads
/// as the Outside diameter its slider shows ("Outside diameter 42.8 -> 40.1 mm"), or as the ring
/// radius where a design has no diameter range (an odd count). Millimetres to the sliders'
/// 0.1 mm, the magnet count whole.
pub fn change_text(variable: FreeVariable, from: &DesignInputs, to: &DesignInputs) -> String {
    let millimetres = |name: &str, value: &dyn Fn(&DesignInputs) -> f64| {
        format!(
            "{name} {} -> {} mm",
            fixed(value(from), 1),
            fixed(value(to), 1)
        )
    };
    match variable {
        FreeVariable::AxialLength => millimetres("Magnet length", &|design| {
            // A blank override reads as the parts' own length.
            design
                .coupling
                .magnets
                .axial_length_mm
                .unwrap_or_else(|| compute_all(design).model.active_length_mm)
        }),
        FreeVariable::RingRadius => match (dimensions(from), dimensions(to)) {
            (Some(from), Some(to)) => format!(
                "{DIAMETER} {} -> {} mm",
                fixed(from.diameter_mm, 1),
                fixed(to.diameter_mm, 1)
            ),
            _ => millimetres(variable_label(variable), &|design| {
                design.coupling.inner_back_apothem_mm
            }),
        },
        FreeVariable::MagnetsPerRing => format!(
            "{} {} -> {}",
            variable_label(variable),
            from.coupling.npole,
            to.coupling.npole
        ),
    }
}

/// The Key design group's line while Auto is on: which size the solver sets, and where.
pub fn auto_status(variable: FreeVariable) -> String {
    format!(
        "{AUTO_STATUS_START}{} set by the solver (Size tab)",
        auto_choice(variable)
    )
}

/// The Auto row: the toggle, the size it sets and the rest of the sentence. Returns the sizing
/// state asked for; the panel applies a mode change through its own switch, which adopts the
/// value shown when Auto turns off (decision M41-7).
pub(crate) fn auto_ui(ui: &mut egui::Ui, state: SizingState) -> SizingState {
    let mut asked = state;
    ui.horizontal_wrapped(|ui| {
        let mut on = state.mode == SizingMode::TorqueToMagnets;
        if ui
            .checkbox(&mut on, AUTO)
            .on_hover_text(AUTO_HOVER)
            .changed()
        {
            asked.mode = if on {
                SizingMode::TorqueToMagnets
            } else {
                SizingMode::MagnetsToTorque
            };
        }
        egui::ComboBox::from_id_salt("auto_free_variable")
            .selected_text(auto_choice(asked.variable))
            .show_ui(ui, |ui| {
                for variable in FreeVariable::ALL {
                    ui.selectable_value(&mut asked.variable, variable, auto_choice(variable));
                }
            });
        ui.label(AUTO_SUFFIX);
    });
    asked
}

/// The label of the requirement's slider (`metal.required_min_Nm`): the input's display label.
pub const REQUIRED: &str = REQUIRED_TORQUE;

/// The label of the hot-low torque (`metal.torque_hot_low_Nm`), its gap to the requirement
/// beside it ([`torque_gap_text`]): the dashboard row's display label.
pub const HOT_LOW: &str = HOT_LOW_TORQUE;

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
/// `inputs` is the design shown. `consequences` are its what-moves-what lines (spec 3.3), each
/// drawn under its control: they show the design before this frame's edit, and the panel asks
/// for the next frame. With `auto`, the slider of the size Auto sets ([`variable_slider`]) is
/// locked at its value there: disabled, saying why.
pub(crate) fn controls_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    minimum_diameter_mm: Option<f64>,
    consequences: &Consequences,
    auto: Option<FreeVariable>,
) -> ControlsOutput {
    let is_locked = |label: &str| auto.is_some_and(|variable| variable_slider(variable) == label);
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
            for (label, hover, path, mut value, range, line) in [
                (
                    DIAMETER,
                    DIAMETER_HOVER,
                    "coupling.inner_back_apothem_mm",
                    size.diameter_mm,
                    size.diameter_range,
                    consequences.diameter_line(),
                ),
                (
                    LENGTH,
                    LENGTH_HOVER,
                    "coupling.magnets.axial_length_mm",
                    size.overall_length_mm,
                    size.length_range,
                    consequences.length_line(),
                ),
            ] {
                ui.label(label).on_hover_text(hover);
                let entry = catalogue
                    .entry(path)
                    .expect("a size control has input metadata");
                let response =
                    size_slider(ui, path, &mut value, entry.meta, range, is_locked(label));
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
                consequences.magnet_lines(inputs.coupling.npole, auto),
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
            ui.label(label).on_hover_text(help_for(path));
            let range: SliderRange = entry.meta.range.expect("a size control has a slider");
            let response = size_slider(ui, path, &mut value, entry.meta, range, is_locked(label));
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

/// One size slider: a `locked` one (set by Auto) is disabled, so egui reports no change for it,
/// and it says why on hover and in a line under it.
fn size_slider(
    ui: &mut egui::Ui,
    path: &str,
    value: &mut f64,
    meta: &InputMeta,
    range: SliderRange,
    locked: bool,
) -> egui::Response {
    let response = ui
        .push_id(path, |ui| {
            ui.add_enabled(!locked, slider(value, meta, range))
        })
        .inner
        .on_disabled_hover_text(SET_BY_AUTO);
    if locked {
        ui.weak(SET_BY_AUTO);
    }
    response
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
            DIAMETER_HOVER.to_owned(),
            LENGTH_HOVER.to_owned(),
            LINKED_HOUSING.to_owned(),
        ];
        for text in &texts {
            assert_glyphs(&ctx, text, "a Size tab readout");
            let violations = plain_text_violations(text);
            assert!(violations.is_empty(), "{text:?}: {violations:?}");
        }
    }

    #[test]
    fn the_size_tab_names_each_quantity_as_the_rest_of_the_panel_does() {
        use crate::gui::plain_text::label_for;
        // Spec 3.5, one name per quantity: the requirement's slider and the hot-low readout
        // read as the input row and the dashboard row name them.
        assert_eq!(REQUIRED, label_for("metal.required_min_Nm"));
        assert_eq!(HOT_LOW, label_for("metal.torque_hot_low_Nm"));
        // The Auto row and its hover name the required torque; the hover the hot-low torque.
        let required = label_for("metal.required_min_Nm").to_lowercase();
        assert!(AUTO_SUFFIX.contains(&required), "{AUTO_SUFFIX}");
        assert!(AUTO_HOVER.contains(&required), "{AUTO_HOVER}");
        assert!(
            AUTO_HOVER.contains(&label_for("metal.torque_hot_low_Nm").to_lowercase()),
            "{AUTO_HOVER}"
        );
        // O-7: the footnote says the Outside diameter moves the ring radius, so the slider and
        // the input row are seen to be one quantity.
        let ring = label_for("coupling.inner_back_apothem_mm").to_lowercase();
        assert!(LINKED_HOUSING.contains(&ring), "{LINKED_HOUSING}");
    }

    #[test]
    fn the_diameter_and_length_hovers_say_what_they_move_within_the_help_caps() {
        use crate::engine::explain::plain::{sentence_count, word_count};
        use crate::gui::plain_text::label_for;
        // R-3: the Outside diameter control says it moves the ring radius (the input row's
        // name), and Overall length the magnet length of both rings.
        let ring = label_for("coupling.inner_back_apothem_mm").to_lowercase();
        assert!(DIAMETER_HOVER.contains(&ring), "{DIAMETER_HOVER}");
        assert!(LENGTH_HOVER.contains("magnet length"), "{LENGTH_HOVER}");
        // The Outside diameter counts the shaft boss, so it is not the space claim's rotating
        // diameter (the cup and the cap only, `engine::metal_design`): the hover never uses
        // that name.
        assert!(DIAMETER_HOVER.contains("shaft boss"), "{DIAMETER_HOVER}");
        assert!(
            !DIAMETER_HOVER.to_lowercase().contains("rotating"),
            "{DIAMETER_HOVER}"
        );
        // Spec 3.5's help caps: 1-2 sentences, at most 30 words.
        for hover in [DIAMETER_HOVER, LENGTH_HOVER] {
            assert!(word_count(hover) <= 30, "{hover}");
            assert!(
                hover.ends_with('.') && (1..=2).contains(&sentence_count(hover)),
                "{hover}"
            );
        }
    }
}

#[cfg(test)]
mod auto_tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::sizing::FreeVariable;
    use crate::gui::format::fixed;
    use crate::gui::guidance::plain_text_violations;
    use crate::gui::sizing::{REQUIREMENT_NOT_POSITIVE, auto_choice, sized};
    use crate::gui::test_support::with_poles;

    #[test]
    fn each_free_variable_locks_its_own_size_slider() {
        assert_eq!(variable_slider(FreeVariable::AxialLength), LENGTH);
        assert_eq!(variable_slider(FreeVariable::RingRadius), DIAMETER);
        assert_eq!(variable_slider(FreeVariable::MagnetsPerRing), MAGNETS);
        // Never the gap or the minimum: they stay the user's.
        for variable in FreeVariable::ALL {
            assert_ne!(variable_slider(variable), GAP);
            assert_ne!(variable_slider(variable), MINIMUM);
        }
    }

    #[test]
    fn change_text_names_the_size_and_both_values() {
        let from = DesignInputs::default();
        // A blank override starts from the parts' own length (model.active_length_mm).
        let own = compute_all(&from).model.active_length_mm;
        let to = FreeVariable::AxialLength.apply(&from, 13.5);
        assert_eq!(
            change_text(FreeVariable::AxialLength, &from, &to),
            format!("Magnet length {} -> 13.5 mm", fixed(own, 1))
        );
        let set = FreeVariable::AxialLength.apply(&from, 20.0);
        assert_eq!(
            change_text(FreeVariable::AxialLength, &set, &to),
            "Magnet length 20.0 -> 13.5 mm"
        );
        // Ring size names the Outside diameter the slider shows, not the internal radius.
        let eight = with_poles(8);
        let to = sized(FreeVariable::RingRadius, &eight, 8.5);
        let diameter = |design: &DesignInputs| fixed(dimensions(design).unwrap().diameter_mm, 1);
        assert_ne!(diameter(&eight), diameter(&to), "the diameter moves");
        assert_eq!(
            change_text(FreeVariable::RingRadius, &eight, &to),
            format!(
                "Outside diameter {} -> {} mm",
                diameter(&eight),
                diameter(&to)
            )
        );
        // Without a diameter range (an odd count) it falls back to the ring radius.
        let odd = with_poles(7);
        let to = FreeVariable::RingRadius.apply(&odd, 8.5);
        assert_eq!(
            change_text(FreeVariable::RingRadius, &odd, &to),
            format!(
                "Ring radius {} -> 8.5 mm",
                fixed(odd.coupling.inner_back_apothem_mm, 1)
            )
        );
        let to = FreeVariable::MagnetsPerRing.apply(&from, 12.0);
        assert_eq!(
            change_text(FreeVariable::MagnetsPerRing, &from, &to),
            format!("Magnets per ring {} -> 12", from.coupling.npole)
        );
    }

    #[test]
    fn auto_status_names_the_choice_and_the_size_tab() {
        assert_eq!(
            auto_status(FreeVariable::AxialLength),
            "Auto: magnet length set by the solver (Size tab)"
        );
        for variable in FreeVariable::ALL {
            let status = auto_status(variable);
            assert!(status.starts_with(AUTO_STATUS_START), "{status}");
            assert!(status.contains(auto_choice(variable)), "{status}");
        }
    }

    #[test]
    fn every_auto_text_is_plain_words_with_ascii_arrows() {
        let from = DesignInputs::default();
        let mut texts: Vec<String> = [
            AUTO,
            AUTO_SUFFIX,
            AUTO_HOVER,
            SET_BY_AUTO,
            OPEN_SIZE_TAB,
            REQUIREMENT_NOT_POSITIVE,
        ]
        .map(str::to_owned)
        .to_vec();
        for variable in FreeVariable::ALL {
            texts.push(auto_choice(variable).to_owned());
            texts.push(auto_status(variable));
            let to = sized(variable, &from, variable.range().max);
            texts.push(change_text(variable, &from, &to));
        }
        for text in &texts {
            assert_eq!(plain_text_violations(text), Vec::<&str>::new(), "{text:?}");
            assert!(
                !text.contains('\u{2192}') && !text.contains('\u{2264}'),
                "{text:?}"
            );
        }
    }
}
