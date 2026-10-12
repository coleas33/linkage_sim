//! Direct controls for the complete envelope, alongside the existing magnet count and gap.
//! The engine adapter links housing dimensions; every torque still comes from compute_all.

use crate::DesignInputs;
use crate::engine::meta::{InputSet, SliderRange, Value};
use crate::engine::size_controls::{
    dimensions, dimensions_with_minimum_diameter, resize_diameter_with_minimum, resize_length,
};
use crate::gui::input_ui::slider;
use crate::gui::inputs::InputCatalogue;

pub const HEADING: &str = "Size and torque";
pub const MINIMUM: &str = "Minimum OD";
pub const DIAMETER: &str = "Outside diameter";
pub const LENGTH: &str = "Overall length";
pub const GAP: &str = "Magnet face gap";
pub const MAGNETS: &str = "Magnets per ring";
pub const LINKED_HOUSING: &str = "Diameter and length edits resize the associated housing, keeping its allowances. Gap and magnet count remain your choices.";

pub(crate) struct ControlsOutput {
    pub inputs: DesignInputs,
    pub minimum_diameter_mm: Option<f64>,
    pub widgets: Vec<egui::Id>,
    pub error: Option<String>,
}

/// Draw before computing the panel's results, so the readouts reflect this frame's edits.
pub(crate) fn controls_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    minimum_diameter_mm: Option<f64>,
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
            for (label, path, mut value, range) in [
                (
                    DIAMETER,
                    "coupling.inner_back_apothem_mm",
                    size.diameter_mm,
                    size.diameter_range,
                ),
                (
                    LENGTH,
                    "coupling.magnets.axial_length_mm",
                    size.overall_length_mm,
                    size.length_range,
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
        for (label, path) in [(GAP, "metal.face_gap_mm"), (MAGNETS, "coupling.npole")] {
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
        }
        if let Some(widget) = minimum_widget {
            output.widgets.push(widget);
        }
        ui.label("Minimum OD changes the slider limit; your current diameter stays unchanged.");
        ui.label(LINKED_HOUSING);
    });
    output
}
