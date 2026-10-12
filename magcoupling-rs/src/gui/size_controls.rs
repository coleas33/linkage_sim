//! Direct controls for the complete envelope, alongside the existing magnet count and gap.
//! The engine adapter links housing dimensions; every torque still comes from compute_all.

use crate::DesignInputs;
use crate::engine::meta::{InputSet, SliderRange, Value};
use crate::engine::size_controls::{dimensions, resize_diameter, resize_length};
use crate::gui::input_ui::slider;
use crate::gui::inputs::InputCatalogue;

pub const HEADING: &str = "Size and torque";
pub const DIAMETER: &str = "Outside diameter";
pub const LENGTH: &str = "Overall length";
pub const GAP: &str = "Magnet face gap";
pub const MAGNETS: &str = "Magnets per ring";
pub const LINKED_HOUSING: &str = "Diameter and length edits resize the associated housing, keeping its allowances. Gap and magnet count remain your choices.";

pub(crate) struct ControlsOutput {
    pub inputs: DesignInputs,
    pub widgets: Vec<egui::Id>,
    pub error: Option<String>,
}

/// Draw before computing the panel's results, so the readouts reflect this frame's edits.
pub(crate) fn controls_ui(ui: &mut egui::Ui, inputs: &DesignInputs) -> ControlsOutput {
    let mut output = ControlsOutput {
        inputs: inputs.clone(),
        widgets: Vec::new(),
        error: None,
    };
    let catalogue = InputCatalogue::get();
    ui.push_id("direct_size_controls", |ui| {
        if let Some(size) = dimensions(inputs) {
            for (label, path, mut value, range, resize) in [
                (
                    DIAMETER,
                    "coupling.inner_back_apothem_mm",
                    size.diameter_mm,
                    size.diameter_range,
                    resize_diameter as fn(&DesignInputs, f64) -> Result<DesignInputs, String>,
                ),
                (
                    LENGTH,
                    "coupling.magnets.axial_length_mm",
                    size.overall_length_mm,
                    size.length_range,
                    resize_length,
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
                    match resize(&output.inputs, value) {
                        Ok(resized) => output.inputs = resized,
                        Err(error) => output.error = Some(error),
                    }
                }
            }
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
        ui.label(LINKED_HOUSING);
    });
    output
}
