//! One input row of the left side (spec M4 "Sliders"): live recompute while dragging, a value
//! box for typed entry, arrow-key nudges, step snapping, logarithmic scale, per-field reset, a
//! dot when changed from default, a tooltip with help text and workbook cell; selectors as
//! drop-downs, an optional input as a checkbox and a slider, a text input as a text field.
//!
//! [`input_row`] draws a row from its [`InputEntry`] and the current value and returns the
//! edit asked for; the panel applies it with `InputSet::set`, so every edit is checked the
//! same way. The material, magnet-part and grade rows add their pickers
//! ([`crate::gui::pickers`]): a material choice's properties on hover, a part or grade picked
//! from the library tables.

use crate::engine::meta::{FieldType, InputMeta, SliderRange, Value};
use crate::gui::format::{format_value, with_unit};
use crate::gui::inputs::{InputEntry, input_tooltip, outside_range, step_decimals, text_hint};
use crate::gui::pickers::{choice_hover, picker_ui};

/// The changed-from-default dot.
pub const CHANGED_DOT: &str = "\u{2022}";

/// The per-field reset button.
pub const RESET_LABEL: &str = "reset";

/// The note on a value outside its slider range.
pub const OUTSIDE_RANGE_NOTE: &str = "outside the slider range";

/// The text of a blank optional input.
pub const BLANK_TEXT: &str = "blank";

/// What the user asked of an input row this frame.
#[derive(Clone, Debug, PartialEq)]
pub enum RowEdit {
    /// Set the input to this value.
    Set(Value),
    /// Back to the default.
    Reset,
}

/// What an input row drew and asked for.
pub struct RowOutput {
    /// The edit, if any.
    pub edit: Option<RowEdit>,
    /// The row's main widget (the slider, drop-down, checkbox or text field).
    pub widget: egui::Response,
}

/// A slider over `value` set up from the input's metadata: its range (logarithmic when
/// flagged), its step (also the arrow-key nudge), values rounded to the step's decimals
/// (decision M41-1), edits clamped to the range while a value already outside it is kept until
/// edited (decision M41-2), typed values applied on Enter or when the box loses focus, the
/// unit as a suffix.
pub fn slider<'a>(value: &'a mut f64, meta: &InputMeta, range: SliderRange) -> egui::Slider<'a> {
    let mut slider = egui::Slider::new(value, range.min..=range.max)
        .logarithmic(range.log)
        .clamping(egui::SliderClamping::Edits)
        .update_while_editing(false);
    if meta.ty == FieldType::I64 {
        slider = slider.integer();
    } else {
        slider = slider.max_decimals(step_decimals(range.step));
    }
    // After integer(), which sets a step of 1: pole counts step by 2.
    slider = slider.step_by(range.step);
    let suffix = with_unit(String::new(), meta.unit);
    if !suffix.is_empty() {
        slider = slider.suffix(suffix);
    }
    slider
}

/// The number in a value, if it is one.
fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Num(x) => Some(*x),
        Value::Int(i) => Some(*i as f64),
        _ => None,
    }
}

/// The value a slider position stands for, in the input's type.
fn typed(meta: &InputMeta, x: f64) -> Value {
    match meta.ty {
        FieldType::I64 => Value::Int(x.round() as i64),
        _ => Value::Num(x),
    }
}

/// Draws one input row: a header line (the changed dot, the label, an out-of-range note, the
/// reset button) and the widget under it. `seed` is where an optional input starts when the
/// user enters a value (`inputs::OPTIONAL_SEEDS`).
pub fn input_row(
    ui: &mut egui::Ui,
    entry: &InputEntry,
    current: &Value,
    seed: Option<f64>,
) -> RowOutput {
    let meta = entry.meta;
    let tooltip = input_tooltip(entry);
    let changed = *current != entry.default;
    let mut edit = None;
    ui.push_id(&entry.path, |ui| {
        ui.horizontal(|ui| {
            let dot = if changed { CHANGED_DOT } else { " " };
            ui.colored_label(ui.visuals().selection.stroke.color, dot)
                .on_hover_text("Changed from the default");
            ui.label(meta.label).on_hover_text(&tooltip);
            if outside_range(meta.range, current) {
                ui.colored_label(ui.visuals().warn_fg_color, OUTSIDE_RANGE_NOTE)
                    .on_hover_text(
                        "Kept until edited. The differential tests cover the slider range only.",
                    );
            }
            if changed
                && ui
                    .small_button(RESET_LABEL)
                    .on_hover_text(format!(
                        "Back to the default: {}",
                        format_value(&entry.default)
                    ))
                    .clicked()
            {
                edit = Some(RowEdit::Reset);
            }
        });
        let widget = widget(ui, entry, current, seed, &mut edit).on_hover_text(&tooltip);
        if let Some(picked) = picker_ui(ui, &entry.path, current) {
            edit = Some(picked);
        }
        // Text inputs only (`text_hint` knows no other path).
        if let Some(hint) = text_hint(&entry.path, current_text(current)) {
            ui.weak(hint);
        }
        RowOutput { edit, widget }
    })
    .inner
}

/// The text of a text input (empty for any other value): the row's hint and the part and
/// grade pickers read it.
pub(crate) fn current_text(value: &Value) -> &str {
    match value {
        Value::Text(text) => text,
        _ => "",
    }
}

/// The row's widget; sets `edit` when the user changed the value. A new edit replaces a reset
/// asked in the same frame (it cannot happen: the reset button and the widget are two clicks).
fn widget(
    ui: &mut egui::Ui,
    entry: &InputEntry,
    current: &Value,
    seed: Option<f64>,
    edit: &mut Option<RowEdit>,
) -> egui::Response {
    let meta = entry.meta;
    match (meta.ty, current) {
        (FieldType::I64, Value::Int(code)) if !meta.choices.is_empty() => {
            let mut selected = *code;
            let text = meta.choices.iter().find(|(c, _)| c == code).map_or_else(
                || format!("{code} (not a choice)"),
                |(_, t)| (*t).to_owned(),
            );
            let response = egui::ComboBox::from_id_salt("choice")
                .selected_text(text)
                .show_ui(ui, |ui| {
                    for &(choice, label) in meta.choices {
                        let option = ui.selectable_value(&mut selected, choice, label);
                        if let Some(properties) = choice_hover(&entry.path, choice) {
                            option.on_hover_text(properties);
                        }
                    }
                })
                .response;
            if selected != *code {
                *edit = Some(RowEdit::Set(Value::Int(selected)));
            }
            response
        }
        (FieldType::OptF64, _) => {
            ui.horizontal(|ui| {
                let mut entered = !matches!(current, Value::None);
                let checkbox = ui
                    .checkbox(&mut entered, "")
                    .on_hover_text("Enter a value; clear to leave the input blank");
                if checkbox.changed() {
                    *edit = Some(RowEdit::Set(match (entered, seed) {
                        (true, Some(x)) => Value::Num(x),
                        (true, None) => Value::Num(meta.range.map_or(0.0, |r| r.min)),
                        (false, _) => Value::None,
                    }));
                }
                // The slider while a value is entered, else the checkbox.
                match (number(current), meta.range) {
                    (Some(x), Some(range)) => {
                        let mut value = x;
                        let response = ui.add(slider(&mut value, meta, range));
                        if response.changed() && value != x {
                            *edit = Some(RowEdit::Set(Value::Num(value)));
                        }
                        response
                    }
                    _ => {
                        ui.weak(BLANK_TEXT);
                        checkbox
                    }
                }
            })
            .inner
        }
        (FieldType::Text, _) => {
            let mut text = current_text(current).to_owned();
            let response =
                ui.add(egui::TextEdit::singleline(&mut text).desired_width(f32::INFINITY));
            if response.changed() {
                *edit = Some(RowEdit::Set(Value::Text(text)));
            }
            response
        }
        _ => match (number(current), meta.range) {
            (Some(x), Some(range)) => {
                let mut value = x;
                let response = ui.add(slider(&mut value, meta, range));
                if response.changed() && value != x {
                    *edit = Some(RowEdit::Set(typed(meta, value)));
                }
                response
            }
            (Some(x), None) => {
                let mut value = x;
                let response = ui.add(egui::DragValue::new(&mut value));
                if response.changed() && value != x {
                    *edit = Some(RowEdit::Set(typed(meta, value)));
                }
                response
            }
            _ => ui.label(format_value(current)),
        },
    }
}
