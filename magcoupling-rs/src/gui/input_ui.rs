//! One input row of the left side (spec M4 "Sliders"): live recompute while dragging, a value
//! box for typed entry, arrow-key nudges, step snapping, logarithmic scale, per-field reset, a
//! dot when changed from the default it is handed (the panel's baseline value, UX overhaul spec
//! 7.1), a tooltip with help text (and, with the spreadsheet details,
//! path and workbook cell); selectors as
//! drop-downs, an optional input as a checkbox and a slider, a text input as a text field.
//!
//! [`input_row`] draws a row from its [`InputEntry`] and the current value and returns the
//! edit asked for; the panel applies it with `InputSet::set`, so every edit is checked the
//! same way. The material, magnet-part and grade rows add their pickers
//! ([`crate::gui::pickers`]): a material choice's properties on hover, a part or grade picked
//! from the library tables.
//!
//! [`hint_line`] draws the short always-visible hint under a Key design row or a Size-tab control
//! (slice 2 spec 3.5), as a [`weak_line`].

use crate::engine::meta::{FieldType, InputMeta, SliderRange, Value};
use crate::gui::format::{format_value, with_unit};
use crate::gui::inputs::{InputEntry, input_tooltip, outside_range, step_decimals, text_hint};
use crate::gui::pickers::{choice_hover, picker_ui};
use crate::gui::plain_text::{hint_for, label_for};

/// The changed-from-default dot.
pub const CHANGED_DOT: &str = "\u{2022}";

/// The per-field reset button.
pub const RESET_LABEL: &str = "reset";

/// The note on a value outside its slider range.
pub const OUTSIDE_RANGE_NOTE: &str = "outside the slider range";

/// The text of a blank optional input.
pub const BLANK_TEXT: &str = "blank";

/// The hover text of the changed-from-default dot, on an input row and an assumption alike.
pub const CHANGED_FROM_DEFAULT: &str = "Changed from the default";

/// The hover text of [`OUTSIDE_RANGE_NOTE`].
pub const OUTSIDE_RANGE_HINT: &str = "Kept until you edit it. Outside the slider range the calculator is not checked against the original spreadsheet.";

/// What the user asked of an input row this frame.
#[derive(Clone, Debug, PartialEq)]
pub enum RowEdit {
    /// Set the input to this value.
    Set(Value),
    /// Back to the default the row was drawn with.
    Reset,
}

/// What an input row drew and asked for.
pub struct RowOutput {
    /// The edit, if any.
    pub edit: Option<RowEdit>,
    /// The row's main widget (the slider, drop-down, checkbox or text field).
    pub widget: egui::Response,
    /// The label was clicked: select the input, whose Explain entry traces it (UX overhaul spec
    /// 5.2).
    pub label_clicked: bool,
}

/// The decimals a slider's value box shows: none for a count, else the step's.
fn slider_decimals(meta: &InputMeta, range: SliderRange) -> usize {
    match meta.ty {
        FieldType::I64 => 0,
        _ => step_decimals(range.step),
    }
}

/// A slider over `value` set up from the input's metadata: its range (logarithmic when
/// flagged), its step (also the arrow-key nudge), values rounded to the step's decimals
/// (decision M41-1), edits clamped to the range while a value already outside it is kept until
/// edited (decision M41-2), typed values applied on Enter or when the box loses focus, the
/// unit as a suffix. The box's own text is no typed value ([`typed_number`], BL-097).
pub fn slider<'a>(value: &'a mut f64, meta: &InputMeta, range: SliderRange) -> egui::Slider<'a> {
    let decimals = slider_decimals(meta, range);
    // The box's text as egui's own formatter writes it this frame, kept for the parser: egui
    // formats before it parses, in the same frame.
    let shown = std::rc::Rc::new(std::cell::RefCell::new(String::new()));
    let written = std::rc::Rc::clone(&shown);
    let mut slider = egui::Slider::new(value, range.min..=range.max)
        .logarithmic(range.log)
        .clamping(egui::SliderClamping::Edits)
        .update_while_editing(false)
        .custom_formatter(move |x, counts| {
            let text = egui::emath::format_with_decimals_in_range(x, counts);
            written.replace(text.clone());
            text
        })
        .custom_parser(move |text| typed_number(text, &shown.borrow()));
    if meta.ty == FieldType::I64 {
        slider = slider.integer();
    } else {
        slider = slider.max_decimals(decimals);
    }
    // After integer(), which sets a step of 1: pole counts step by 2.
    slider = slider.step_by(range.step);
    let suffix = with_unit(String::new(), meta.unit);
    if !suffix.is_empty() {
        slider = slider.suffix(suffix);
    }
    slider
}

/// Draws `slider` in the room the row has: a slider and its value box cannot wrap, so the
/// slider takes the width the box leaves, at most egui's own (BL-103: at a pixel ratio of 0.9
/// the longest box, '0.00000125664 T·m/A', ran 2.9 points past the left column's row). The
/// box is measured at the most decimals it can show, so it never needs more than it is given.
pub fn add_slider(
    ui: &mut egui::Ui,
    value: &mut f64,
    meta: &InputMeta,
    range: SliderRange,
) -> egui::Response {
    let decimals = slider_decimals(meta, range);
    let text = with_unit(
        egui::emath::format_with_decimals_in_range(*value, decimals..=decimals),
        meta.unit,
    );
    let font = egui::TextStyle::Button.resolve(ui.style());
    let galley = ui
        .painter()
        .layout_no_wrap(text, font, egui::Color32::PLACEHOLDER);
    let boxed = galley.size().x + 2.0 * ui.spacing().button_padding.x;
    // One point kept free, as `gui::wrapping_line` does: a width is rounded to the pixel.
    let room = ui.available_width() - ui.spacing().item_spacing.x - boxed - 1.0;
    ui.scope(|ui| {
        let width = &mut ui.spacing_mut().slider_width;
        *width = width.min(room.max(0.0));
        ui.add(slider(value, meta, range))
    })
    .inner
}

/// The number typed into a value box whose own text is `shown`, read as egui reads one (blanks
/// ignored, U+2212 a minus); none when `text` is no number, or reads as the number `shown`
/// does. egui writes a box's text back when the keyboard's focus leaves it, typed or not, so
/// the box's own text would round a value off its step, or bring one outside the range to its
/// end, on a Tab pass (BL-097): an edit, an undo step and a selection with nothing typed. The
/// same number typed again is no edit either. Only the box's own text counts, never a coarser
/// rounding of its value: egui reads any two numbers under about 1.9e-6 as the same, so a
/// coarser rounding of a tiny value would swallow a typed number (BL097-EP-1). Bits are
/// compared, so a NaN reads as itself.
///
/// egui fixes the box's text as the focus arrives, and `shown` is the text of the frame the
/// focus leaves: the same text, as egui picks its decimals from the value (held while the box
/// is edited, `update_while_editing(false)`) and the drag speed over the slider's width.
/// shortcut: in a frame with an arrow key the speed is the step, so a box left in the very
/// frame an arrow key is pressed is compared at the step's decimals, and its text, if shown at
/// fewer, reads as typed (as every box's did on main); keep the text of the frame the focus
/// arrives if that is ever seen.
fn typed_number(text: &str, shown: &str) -> Option<f64> {
    let text: String = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| if c == '\u{2212}' { '-' } else { c })
        .collect();
    let typed: f64 = text.parse().ok()?;
    let own = shown.parse::<f64>().map(f64::to_bits) == Ok(typed.to_bits());
    (!own).then_some(typed)
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
/// reset button) and the widget under it. `default` is the value the dot, the reset button's
/// hover and the tooltip's Default line compare with and show: the session's baseline value
/// (UX overhaul spec 7.1, C-22). `seed` is where an optional input starts when the
/// user enters a value (`inputs::OPTIONAL_SEEDS`). A `locked` row (Torque -> Magnets' free
/// variable) disables everything but its dot and label: the label still selects the input.
/// `show_details` (the header's Show spreadsheet details) adds the path, the cell and the
/// original spreadsheet note to the tooltip.
pub fn input_row(
    ui: &mut egui::Ui,
    entry: &InputEntry,
    current: &Value,
    default: &Value,
    seed: Option<f64>,
    locked: bool,
    show_details: bool,
) -> RowOutput {
    let meta = entry.meta;
    // Built only when a hover shows it (the closures below run only then).
    let tooltip = || input_tooltip(entry, default, show_details);
    let changed = current != default;
    let mut edit = None;
    let mut label_clicked = false;
    ui.push_id(&entry.path, |ui| {
        head_line(ui, changed, |ui| {
            // One Tab stop however the label wraps; after a wrapped label the note and the
            // reset button start a line of their own (SK-1).
            let label = egui::Label::new(label_for(&entry.path)).sense(egui::Sense::click());
            let label = crate::gui::label_block(ui, label).on_hover_ui(|ui| {
                ui.add(egui::Label::new(tooltip()));
            });
            label_clicked = label.clicked();
            // Each on the head line itself, not in a child of it: a child starts where the
            // label ends and wraps no further left, so it would run past the row's end.
            if outside_range(meta.range, current) {
                let note =
                    egui::RichText::new(OUTSIDE_RANGE_NOTE).color(ui.visuals().warn_fg_color);
                ui.add_enabled(!locked, egui::Label::new(note))
                    .on_hover_text(OUTSIDE_RANGE_HINT);
            }
            if changed
                && ui
                    .add_enabled(!locked, egui::Button::new(RESET_LABEL).small())
                    .on_hover_ui(|ui| {
                        ui.add(egui::Label::new(format!(
                            "Back to the default: {}",
                            format_value(default)
                        )));
                    })
                    .clicked()
            {
                edit = Some(RowEdit::Reset);
            }
        });
        let widget = ui
            .add_enabled_ui(!locked, |ui| {
                let widget = widget(ui, entry, current, seed, &mut edit).on_hover_ui(|ui| {
                    ui.add(egui::Label::new(tooltip()));
                });
                if let Some(picked) = picker_ui(ui, &entry.path, current) {
                    edit = Some(picked);
                }
                // Text inputs only (`text_hint` knows no other path).
                if let Some(hint) = text_hint(&entry.path, current_text(current)) {
                    ui.weak(hint);
                }
                widget
            })
            .inner;
        RowOutput {
            edit,
            widget,
            label_clicked,
        }
    })
    .inner
}

/// The head line of a row of the left column (an input's, an assumption's): the changed dot,
/// then what `rest` draws, on a line that wraps at the width the row is given. A line that does
/// not wrap runs a long label past the column's edge, and egui's side panel grows to what it
/// holds and keeps the width for the rest of the session (BL-099).
pub(crate) fn head_line(ui: &mut egui::Ui, changed: bool, rest: impl FnOnce(&mut egui::Ui)) {
    crate::gui::wrapping_line(ui, |ui| {
        let dot = if changed { CHANGED_DOT } else { " " };
        ui.colored_label(crate::gui::look::mark_color(ui.visuals()), dot)
            .on_hover_text(CHANGED_FROM_DEFAULT);
        rest(ui);
    });
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
                        let response = add_slider(ui, &mut value, meta, range);
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
                let response = add_slider(ui, &mut value, meta, range);
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

/// A line of weak text, wrapped to its region's width: an input's hint line ([`hint_line`]) and
/// the Size tab's what-moves-what lines under a control, and the torque card's lines.
pub(crate) fn weak_line(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(egui::RichText::new(text).weak()).wrap());
}

/// The hint line under a control (slice 2 spec 3.5): the [`hint_for`] text of `key` (an input
/// path, or one of the Size tab's `plain_text::SIZE_TAB_HINT_KEYS`) as a [`weak_line`];
/// nothing for a key without a hint.
pub(crate) fn hint_line(ui: &mut egui::Ui, key: &str) {
    if let Some(hint) = hint_for(key) {
        weak_line(ui, hint);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_box_s_own_text_is_no_typed_number() {
        // BL-097: 1.4123 at two decimals is shown as "1.41"; that text, and the same number
        // written another way, is not a typed value. Any other number is.
        assert_eq!(typed_number("1.41", "1.41"), None);
        assert_eq!(typed_number(" 1.410 ", "1.41"), None);
        assert_eq!(typed_number("1.4", "1.41"), Some(1.4));
        assert_eq!(typed_number("1.42", "1.41"), Some(1.42));
        // A whole number's box (a magnet count).
        assert_eq!(typed_number("7", "7"), None);
        assert_eq!(typed_number("8", "7"), Some(8.0));
        // Only the box's own text: 5000.006 shown as "5000" (egui writes fewer decimals where
        // they read the same to seven digits) takes a typed "5000.01".
        assert_eq!(typed_number("5000", "5000"), None);
        assert_eq!(typed_number("5000.01", "5000"), Some(5000.01));
        // BL097-EP-1: a tiny value's coarser roundings are no text of its box. egui reads any
        // two numbers under about 1.9e-6 as the same, so each was taken for the box's own.
        assert_eq!(typed_number("1.2566e-6", "0.00000125664"), Some(1.2566e-6));
        assert_eq!(typed_number("0.00000125664", "0.00000125664"), None);
        assert_eq!(typed_number("-1e-6", "-0.0000008"), Some(-1e-6));
        assert_eq!(typed_number("0", "0.00000000040"), Some(0.0));
        assert_eq!(typed_number("1e-9", "0.00000000053"), Some(1e-9));
        // Read as egui reads a number; text that is none changes nothing.
        assert_eq!(typed_number("\u{2212}1 000", "2"), Some(-1000.0));
        assert_eq!(typed_number("wide", "2"), None);
        assert_eq!(typed_number("NaN", "NaN"), None);
    }
}
