//! The panel's own part of the look (UX overhaul spec 6.3). The panel takes its colours from
//! the host's `Visuals`, and an egui theme has no field for a brand accent or for a passing
//! check: the panel owns those two, at the values of the spec's token table (6.1). The
//! standalone app's theme (`app::theme`) lists them in its table under the same names.
//!
//! The accent marks two things and no more: the brand mark before the title ([`mark_ui`]) and
//! the active tab of the centre's row ([`tab_ui`]).

/// The brand accent, crimson (decision C-5). Never a text colour, and never on a value or a
/// check: red means a failing check and nothing else.
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0xc8, 0x10, 0x2e);

/// A passing check, in a dark theme.
pub const OK: egui::Color32 = egui::Color32::from_rgb(0x4c, 0xc3, 0x8a);

/// The side of the brand mark [points].
pub const MARK_SIDE: f32 = 8.0;

/// The width of the line under the active tab [points].
pub const UNDERLINE_WIDTH: f32 = 2.0;

/// The brand mark: a small square in the accent, at the middle of a heading's height. The
/// title follows it.
pub fn mark_ui(ui: &mut egui::Ui) {
    let height = ui.text_style_height(&egui::TextStyle::Heading);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(MARK_SIDE, height), egui::Sense::hover());
    let square = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(MARK_SIDE));
    ui.painter().rect_filled(square, 2.0, ACCENT);
}

/// A tab of the centre's row. The active tab is marked by a line in the accent under its
/// label, not by a filled chip, and its label is in the strong text colour; the other tabs
/// are in the weak text colour until the pointer is on them.
pub fn tab_ui(ui: &mut egui::Ui, selected: bool, label: &str) -> egui::Response {
    let visuals = ui.visuals_mut();
    let was = (visuals.selection, visuals.widgets.inactive);
    visuals.selection.bg_fill = egui::Color32::TRANSPARENT;
    visuals.selection.stroke.color = visuals.widgets.active.fg_stroke.color;
    visuals.widgets.inactive.bg_stroke = egui::Stroke::NONE;
    visuals.widgets.inactive.fg_stroke.color = visuals.weak_text_color();
    let response = ui.selectable_label(selected, label);
    let visuals = ui.visuals_mut();
    (visuals.selection, visuals.widgets.inactive) = was;
    if selected {
        ui.painter().hline(
            response.rect.x_range(),
            response.rect.bottom(),
            egui::Stroke::new(UNDERLINE_WIDTH, ACCENT),
        );
    }
    response
}
