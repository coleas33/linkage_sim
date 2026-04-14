//! Theme configuration for the linkage simulator GUI.

use eframe::egui;

/// Build the professional dark visuals inspired by CAD tools (SolidWorks, ANSYS).
pub(crate) fn cad_dark_visuals() -> egui::Visuals {
    let mut v = egui::Visuals::dark();

    // Darker, more professional background tones
    v.panel_fill = egui::Color32::from_rgb(30, 32, 38);
    v.window_fill = egui::Color32::from_rgb(35, 37, 44);
    v.extreme_bg_color = egui::Color32::from_rgb(20, 22, 28);
    v.faint_bg_color = egui::Color32::from_rgb(38, 40, 48);

    // Accent color for selections and interactions
    v.selection.bg_fill = egui::Color32::from_rgb(40, 100, 200);
    v.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 160, 255));

    // Widget styling — more rounded, cleaner
    v.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(42, 44, 52);
    v.widgets.noninteractive.bg_stroke =
        egui::Stroke::new(0.5, egui::Color32::from_rgb(60, 62, 72));

    v.widgets.inactive.bg_fill = egui::Color32::from_rgb(50, 52, 62);
    v.widgets.inactive.bg_stroke = egui::Stroke::new(0.5, egui::Color32::from_rgb(70, 72, 82));

    v.widgets.hovered.bg_fill = egui::Color32::from_rgb(60, 65, 80);
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 140, 220));

    v.widgets.active.bg_fill = egui::Color32::from_rgb(40, 100, 200);
    v.widgets.active.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 160, 255));

    // Separator and window stroke
    v.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(55, 58, 68));

    v
}

/// Apply the full CAD theme to the egui context (visuals + spacing).
/// Called once at startup.
pub(crate) fn apply_cad_theme(ctx: &egui::Context) {
    ctx.set_visuals(cad_dark_visuals());

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(6.0, 4.0);
    style.spacing.button_padding = egui::vec2(8.0, 4.0);
    ctx.set_style(style);
}

/// Convert a color to grayscale using luminance weights.
fn gray(c: egui::Color32) -> egui::Color32 {
    let lum = (c.r() as f32 * 0.299 + c.g() as f32 * 0.587 + c.b() as f32 * 0.114) as u8;
    egui::Color32::from_rgba_premultiplied(lum, lum, lum, c.a())
}

/// Override egui visuals with grayscale colors (Nathan Mode).
pub(crate) fn apply_nathan_mode(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();

    v.widgets.noninteractive.bg_fill = gray(v.widgets.noninteractive.bg_fill);
    v.widgets.noninteractive.fg_stroke.color = gray(v.widgets.noninteractive.fg_stroke.color);
    v.widgets.inactive.bg_fill = gray(v.widgets.inactive.bg_fill);
    v.widgets.inactive.fg_stroke.color = gray(v.widgets.inactive.fg_stroke.color);
    v.widgets.hovered.bg_fill = gray(v.widgets.hovered.bg_fill);
    v.widgets.hovered.fg_stroke.color = gray(v.widgets.hovered.fg_stroke.color);
    v.widgets.active.bg_fill = gray(v.widgets.active.bg_fill);
    v.widgets.active.fg_stroke.color = gray(v.widgets.active.fg_stroke.color);
    v.widgets.open.bg_fill = gray(v.widgets.open.bg_fill);
    v.widgets.open.fg_stroke.color = gray(v.widgets.open.fg_stroke.color);
    v.selection.bg_fill = gray(v.selection.bg_fill);
    v.selection.stroke.color = gray(v.selection.stroke.color);
    v.hyperlink_color = gray(v.hyperlink_color);
    v.window_fill = gray(v.window_fill);
    v.panel_fill = gray(v.panel_fill);
    v.extreme_bg_color = gray(v.extreme_bg_color);

    ctx.set_visuals(v);
}

/// Restore the custom CAD-inspired dark visuals after leaving Nathan Mode.
pub(crate) fn restore_normal_visuals(ctx: &egui::Context) {
    ctx.set_visuals(cad_dark_visuals());
}
