//! Mobile layout detection and style adjustments.

use eframe::egui;

/// Returns true if the screen is small enough to use mobile layout.
pub fn is_mobile(ctx: &egui::Context) -> bool {
    ctx.screen_rect().width() < 768.0
}

/// Apply mobile-friendly style overrides.
pub fn apply_mobile_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.button_padding = egui::vec2(16.0, 12.0);
    style.spacing.item_spacing = egui::vec2(10.0, 8.0);
    style.spacing.interact_size.y = 44.0;
    style.spacing.slider_width = 200.0;
    style.text_styles.insert(egui::TextStyle::Body, egui::FontId::proportional(16.0));
    style.text_styles.insert(egui::TextStyle::Button, egui::FontId::proportional(16.0));
    style.text_styles.insert(egui::TextStyle::Small, egui::FontId::proportional(13.0));
    ctx.set_style(style);
}

/// Restore default desktop style.
pub fn apply_desktop_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.button_padding = egui::vec2(8.0, 4.0);
    style.spacing.item_spacing = egui::vec2(6.0, 4.0);
    style.spacing.interact_size.y = 18.0;
    style.spacing.slider_width = 100.0;
    style.text_styles.insert(egui::TextStyle::Body, egui::FontId::proportional(14.0));
    style.text_styles.insert(egui::TextStyle::Button, egui::FontId::proportional(14.0));
    style.text_styles.insert(egui::TextStyle::Small, egui::FontId::proportional(10.0));
    ctx.set_style(style);
}
