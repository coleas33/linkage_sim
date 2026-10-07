//! The standalone app's theme: the linkage app's (spec M4 "Session": "Theme matches the
//! linkage app"). The linkage app applies its CAD dark visuals at start-up whatever the
//! system's light or dark preference; so does this app (decision M41-3), and the web page's
//! background is the same panel colour.
//!
//! The visuals are a copy of `cad_dark_visuals` in `linkage-sim-rs/src/gui/theme.rs` (the
//! crates are separate: linkage-sim-rs will depend on this one in M5, not the other way
//! round). A test compares the two function bodies, so they cannot drift apart. The panel
//! itself sets no theme: in M5 the linkage app's own theme applies.

/// Build the professional dark visuals inspired by CAD tools (SolidWorks, ANSYS).
pub fn cad_dark_visuals() -> egui::Visuals {
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

/// The linkage app's spacing (`apply_cad_theme`): item spacing and button padding [points].
pub const ITEM_SPACING: egui::Vec2 = egui::vec2(6.0, 4.0);
pub const BUTTON_PADDING: egui::Vec2 = egui::vec2(8.0, 4.0);

/// Applies the theme: dark whatever the system prefers, with the CAD visuals and the linkage
/// app's spacing.
pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.visuals = cad_dark_visuals();
        style.spacing.item_spacing = ITEM_SPACING;
        style.spacing.button_padding = BUTTON_PADDING;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The body of `fn cad_dark_visuals`, from its signature to the closing brace,
    /// with line endings normalized: a checkout may write either file with CRLF.
    fn body(source: &str) -> String {
        let source = source.replace("\r\n", "\n");
        let start = source
            .find("fn cad_dark_visuals() -> egui::Visuals {")
            .expect("the function is there");
        let end = source[start..].find("\n}\n").expect("it ends") + start;
        source[start..end].to_owned()
    }

    #[test]
    fn the_visuals_are_the_linkage_app_s() {
        let linkage = include_str!("../../../linkage-sim-rs/src/gui/theme.rs");
        let ours = include_str!("theme.rs");
        assert_eq!(body(ours), body(linkage));
        // And the spacing of its apply_cad_theme.
        assert!(linkage.contains("style.spacing.item_spacing = egui::vec2(6.0, 4.0);"));
        assert!(linkage.contains("style.spacing.button_padding = egui::vec2(8.0, 4.0);"));
        assert_eq!(
            (ITEM_SPACING, BUTTON_PADDING),
            (egui::vec2(6.0, 4.0), egui::vec2(8.0, 4.0))
        );
    }

    #[test]
    fn the_theme_is_dark_whatever_the_system_prefers() {
        let ctx = egui::Context::default();
        ctx.options_mut(|o| o.fallback_theme = egui::Theme::Light);
        apply(&ctx);
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        assert_eq!(ctx.theme(), egui::Theme::Dark);
        assert_eq!(ctx.style().visuals, cad_dark_visuals());
        assert_eq!(ctx.style().spacing.item_spacing, ITEM_SPACING);
        // The system turning light changes nothing: the preference is dark.
        let light = egui::RawInput {
            system_theme: Some(egui::Theme::Light),
            ..Default::default()
        };
        let _ = ctx.run(light, |_| {});
        assert_eq!(ctx.theme(), egui::Theme::Dark);
    }

    #[test]
    fn the_web_page_background_is_the_panel_colour() {
        let page = include_str!("../../../linkage-sim-rs/web/tools/magcoupler/index.html");
        let [r, g, b, _] = cad_dark_visuals().panel_fill.to_array();
        let colour = format!("background: #{r:02x}{g:02x}{b:02x};");
        assert!(page.contains(&colour), "index.html has no {colour}");
    }
}
