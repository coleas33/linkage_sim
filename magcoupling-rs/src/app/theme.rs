//! The standalone app's theme: "Charcoal and crimson" (UX overhaul spec 6; owner decision
//! O-4). The app applies it at start-up whatever the system's light or dark preference, and
//! the web page's background is the panel colour.
//!
//! It is the calculator's own look. The linkage app keeps its CAD dark visuals, so this
//! reverses decision M41-3 ("Theme matches the linkage app": the spec's question Q-1, taken on
//! its recommendation). The panel itself sets no theme: it reads the host's `Visuals`, so an
//! app that embeds it (M5) shows it in that app's theme.

use egui::style::WidgetVisuals;
use egui::{Color32, CornerRadius, Stroke};

// The token table (spec 6.1). Every colour [`visuals`] sets is one of these.

/// Behind the drawing and the plots; a field's background.
pub const BG: Color32 = Color32::from_rgb(0x12, 0x14, 0x17);
/// The three columns and the top bar; the web page behind the canvas.
pub const PANEL: Color32 = Color32::from_rgb(0x18, 0x1b, 0x1f);
/// A button at rest, a menu, a tooltip.
pub const PANEL2: Color32 = Color32::from_rgb(0x1e, 0x22, 0x27);
/// Hairlines; the fill of a pressed or hovered button and of a selected choice.
pub const BORDER: Color32 = Color32::from_rgb(0x26, 0x2b, 0x31);
/// Values, headings, body text.
pub const TEXT: Color32 = Color32::from_rgb(0xe6, 0xe8, 0xeb);
/// Captions and hints (decision C-14: lightened from the mockup's #7c838c, which is under
/// 4.5:1 on [`PANEL2`]).
pub const MUTED: Color32 = Color32::from_rgb(0x8a, 0x91, 0x9a);
/// The brand accent (decision C-5), which the panel draws from its own constant.
pub const ACCENT: Color32 = crate::gui::look::ACCENT;
/// A failing check.
pub const BAD: Color32 = Color32::from_rgb(0xff, 0x4d, 0x57);
/// A caution.
pub const WARN: Color32 = Color32::from_rgb(0xe0, 0xa1, 0x00);
/// A passing check, which the panel draws from its own constant.
pub const OK: Color32 = crate::gui::look::OK;
/// A link, and the mark of what is selected or traced (decision C-13: egui's link blue, the
/// hue of the drawing's dimension lines; blue means "click me").
pub const LINK: Color32 = Color32::from_rgb(90, 170, 255);

/// The corner radius of buttons, fields, menus and tooltips [points].
pub const RADIUS: u8 = 4;

/// Item spacing and button padding [points]: the values the panel's layout was built on (the
/// linkage app's). The look changes no position.
pub const ITEM_SPACING: egui::Vec2 = egui::vec2(6.0, 4.0);
pub const BUTTON_PADDING: egui::Vec2 = egui::vec2(8.0, 4.0);

/// One widget state: its fill (a checkbox's box, a slider's rail), a button's fill and its
/// outline. Its text is in the text colour and its corners are [`RADIUS`].
fn state(state: &mut WidgetVisuals, fill: Color32, button: Color32, outline: Stroke) {
    state.bg_fill = fill;
    state.weak_bg_fill = button;
    state.bg_stroke = outline;
    state.fg_stroke.color = TEXT;
    state.corner_radius = CornerRadius::same(RADIUS);
}

/// The theme's visuals: egui's dark visuals with the tokens in place of its colours.
///
/// Body text, values and headings share [`TEXT`]; egui has no bold, so `strong()` no longer
/// sets a value apart from the text beside it (a second, heavier face is the cure if a later
/// slice needs one). `weak()` is [`MUTED`], and the panel draws a label beside its value in it
/// (`gui::look::quiet_text`), so the value is the bright one of the pair. Borders are hairlines
/// and nothing casts a shadow (decision C-6).
pub fn visuals() -> egui::Visuals {
    let hairline = Stroke::new(1.0, BORDER);
    let radius = CornerRadius::same(RADIUS);
    let mut v = egui::Visuals::dark();
    v.panel_fill = PANEL;
    v.window_fill = PANEL2;
    v.window_stroke = hairline;
    v.window_shadow = egui::Shadow::NONE;
    v.popup_shadow = egui::Shadow::NONE;
    v.window_corner_radius = radius;
    v.menu_corner_radius = radius;
    v.extreme_bg_color = BG;
    v.faint_bg_color = PANEL2;
    v.code_bg_color = PANEL2;
    v.weak_text_color = Some(MUTED);
    v.hyperlink_color = LINK;
    v.warn_fg_color = WARN;
    v.error_fg_color = BAD;
    // A selected choice is a filled chip with its text in the link blue; selected text in a
    // field sits on the same fill. The accent is not used here: it stays rare (decision C-5).
    v.selection.bg_fill = BORDER;
    v.selection.stroke = Stroke::new(1.0, LINK);
    state(&mut v.widgets.noninteractive, PANEL2, PANEL2, hairline);
    state(&mut v.widgets.inactive, BORDER, PANEL2, hairline);
    state(
        &mut v.widgets.hovered,
        BORDER,
        BORDER,
        Stroke::new(1.0, MUTED),
    );
    state(
        &mut v.widgets.active,
        BORDER,
        BORDER,
        Stroke::new(1.0, TEXT),
    );
    state(&mut v.widgets.open, PANEL2, PANEL2, hairline);
    v
}

/// Applies the theme: dark whatever the system prefers, with [`visuals`] and the spacing.
pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.visuals = visuals();
        style.spacing.item_spacing = ITEM_SPACING;
        style.spacing.button_padding = BUTTON_PADDING;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WCAG 2 relative luminance of an opaque colour.
    fn luminance(colour: Color32) -> f64 {
        let linear = |channel: u8| {
            let s = f64::from(channel) / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * linear(colour.r()) + 0.7152 * linear(colour.g()) + 0.0722 * linear(colour.b())
    }

    /// The WCAG 2 contrast ratio of two opaque colours.
    fn contrast(a: Color32, b: Color32) -> f64 {
        let (a, b) = (luminance(a), luminance(b));
        (a.max(b) + 0.05) / (a.min(b) + 0.05)
    }

    #[test]
    fn the_tokens_are_the_spec_s_table() {
        // UX overhaul spec 6.1, value for value; the link blue is decision C-13's.
        let hex = |c: Color32| format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b());
        let table = [
            (BG, "#121417"),
            (PANEL, "#181b1f"),
            (PANEL2, "#1e2227"),
            (BORDER, "#262b31"),
            (TEXT, "#e6e8eb"),
            (MUTED, "#8a919a"),
            (ACCENT, "#c8102e"),
            (BAD, "#ff4d57"),
            (WARN, "#e0a100"),
            (OK, "#4cc38a"),
            (LINK, "#5aaaff"),
        ];
        for (token, value) in table {
            assert_eq!(hex(token), value);
        }
        assert_eq!(LINK, egui::Visuals::dark().hyperlink_color);
    }

    #[test]
    fn the_visuals_are_the_token_table() {
        // UX overhaul spec 8.1, test 6 (it replaces the_visuals_are_the_linkage_app_s: the
        // spec's question Q-1).
        let v = visuals();
        let hairline = Stroke::new(1.0, BORDER);
        let radius = CornerRadius::same(RADIUS);
        // Surfaces.
        assert_eq!(
            (v.panel_fill, v.window_fill, v.extreme_bg_color),
            (PANEL, PANEL2, BG)
        );
        assert_eq!((v.faint_bg_color, v.code_bg_color), (PANEL2, PANEL2));
        // Text: body and values, captions, links, a caution, a failing check.
        assert_eq!(
            (v.text_color(), v.strong_text_color(), v.weak_text_color()),
            (TEXT, TEXT, MUTED)
        );
        assert_eq!(
            (v.hyperlink_color, v.warn_fg_color, v.error_fg_color),
            (LINK, WARN, BAD)
        );
        // Hairlines, no shadow, one radius (decision C-6).
        assert_eq!(v.window_stroke, hairline);
        assert_eq!(
            (v.window_shadow, v.popup_shadow),
            (egui::Shadow::NONE, egui::Shadow::NONE)
        );
        assert_eq!(
            (v.window_corner_radius, v.menu_corner_radius),
            (radius, radius)
        );
        // What is selected: a chip on the border colour, its text in the link blue.
        assert_eq!(v.selection.bg_fill, BORDER);
        assert_eq!(v.selection.stroke, Stroke::new(1.0, LINK));
        // The widget states: fill, a button's fill, outline.
        let w = &v.widgets;
        for (name, state, fill, button, outline) in [
            ("noninteractive", w.noninteractive, PANEL2, PANEL2, hairline),
            ("inactive", w.inactive, BORDER, PANEL2, hairline),
            (
                "hovered",
                w.hovered,
                BORDER,
                BORDER,
                Stroke::new(1.0, MUTED),
            ),
            ("active", w.active, BORDER, BORDER, Stroke::new(1.0, TEXT)),
            ("open", w.open, PANEL2, PANEL2, hairline),
        ] {
            assert_eq!(
                (state.bg_fill, state.weak_bg_fill, state.bg_stroke),
                (fill, button, outline),
                "{name}"
            );
            assert_eq!(
                (state.fg_stroke.color, state.corner_radius),
                (TEXT, radius),
                "{name}"
            );
        }
        // Nothing else moved: with egui's values back in the fields above, these are egui's
        // dark visuals. A colour set outside the table fails here.
        let dark = egui::Visuals::dark();
        let mut rest = v.clone();
        rest.panel_fill = dark.panel_fill;
        rest.window_fill = dark.window_fill;
        rest.window_stroke = dark.window_stroke;
        rest.window_shadow = dark.window_shadow;
        rest.popup_shadow = dark.popup_shadow;
        rest.window_corner_radius = dark.window_corner_radius;
        rest.menu_corner_radius = dark.menu_corner_radius;
        rest.extreme_bg_color = dark.extreme_bg_color;
        rest.faint_bg_color = dark.faint_bg_color;
        rest.code_bg_color = dark.code_bg_color;
        rest.weak_text_color = dark.weak_text_color;
        rest.hyperlink_color = dark.hyperlink_color;
        rest.warn_fg_color = dark.warn_fg_color;
        rest.error_fg_color = dark.error_fg_color;
        rest.selection = dark.selection;
        for (ours, egui_s) in [
            (
                &mut rest.widgets.noninteractive,
                dark.widgets.noninteractive,
            ),
            (&mut rest.widgets.inactive, dark.widgets.inactive),
            (&mut rest.widgets.hovered, dark.widgets.hovered),
            (&mut rest.widgets.active, dark.widgets.active),
            (&mut rest.widgets.open, dark.widgets.open),
        ] {
            ours.bg_fill = egui_s.bg_fill;
            ours.weak_bg_fill = egui_s.weak_bg_fill;
            ours.bg_stroke = egui_s.bg_stroke;
            ours.fg_stroke.color = egui_s.fg_stroke.color;
            ours.corner_radius = egui_s.corner_radius;
        }
        assert_eq!(rest, dark);
        // The spacing the layout was built on.
        assert_eq!(
            (ITEM_SPACING, BUTTON_PADDING),
            (egui::vec2(6.0, 4.0), egui::vec2(8.0, 4.0))
        );
    }

    #[test]
    fn every_text_token_reads_on_every_surface_it_is_drawn_on() {
        // UX overhaul spec 6.1 and goal G-7: at least 4.5:1, computed here by the WCAG
        // formula. Text and links are drawn on all four surfaces. Captions and the three
        // level colours are not drawn on the border colour (a hovered button's and a selected
        // chip's fill, whose text is the text colour and the link blue): there they are under
        // 4.5:1, and the last assertions pin that so the rule is not lost.
        // Not counted: a disabled button's faded text (WCAG exempts an inactive control), and
        // a caption while its text is selected with the pointer.
        let everywhere = [BG, PANEL, PANEL2, BORDER];
        let off_the_fill = [BG, PANEL, PANEL2];
        let drawn: [(&str, Color32, &[Color32]); 6] = [
            ("text", TEXT, &everywhere),
            ("link", LINK, &everywhere),
            ("muted", MUTED, &off_the_fill),
            ("bad", BAD, &off_the_fill),
            ("warn", WARN, &off_the_fill),
            ("ok", OK, &off_the_fill),
        ];
        for (name, token, surfaces) in drawn {
            for &surface in surfaces {
                let ratio = contrast(token, surface);
                assert!(ratio >= 4.5, "{name} on {surface:?}: {ratio:.2}");
            }
        }
        // The spec's own figures (6.1), to one decimal.
        let one_decimal = |a, b| (contrast(a, b) * 10.0).round() / 10.0;
        assert_eq!(one_decimal(MUTED, PANEL), 5.4);
        assert_eq!(one_decimal(MUTED, PANEL2), 5.0);
        assert_eq!(one_decimal(BAD, PANEL), 5.3);
        assert_eq!(one_decimal(Color32::WHITE, ACCENT), 5.9);
        // Decision C-5: the accent is never a text colour. On the panel it is 2.9:1.
        assert_eq!(one_decimal(ACCENT, PANEL), 2.9);
        // Why captions and level colours stay off the border colour.
        assert!(contrast(MUTED, BORDER) < 4.5 && contrast(BAD, BORDER) < 4.5);
    }

    #[test]
    fn the_theme_is_dark_whatever_the_system_prefers() {
        let ctx = egui::Context::default();
        ctx.options_mut(|o| o.fallback_theme = egui::Theme::Light);
        apply(&ctx);
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        assert_eq!(ctx.theme(), egui::Theme::Dark);
        assert_eq!(ctx.style().visuals, visuals());
        assert_eq!(ctx.style().spacing.item_spacing, ITEM_SPACING);
        assert_eq!(ctx.style().spacing.button_padding, BUTTON_PADDING);
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
        let [r, g, b, _] = visuals().panel_fill.to_array();
        let colour = format!("background: #{r:02x}{g:02x}{b:02x};");
        assert!(page.contains(&colour), "index.html has no {colour}");
    }
}
