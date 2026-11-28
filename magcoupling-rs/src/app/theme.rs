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

// The token table (spec 6.1). Every colour [`visuals`] sets is one of these, but the fill
// behind what is selected ([`SELECTION`]) and the edge of a control at rest ([`CONTROL`]).

/// Behind the drawing and the plots; a field's background.
pub const BG: Color32 = Color32::from_rgb(0x12, 0x14, 0x17);
/// The three columns and the top bar; the web page behind the canvas.
pub const PANEL: Color32 = Color32::from_rgb(0x18, 0x1b, 0x1f);
/// A button at rest, a menu, a tooltip.
pub const PANEL2: Color32 = Color32::from_rgb(0x1e, 0x22, 0x27);
/// Hairlines; the fill of a pressed or hovered button.
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

/// The fill behind selected text and of a selected choice: a colour outside the spec's table
/// (6.1), which names none that a reader can see behind text. A quiet blue, the hue of
/// [`LINK`]: 3:1 or more against a field and the panel and 2.8:1 on a tooltip, with [`TEXT`]
/// on it at 4.5:1 (the theme's tests pin each). Never the accent (decision C-5).
pub const SELECTION: Color32 = Color32::from_rgb(0x46, 0x6a, 0x90);

/// The edge of a control at rest (the stage 1 review, E-4 and R-6): the outline of a value
/// box, a framed button, a picker (open too) and a text box, and the fill of a checkbox's box,
/// a radio button, a slider's rail and handle and the disc behind a plot legend's dot (egui_plot
/// draws it in the resting fill and outline). The other colour outside the spec's table (6.1),
/// whose border colour is 1.2:1 on the panel: a grey of the tokens' hue, 3:1 or more on the
/// panel, a menu and a field, under [`MUTED`], with the tick of a ticked box ([`TEXT`]) at
/// 4.3:1 on it (the theme's tests pin each).
pub const CONTROL: Color32 = Color32::from_rgb(0x65, 0x6c, 0x76);

/// The corner radius of buttons, fields, menus and tooltips [points].
pub const RADIUS: u8 = 4;

/// Item spacing and button padding [points]: the values the panel's layout was built on (the
/// linkage app's). The look changes no position.
pub const ITEM_SPACING: egui::Vec2 = egui::vec2(6.0, 4.0);
pub const BUTTON_PADDING: egui::Vec2 = egui::vec2(8.0, 4.0);

/// One widget state: its fill (a checkbox's box, a slider's rail and handle), a button's fill
/// and its outline. Its text is in the text colour and its corners are [`RADIUS`].
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
/// and nothing casts a shadow (decision C-6). A control at rest shows its edge in [`CONTROL`];
/// under the pointer and pressed its fill stays and its outline brightens.
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
    // egui has one fill for selected text and for a selected choice. It is a blue a reader can
    // see behind text; a choice's name on it is the text colour, as a selected value's is (the
    // link blue would be 2.3:1 there). A focused field's outline takes that stroke too. The
    // accent is not used here: it stays rare (decision C-5).
    v.selection.bg_fill = SELECTION;
    v.selection.stroke = Stroke::new(1.0, TEXT);
    state(&mut v.widgets.noninteractive, PANEL2, PANEL2, hairline);
    state(
        &mut v.widgets.inactive,
        CONTROL,
        PANEL2,
        Stroke::new(1.0, CONTROL),
    );
    state(
        &mut v.widgets.hovered,
        CONTROL,
        BORDER,
        Stroke::new(1.0, MUTED),
    );
    state(
        &mut v.widgets.active,
        CONTROL,
        BORDER,
        Stroke::new(1.0, TEXT),
    );
    // An open picker's box (egui draws it in this state) keeps the resting outline, so opening
    // it does not make it look dimmed; menus drop this outline (egui's `menu_style`).
    state(
        &mut v.widgets.open,
        PANEL2,
        PANEL2,
        Stroke::new(1.0, CONTROL),
    );
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

/// The WCAG 2 relative luminance of an opaque colour.
#[cfg(test)]
pub(crate) fn luminance(colour: Color32) -> f64 {
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
#[cfg(test)]
pub(crate) fn contrast(a: Color32, b: Color32) -> f64 {
    let (a, b) = (luminance(a), luminance(b));
    (a.max(b) + 0.05) / (a.min(b) + 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // What is selected: selected text and a selected choice on the selection fill, the
        // choice's name in the text colour.
        assert_eq!(v.selection.bg_fill, SELECTION);
        assert_eq!(v.selection.stroke, Stroke::new(1.0, TEXT));
        // The widget states: fill, a button's fill, outline.
        let w = &v.widgets;
        for (name, state, fill, button, outline) in [
            ("noninteractive", w.noninteractive, PANEL2, PANEL2, hairline),
            (
                "inactive",
                w.inactive,
                CONTROL,
                PANEL2,
                Stroke::new(1.0, CONTROL),
            ),
            (
                "hovered",
                w.hovered,
                CONTROL,
                BORDER,
                Stroke::new(1.0, MUTED),
            ),
            ("active", w.active, CONTROL, BORDER, Stroke::new(1.0, TEXT)),
            ("open", w.open, PANEL2, PANEL2, Stroke::new(1.0, CONTROL)),
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
        // level colours are not drawn on the border colour (a hovered button's fill, whose
        // text is the text colour): there they are under 4.5:1, and the last assertions pin
        // that so the rule is not lost. The text colour is also the name of a selected choice
        // and a selected value, on the selection fill.
        // Not counted: a disabled button's faded text (WCAG exempts an inactive control), and
        // a caption, a link or a level colour while its text is selected with the pointer
        // (`selected_text_shows_on_every_surface_and_reads_on_its_fill`).
        let everywhere = [BG, PANEL, PANEL2, BORDER];
        let off_the_fill = [BG, PANEL, PANEL2];
        let drawn: [(&str, Color32, &[Color32]); 6] = [
            ("text", TEXT, &[BG, PANEL, PANEL2, BORDER, SELECTION]),
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
    fn selected_text_shows_on_every_surface_and_reads_on_its_fill() {
        // The whole-branch review (C-1, R-3, E-5): with the border token as its fill a
        // selection was 1.1:1 to 1.3:1, so a reader could not see what typing would replace.
        // A field, the panel, a tooltip: where text is selected.
        assert!(contrast(SELECTION, BG) >= 3.0);
        assert!(contrast(SELECTION, PANEL) >= 3.0);
        let one_decimal = |a, b| (contrast(a, b) * 10.0).round() / 10.0;
        assert_eq!(one_decimal(SELECTION, PANEL2), 2.8);
        // A selected value and a selected choice's name are the text colour, on the fill.
        assert!(contrast(TEXT, SELECTION) >= 4.5);
        // Why a tooltip gets 2.8:1 and not 3:1: the lightest fill that keeps the text colour at
        // 4.5:1 is under 3:1 on the tooltip's surface. The text comes first.
        let lightest_fill = (luminance(TEXT) + 0.05) / 4.5 - 0.05;
        assert!((lightest_fill + 0.05) / (luminance(PANEL2) + 0.05) < 3.0);
        // A blue, the hue of what is clickable, selected or traced (decision C-13): never the
        // accent's crimson or a failing check's red (decision C-5).
        let [r, g, b, _] = SELECTION.to_array();
        assert!(b > g && g > r, "{SELECTION:?}");
        // No other text colour reads on the fill, so none is the name of a selected choice
        // (while one is selected with the pointer its band shows and its words are faint).
        for token in [MUTED, LINK, BAD, WARN, OK] {
            assert!(contrast(token, SELECTION) < 4.5, "{token:?}");
        }
    }

    #[test]
    fn a_control_at_rest_shows_its_edge_at_three_to_one_on_every_surface_it_sits_on() {
        // The stage 1 review (E-4, R-6), by WCAG 2's rule for a control's edge (1.4.11): at
        // least 3:1 against what is around it. At rest a value box, a framed button, a picker
        // and a text box show their resting outline; a checkbox's box, a radio button and a
        // slider's rail and handle show their resting fill, and in a menu, which drops the
        // outline, a checkbox's box shows by its fill alone. Both were the border colour,
        // 1.2:1 on the panel. A control sits on the panel, in a menu and on a field.
        let v = visuals();
        let rest = v.widgets.inactive;
        assert!(rest.bg_stroke.width >= 1.0);
        for (what, edge) in [("outline", rest.bg_stroke.color), ("fill", rest.bg_fill)] {
            for (surface, colour) in [("the panel", PANEL), ("a menu", PANEL2), ("a field", BG)] {
                let ratio = contrast(edge, colour);
                assert!(ratio >= 3.0, "the resting {what} on {surface}: {ratio:.2}");
            }
        }
        // A ticked box's tick and a slider handle's ring are the text colour, on that fill.
        assert!(contrast(rest.fg_stroke.color, rest.bg_fill) >= 3.0);
        // Under the pointer and pressed, a box and a handle keep their fill and a box's outline
        // brightens: the muted colour, then the text colour. A handle's ring is the text colour
        // in every state (egui only thickens it and grows the handle), so its state shows by
        // the ring's width and the handle's size alone (decision D-98).
        let hovered = v.widgets.hovered;
        let pressed = v.widgets.active;
        assert_eq!(
            (hovered.bg_fill, pressed.bg_fill),
            (rest.bg_fill, rest.bg_fill)
        );
        assert_eq!(
            (hovered.bg_stroke.color, pressed.bg_stroke.color),
            (MUTED, TEXT)
        );
        assert!(luminance(MUTED) > luminance(rest.bg_stroke.color));
        assert_eq!(
            (hovered.fg_stroke.color, pressed.fg_stroke.color),
            (rest.fg_stroke.color, rest.fg_stroke.color)
        );
        // An open picker keeps the resting outline: egui draws its box in the open state's
        // colours, and opening it must not make it look dimmed.
        assert_eq!(v.widgets.open.bg_stroke, rest.bg_stroke);
        // The figures [`CONTROL`]'s comment quotes, to one decimal; the border colour it
        // replaces at rest.
        let one_decimal = |a, b| (contrast(a, b) * 10.0).round() / 10.0;
        assert_eq!(
            (one_decimal(CONTROL, PANEL), one_decimal(CONTROL, PANEL2)),
            (3.3, 3.0)
        );
        assert_eq!(
            (one_decimal(TEXT, CONTROL), one_decimal(BORDER, PANEL)),
            (4.3, 1.2)
        );
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
