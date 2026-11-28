//! The standalone app's font: Inter (UX overhaul spec 6.2), ahead of egui's built-in fonts.
//!
//! The files are cuts of Inter Regular and Inter SemiBold, made once by
//! `assets/fonts/subset_inter.py` and committed with their licence (`assets/fonts/OFL.txt`,
//! SIL Open Font License 1.1); the build runs no font tools. What each cut does:
//!
//! - It holds the characters the app draws (Latin, Greek, the maths and unit signs in use),
//!   about 30 KB. A character outside it is drawn in a built-in font, which the glyph guards
//!   of the panel's tests refuse (`gui::test_support::assert_glyphs`).
//! - Its ten digits are Inter's tabular ones, so numbers keep their columns: egui cannot
//!   switch an OpenType feature on, so the file carries them as its default digits.
//! - Its ascender, descender and line gap are those of egui's built-in font, so a row is as
//!   tall as the font the panel's layout was built on made it (1.149 em; Inter's own rows are
//!   1.21 em) and a text size is rounded to whole pixels as that font's is.
//!
//! Two faces. egui has no bold, so the app points egui's Heading style at the SemiBold cut:
//! the title and the left column's heading, the panel's two `ui.heading` texts (the stage 1
//! review, S-1). Every other text is Regular; a tile's number keeps it, since the panel names
//! no font. The panel itself installs no font: in another host it is drawn in that host's
//! fonts, so the typesetter's substitutions for egui's built-in fonts
//! (`gui::typeset::glyph_safe`) stay as they are.

/// The cut of Inter Regular.
pub const INTER: &[u8] = include_bytes!("../../assets/fonts/Inter-Regular.subset.ttf");

/// Its name among egui's fonts.
pub const INTER_NAME: &str = "Inter";

/// The cut of Inter SemiBold, for the title and the headings: the same characters, digits and
/// rows as [`INTER`] (the same script, the same list).
pub const INTER_SEMIBOLD: &[u8] = include_bytes!("../../assets/fonts/Inter-SemiBold.subset.ttf");

/// Its name among egui's fonts, and the name of the family it leads.
pub const SEMIBOLD_NAME: &str = "Inter SemiBold";

/// The family the SemiBold cut leads: the title and the headings.
pub fn semibold() -> egui::FontFamily {
    egui::FontFamily::Name(SEMIBOLD_NAME.into())
}

/// Puts Inter Regular first in the proportional family and Inter SemiBold first in its own
/// ([`semibold`]), which egui's Heading style then draws in, at the size it had. egui's
/// built-in fonts stay behind each cut, for a character outside it, and the monospace family
/// is not touched.
pub fn install(ctx: &egui::Context) {
    let mut definitions = egui::FontDefinitions::default();
    let built_in = definitions.families[&egui::FontFamily::Proportional].clone();
    for (name, bytes, family) in [
        (INTER_NAME, INTER, egui::FontFamily::Proportional),
        (SEMIBOLD_NAME, INTER_SEMIBOLD, semibold()),
    ] {
        definitions.font_data.insert(
            name.to_owned(),
            std::sync::Arc::new(egui::FontData::from_static(bytes)),
        );
        let mut fonts = vec![name.to_owned()];
        fonts.extend(built_in.iter().cloned());
        definitions.families.insert(family, fonts);
    }
    ctx.set_fonts(definitions);
    ctx.all_styles_mut(|style| {
        let heading = egui::TextStyle::Heading;
        let size = style.text_styles[&heading].size;
        style
            .text_styles
            .insert(heading, egui::FontId::new(size, semibold()));
    });
}

/// A context whose only font is the cut.
#[cfg(test)]
fn inter_alone() -> egui::Context {
    alone(INTER_NAME, INTER)
}

/// A context whose only font is `bytes`, named `name`.
#[cfg(test)]
fn alone(name: &str, bytes: &'static [u8]) -> egui::Context {
    let mut definitions = egui::FontDefinitions::empty();
    definitions.font_data.insert(
        name.to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(bytes)),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        definitions.families.insert(family, vec![name.to_owned()]);
    }
    let ctx = egui::Context::default();
    ctx.set_fonts(definitions);
    // egui loads fonts on a context's first frame.
    let _ = ctx.run(egui::RawInput::default(), |_| {});
    ctx
}

/// Whether the cut itself holds `c`, whatever egui's built-in fonts hold: the panel's glyph
/// guards ask it of every text (`gui::test_support::assert_glyphs`). The font is loaded once
/// per test thread. egui reports a font's own replacement character as missing; with the cut
/// alone that is the question mark, which the cut has.
#[cfg(test)]
pub(crate) fn cut_holds(c: char) -> bool {
    thread_local! {
        static ALONE: egui::Context = inter_alone();
    }
    c == '?' || ALONE.with(|ctx| ctx.fonts(|f| f.has_glyph(&egui::FontId::proportional(14.0), c)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{FontFamily, FontId};

    /// A context with the font installed and loaded.
    fn installed() -> egui::Context {
        let ctx = egui::Context::default();
        install(&ctx);
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        ctx
    }

    #[test]
    fn inter_leads_the_proportional_family_and_the_built_in_fonts_follow() {
        let definitions = installed().fonts(|f| f.lock().fonts.definitions().clone());
        let built_in = egui::FontDefinitions::default();
        let mut proportional = vec![INTER_NAME.to_owned()];
        proportional.extend(built_in.families[&FontFamily::Proportional].iter().cloned());
        assert_eq!(
            definitions.families[&FontFamily::Proportional],
            proportional
        );
        assert_eq!(
            definitions.families[&FontFamily::Monospace],
            built_in.families[&FontFamily::Monospace]
        );
        // The SemiBold cut leads its own family, the built-in fonts behind it, and egui's
        // Heading style draws in it at the size it had; no other style changes.
        proportional[0] = SEMIBOLD_NAME.to_owned();
        assert_eq!(definitions.families[&semibold()], proportional);
        let ctx = installed();
        let plain = egui::Style::default();
        for (style, font) in &ctx.style().text_styles {
            let mut want = plain.text_styles[style].clone();
            if *style == egui::TextStyle::Heading {
                want.family = semibold();
            }
            assert_eq!(*font, want, "{style:?}");
        }
    }

    #[test]
    fn a_character_outside_the_cut_is_drawn_by_a_built_in_font() {
        // The fallback by a character, not by the list of names (the whole-branch review,
        // R-7): a part name in Cyrillic, and an emoji egui carries. With the cut alone neither
        // has a glyph, so a change that replaced the built-in fonts would fail here.
        let (ctx, alone) = (installed(), inter_alone());
        let font = FontId::proportional(14.0);
        for c in ['\u{0416}', '\u{1F44D}'] {
            assert!(ctx.fonts(|f| f.has_glyph(&font, c)), "{c:?}");
            assert!(!alone.fonts(|f| f.has_glyph(&font, c)), "{c:?}");
        }
    }

    #[test]
    fn the_ten_digits_have_one_advance_width() {
        // UX overhaul spec 8.1, test 7, at the sizes the app draws numbers in (small text, the
        // drawing's tags, body text, a term, a heading and the open equation), and at one, one
        // and a half and two pixels per point: a zoomed browser, a dense display.
        for pixels_per_point in [1.0, 1.5, 2.0] {
            let (ctx, alone) = (installed(), inter_alone());
            for ctx in [&ctx, &alone] {
                ctx.set_pixels_per_point(pixels_per_point);
                let _ = ctx.run(egui::RawInput::default(), |_| {});
            }
            for size in [9.0, 11.0, 12.0, 12.5, 15.0, 18.0, 22.0] {
                let font = FontId::proportional(size);
                let widths = |ctx: &egui::Context| -> Vec<f32> {
                    "0123456789"
                        .chars()
                        .map(|digit| ctx.fonts(|f| f.glyph_width(&font, digit)))
                        .collect()
                };
                let drawn = widths(&ctx);
                assert!(
                    drawn.iter().all(|&w| w > 0.0 && w == drawn[0]),
                    "{pixels_per_point} {size}: {drawn:?}"
                );
                // They are the cut's digits, not a built-in font's.
                assert_eq!(drawn, widths(&alone), "{pixels_per_point} {size}");
            }
        }
    }

    #[test]
    fn a_row_is_as_tall_as_the_built_in_font_s() {
        // The panel's layout was built on egui's built-in font, whose rows are 1.149 em.
        // Inter's own are 1.21 em, 0.76 of a point more per row at body size, which cost the
        // first view's drawing a sixth of its height at 1280 x 800 (the whole-branch review,
        // E-3). The cut carries the built-in font's ascender, descender and line gap
        // (`subset_inter.py`). egui rounds each of the three to 1/32 point, so the two heights
        // can differ by a rounding step or two.
        let ctx = installed();
        let built_in = egui::Context::default();
        let _ = built_in.run(egui::RawInput::default(), |_| {});
        for size in [9.0, 11.0, 12.0, 12.5, 14.0, 15.0, 18.0, 22.0] {
            let font = FontId::proportional(size);
            let ours = ctx.fonts(|f| f.row_height(&font));
            let theirs = built_in.fonts(|f| f.row_height(&font));
            assert!(
                (ours - theirs).abs() <= 2.0 / 32.0,
                "{size}: {ours} {theirs}"
            );
        }
    }

    #[test]
    fn the_cut_holds_the_characters_the_app_draws() {
        // Printable ASCII, and every other character of the app's texts that Inter has (the
        // list the cut was made for, `assets/fonts/subset_inter.py`).
        let ascii = (0x21u8..0x7f).map(char::from);
        let others = "§¯°±²³µ·×Øé÷ΔΣΩαβδεηθλμνπρστφωϑ–—…•⁴⁵⁶⁻→−√∞∫≈≠≤≥".chars();
        let missing: String = ascii.chain(others).filter(|&c| !cut_holds(c)).collect();
        assert_eq!(missing, "", "not in the cut");
        // The check can fail: Inter has no "element of" sign, and the cut leaves Cyrillic out.
        assert!(!cut_holds('\u{2208}') && !cut_holds('\u{0416}'));
    }

    #[test]
    fn the_semibold_cut_has_the_regular_cut_s_characters_digits_and_rows() {
        // The title and the headings are drawn in the SemiBold cut (the stage 1 review, S-1),
        // and the panel's glyph guards ask the Regular cut (`cut_holds`): they hold for the
        // headings too because the two cuts hold the same characters (the script's one list,
        // read here over every character up to the maths signs). Its digits are tabular and
        // its rows are as tall as the Regular cut's, so a heading's row and the brand mark
        // beside the title (`gui::look::mark_ui`) do not move. It is heavier, so wider.
        let (regular, semibold) = (inter_alone(), alone(SEMIBOLD_NAME, INTER_SEMIBOLD));
        let font = FontId::proportional(14.0);
        let held = |ctx: &egui::Context, c: char| ctx.fonts(|f| f.has_glyph(&font, c));
        let differ: String = (0x20..0x2300)
            .filter_map(char::from_u32)
            .filter(|&c| held(&regular, c) != held(&semibold, c))
            .collect();
        assert_eq!(differ, "", "held by one cut and not the other");
        assert!(held(&semibold, '\u{03A9}') && !held(&semibold, '\u{0416}'));
        for size in [12.5, 15.0, 18.0, 22.0] {
            let font = FontId::proportional(size);
            let widths: Vec<f32> = "0123456789"
                .chars()
                .map(|digit| semibold.fonts(|f| f.glyph_width(&font, digit)))
                .collect();
            assert!(
                widths.iter().all(|&w| w > 0.0 && w == widths[0]),
                "{size}: {widths:?}"
            );
            let row = |ctx: &egui::Context| ctx.fonts(|f| f.row_height(&font));
            assert_eq!(row(&semibold), row(&regular), "{size}");
        }
        let width = |ctx: &egui::Context| {
            let heading = FontId::proportional(18.0);
            let words = "Size and torque".to_owned();
            ctx.fonts(|f| f.layout_no_wrap(words, heading, egui::Color32::WHITE))
                .size()
                .x
        };
        assert!(width(&semibold) > width(&regular));
    }
}
