//! Display text of engine values, shared by every readout of the panel.

use crate::engine::meta::Value;

/// Significant digits of a displayed number.
pub const SIGNIFICANT_DIGITS: i32 = 4;

/// Display text of a value: numbers to [`SIGNIFICANT_DIGITS`] significant
/// digits, integers and text as they are.
///
/// Results can be `+inf` (E13) or NaN (E20, a positive beta without a rating),
/// so non-finite numbers get their own text: `+inf`, `-inf`, `NaN`. Python's
/// `None` (a result path the engine does not know, or a value not entered)
/// shows as an em dash.
pub fn format_value(value: &Value) -> String {
    match value {
        Value::Num(x) => format_number(*x),
        Value::Int(i) => i.to_string(),
        Value::Text(text) => text.clone(),
        Value::None => "\u{2014}".to_owned(),
    }
}

/// `text` followed by the unit as the screen writes it (`display_unit`), unless the unit is
/// empty or `-` (dimensionless).
pub fn with_unit(text: String, unit: &str) -> String {
    let unit = display_unit(unit);
    if unit.is_empty() || unit == "-" {
        text
    } else {
        format!("{text} {unit}")
    }
}

/// A unit as the screen writes it: the sweep tables' workbook header "Nm" reads "N·m", as every
/// other torque's unit does (spec 3.5). The metadata and the exports keep "Nm":
/// `tests/schema.rs` holds the sweep columns to the workbook headers.
fn display_unit(unit: &str) -> &str {
    if unit == "Nm" { "N·m" } else { unit }
}

/// What a search box matches an input or a result by: its display label (`gui::plain_text`),
/// its catalogue label when the overlay renames it, its path and its workbook cell, one per
/// line, lowercase. The results search and the inputs filter share it, so both match alike, and
/// an old label, a path or a cell still finds a renamed input (spec 3.5).
pub(crate) fn search_haystack(
    display: &str,
    label: &str,
    path: &str,
    cell: Option<&str>,
) -> String {
    let names = if display == label {
        label.to_owned()
    } else {
        format!("{display}\n{label}")
    };
    format!("{names}\n{path}\n{}", cell.unwrap_or("")).to_lowercase()
}

/// A search box's text as it is matched against a [`search_haystack`]: lowercase, without the
/// surrounding blanks. An empty needle matches everything.
pub(crate) fn search_needle(query: &str) -> String {
    query.trim().to_lowercase()
}

/// The text of a number that is not finite: `+inf`, `-inf` or `NaN`; `None` for a finite
/// number. The display and both results exports share it, so a non-finite value reads the
/// same everywhere (JSON has no infinity or NaN: the exports write this text, decision M41-15).
pub(crate) fn non_finite_text(x: f64) -> Option<&'static str> {
    if x.is_nan() {
        Some("NaN")
    } else if x.is_infinite() {
        Some(if x > 0.0 { "+inf" } else { "-inf" })
    } else {
        None
    }
}

/// A number to `decimals` places: the Size tab's lines, whose precision follows their sliders
/// (0.1 mm) rather than [`SIGNIFICANT_DIGITS`]. A number that is not finite reads as its
/// [`non_finite_text`].
pub(crate) fn fixed(x: f64, decimals: usize) -> String {
    match non_finite_text(x) {
        Some(text) => text.to_owned(),
        None => format!("{x:.decimals$}"),
    }
}

/// The most decimals [`fewest_decimals_apart`] gives a number near one: f64 holds no more of
/// its digits.
const MOST_DECIMALS: usize = 15;

/// The one search for more decimals (BL-059 NB-1): of the guidance's band and window quotes
/// (`gui::guidance`'s `band_quotes` and `window_quotes`), its quote of an input typed below its
/// slider's bottom, and the Size tab's change decimals ([`change_decimals`]). The fewest
/// decimals, from `from` up to [`MOST_DECIMALS`], whose `quotes` read `apart`, with those
/// quotes; else [`MOST_DECIMALS`] and its quotes.
pub(crate) fn fewest_decimals_apart<T>(
    from: usize,
    quotes: impl Fn(usize) -> T,
    apart: impl Fn(&T) -> bool,
) -> (usize, T) {
    (from..=MOST_DECIMALS)
        .map(|decimals| (decimals, quotes(decimals)))
        .find(|(_, quoted)| apart(quoted))
        .unwrap_or_else(|| (MOST_DECIMALS, quotes(MOST_DECIMALS)))
}

/// How many decimals past its own a change may take to show a digit that is not zero
/// ([`change_decimals`]; slice 3 spec T-9). A change still zero there is float noise, such as
/// Auto's solve residual, and reads its own decimals with its true sign. The torque card draws
/// a size or margin change only where it shows a digit within them.
pub(crate) const CHANGE_EXTRA_DECIMALS: usize = 2;

/// The decimals a torque change starts from on the Size tab (spec section 5: N·m to two
/// decimals, then [`change_decimals`]): the readout's gap (`torque_gap_text`) and the torque
/// card's N·m read this one constant.
pub(crate) const TORQUE_DECIMALS: usize = 2;

/// Whether `text` reads a finite number that is not zero: "-0.005" does; "-0.00", "NaN" and
/// "+inf" do not. What [`change_decimals`] looks for, decimal by decimal.
pub(crate) fn reads_not_zero(text: &str) -> bool {
    text.parse::<f64>().is_ok_and(|x| x.is_finite() && x != 0.0)
}

/// The decimals of a change `x` (decision 10B): the fewest, from `from` up to `from` +
/// [`CHANGE_EXTRA_DECIMALS`], at which [`fixed`] reads a digit that is not zero
/// ([`fewest_decimals_apart`]; -0.0046 from 2 takes 3, "-0.005"). `from` where `x` is zero or
/// not finite, or reads zero even at the cap (T-9: -0.00004 from 2 reads "-0.00").
pub(crate) fn change_decimals(x: f64, from: usize) -> usize {
    if x == 0.0 || !x.is_finite() {
        return from;
    }
    let (decimals, text) = fewest_decimals_apart(
        from,
        |decimals| fixed(x, decimals),
        |text| reads_not_zero(text),
    );
    // The search has no cap of its own: it falls back to MOST_DECIMALS when no decimal reads
    // a digit, or when `from` is past MOST_DECIMALS. Its answer counts only inside the cap,
    // and only where it reads a digit.
    if (from..=from + CHANGE_EXTRA_DECIMALS).contains(&decimals) && reads_not_zero(&text) {
        decimals
    } else {
        from
    }
}

/// `x` to exactly `decimals` places with its true sign, as the Size tab writes a change: a
/// value that rounds to zero keeps its sign ("-0.00"), an exact zero of either sign reads "+"
/// (the house rule; `{:+}` alone prints -0.0 as "-0"), and a number that is not finite reads as
/// its [`non_finite_text`].
pub(crate) fn signed_fixed(x: f64, decimals: usize) -> String {
    match non_finite_text(x) {
        Some(text) => text.to_owned(),
        None if x == 0.0 => format!("{:+.decimals$}", 0.0),
        None => format!("{x:+.decimals$}"),
    }
}

/// A change `x` at its own decimals: [`signed_fixed`] at [`change_decimals`] from `from` (the
/// Size tab's N·m from [`TORQUE_DECIMALS`], a size change in mm from 1).
pub(crate) fn signed_change(x: f64, from: usize) -> String {
    signed_fixed(x, change_decimals(x, from))
}

/// The decimals a row of changes shares (slice 3 spec T-14: a torque card row's two N·m at one
/// precision): the larger of their [`change_decimals`] from `from`, so each keeps its true sign
/// and a digit that is not zero, and the two read in their true ratio (-0.0063 and -0.0046 N·m
/// read "-0.006" and "-0.005", not "-0.01" and "-0.005"). `None` for a row with no number.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the torque card's lines call it (slice 3 Task 4, which removes this)"
    )
)]
pub(crate) fn shared_change_decimals(changes: &[f64], from: usize) -> Option<usize> {
    changes.iter().map(|&x| change_decimals(x, from)).max()
}

/// A change, as a fraction, as a signed percentage to one decimal with its unit (decision 10B:
/// the Size tab's one percent rule): 0.0349 reads `+3.5 %`, -0.2679 `-26.8 %`. One that rounds
/// to zero keeps its true sign (`-0.0 %`), an exact zero of either sign reads `+0.0 %`, and one
/// that is not finite reads as its [`non_finite_text`] and the unit ([`signed_fixed`]).
pub(crate) fn signed_percent(fraction: f64) -> String {
    format!("{} %", signed_fixed(fraction * 100.0, 1))
}

/// The place value of the [`SIGNIFICANT_DIGITS`]-th significant digit of a finite positive `x`,
/// its last shown digit unless rounding carries into a new one (9.9996 shows as 10.00): 0.001
/// for 1.703, 0.0001 for 0.8806. A number rounded up or down at its last shown digit moves by
/// it.
pub(crate) fn last_digit_step(x: f64) -> f64 {
    10f64.powi(x.log10().floor() as i32 - (SIGNIFICANT_DIGITS - 1))
}

/// A number to [`SIGNIFICANT_DIGITS`] significant digits: fixed point when it
/// rounds to at least 1e-3 and below 1e6, scientific outside; zero (either
/// sign) as `0`.
fn format_number(x: f64) -> String {
    if let Some(text) = non_finite_text(x) {
        return text.to_owned();
    }
    if x == 0.0 {
        return "0".to_owned();
    }
    // Round once, in scientific form: its exponent is the decimal magnitude
    // after rounding (9.99996 gives "1.000e1"), exact, with no log10.
    let scientific = format!("{:.*e}", (SIGNIFICANT_DIGITS - 1) as usize, x);
    let exponent: i32 = match scientific.split_once('e').map(|(_, e)| e.parse()) {
        Some(Ok(exponent)) => exponent,
        _ => return scientific,
    };
    if !(-3..6).contains(&exponent) {
        return scientific;
    }
    let rounded: f64 = scientific.parse().unwrap_or(x);
    let decimals = (SIGNIFICANT_DIGITS - 1 - exponent).max(0) as usize;
    format!("{rounded:.decimals$}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(x: f64) -> String {
        format_value(&Value::Num(x))
    }

    #[test]
    fn a_search_reads_label_path_and_cell_in_lowercase_and_a_needle_drops_the_blanks() {
        assert_eq!(
            search_haystack(
                "Gearbox ratio",
                "Gearbox ratio",
                "coupling.gear_ratio",
                Some("Calculator!C45")
            ),
            "gearbox ratio\ncoupling.gear_ratio\ncalculator!c45"
        );
        // A display label the overlay gives (gui::plain_text) comes first; the catalogue label
        // stays searchable after it.
        assert_eq!(
            search_haystack(
                "Magnet face gap",
                "Candidate flat-face magnetic gap",
                "metal.face_gap_mm",
                Some("Metal design!C119")
            ),
            "magnet face gap\ncandidate flat-face magnetic gap\nmetal.face_gap_mm\nmetal design!c119"
        );
        // A Rust-only input or result has no cell.
        assert_eq!(
            search_haystack(
                "Highest odd harmonic summed",
                "Highest odd harmonic summed",
                "coupling.max_harmonic",
                None
            ),
            "highest odd harmonic summed\ncoupling.max_harmonic\n"
        );
        assert_eq!(search_needle("  GEARBOX Ratio "), "gearbox ratio");
        assert_eq!(search_needle("   "), "");
    }

    #[test]
    fn numbers_show_four_significant_digits() {
        assert_eq!(num(2.68812345), "2.688");
        assert_eq!(num(96.8), "96.80");
        assert_eq!(num(93.0612), "93.06");
        assert_eq!(num(-0.10312), "-0.1031");
        assert_eq!(num(1234.56), "1235");
        assert_eq!(num(123456.7), "123500");
        assert_eq!(num(0.0012345), "0.001234");
    }

    #[test]
    fn very_large_and_very_small_numbers_switch_to_scientific() {
        assert_eq!(num(2.0e7), "2.000e7");
        assert_eq!(num(1_000_000.0), "1.000e6");
        assert_eq!(num(4.14e-5), "4.140e-5");
        assert_eq!(num(-6.837e-6), "-6.837e-6");
    }

    #[test]
    fn rounding_that_carries_into_a_new_digit_keeps_four_digits() {
        assert_eq!(num(9.99996), "10.00");
        assert_eq!(num(-99.996), "-100.0");
        assert_eq!(num(0.0099996), "0.01000");
        assert_eq!(num(999_999.7), "1.000e6");
    }

    #[test]
    fn the_last_digit_step_is_the_fourth_significant_digit_s_place() {
        // 9.9996 shows as 10.00, but rounded down at its fourth digit it is 9.999.
        for (x, step) in [
            (1.703, 0.001),
            (0.8806, 0.0001),
            (9.9996, 0.001),
            (1234.0, 1.0),
            (0.001, 1e-6),
        ] {
            let got = last_digit_step(x);
            assert!((got - step).abs() <= step * 1e-12, "{x}: {got}");
        }
    }

    #[test]
    fn zero_of_either_sign_is_plain_zero() {
        assert_eq!(num(0.0), "0");
        assert_eq!(num(-0.0), "0");
    }

    #[test]
    fn non_finite_results_have_their_own_text() {
        assert_eq!(num(f64::INFINITY), "+inf");
        assert_eq!(num(f64::NEG_INFINITY), "-inf");
        assert_eq!(num(f64::NAN), "NaN");
    }

    #[test]
    fn only_non_finite_numbers_have_a_non_finite_text() {
        assert_eq!(non_finite_text(f64::INFINITY), Some("+inf"));
        assert_eq!(non_finite_text(f64::NEG_INFINITY), Some("-inf"));
        assert_eq!(non_finite_text(f64::NAN), Some("NaN"));
        assert_eq!(non_finite_text(-f64::NAN), Some("NaN"));
        for finite in [0.0, -0.0, f64::MAX, f64::MIN, f64::MIN_POSITIVE, 2.5] {
            assert_eq!(non_finite_text(finite), None, "{finite}");
        }
    }

    #[test]
    fn integers_text_and_none_show_as_they_are() {
        assert_eq!(format_value(&Value::Int(10)), "10");
        assert_eq!(format_value(&Value::Int(-3)), "-3");
        assert_eq!(
            format_value(&Value::Text("ISO 4762 M4 x 14".into())),
            "ISO 4762 M4 x 14"
        );
        assert_eq!(format_value(&Value::Text(String::new())), "");
        assert_eq!(format_value(&Value::None), "\u{2014}");
    }

    #[test]
    fn units_follow_the_value_except_dimensionless() {
        assert_eq!(with_unit("2.688".into(), "N·m"), "2.688 N·m");
        assert_eq!(with_unit("10".into(), "-"), "10");
        assert_eq!(with_unit("OK".into(), ""), "OK");
    }

    #[test]
    fn the_sweep_tables_torque_unit_reads_as_every_other_torque_does() {
        // The sweep columns keep the workbook header "Nm" (tests/schema.rs
        // table_columns_match_the_workbook_headers holds them to it; the exports write it);
        // the screen writes N·m, as for every other torque (spec 3.5).
        assert_eq!(with_unit("2.688".into(), "Nm"), "2.688 N·m");
        let sweep =
            crate::engine::meta::result_rows(&crate::compute_all(&crate::DesignInputs::default()))
                .into_iter()
                .find(|row| row.path == "gap_sweep[0].pullout_op_Nm")
                .expect("the gap sweep's pull-out column");
        assert_eq!(
            sweep.meta.unit, "Nm",
            "the metadata keeps the workbook header"
        );
        assert_eq!(with_unit("1".into(), sweep.meta.unit), "1 N·m");
        // Every other unit is written as the metadata has it.
        assert_eq!(with_unit("2.688".into(), "N·m"), "2.688 N·m");
        assert_eq!(with_unit("3".into(), "mm"), "3 mm");
    }

    #[test]
    fn fixed_decimals_follow_the_request_and_non_finite_numbers_keep_their_text() {
        assert_eq!(fixed(42.8, 1), "42.8");
        assert_eq!(fixed(42.44, 1), "42.4");
        assert_eq!(fixed(22.0, 1), "22.0");
        assert_eq!(fixed(1.96836, 2), "1.97");
        assert_eq!(fixed(f64::INFINITY, 1), "+inf");
        assert_eq!(fixed(f64::NAN, 2), "NaN");
    }

    #[test]
    fn a_change_reads_as_a_signed_percentage_to_a_tenth_with_its_true_sign() {
        // Decision 10B (slice 3 spec section 5): one decimal of percent, the Size tab's one
        // percent rule, on the torque card and the readout's gap alike.
        assert_eq!(signed_percent(0.0349), "+3.5 %");
        assert_eq!(signed_percent(-0.0037), "-0.4 %");
        assert_eq!(signed_percent(-0.0028), "-0.3 %");
        assert_eq!(signed_percent(0.1121), "+11.2 %");
        assert_eq!(signed_percent(-0.2679), "-26.8 %");
        assert_eq!(signed_percent(1.1279), "+112.8 %");
        // A change that rounds to zero keeps its true sign, so a small fall never reads as a
        // rise ...
        assert_eq!(signed_percent(-0.0004), "-0.0 %");
        assert_eq!(signed_percent(0.0004), "+0.0 %");
        // ... while an exact zero of either sign reads "+" (the house rule; `{:+.1}` alone
        // prints -0.0 as "-0.0").
        assert_eq!(signed_percent(0.0), "+0.0 %");
        assert_eq!(signed_percent(-0.0), "+0.0 %");
        assert_eq!(signed_percent(f64::NAN), "NaN %");
        assert_eq!(signed_percent(f64::INFINITY), "+inf %");
        assert_eq!(signed_percent(f64::NEG_INFINITY), "-inf %");
    }

    #[test]
    fn a_change_keeps_its_sign_and_shows_a_digit_that_is_not_zero() {
        // Decision 10B: a change to the fewest decimals, from its own (N·m 2), that keep its
        // true sign and read a digit that is not zero, never "+0.00" for a fall or "-0.00" for
        // a rise; at most CHANGE_EXTRA_DECIMALS more (T-9), so float noise reads its first
        // decimals with its true sign.
        assert_eq!(CHANGE_EXTRA_DECIMALS, 2);
        for (x, text) in [
            (0.2010, "+0.20"),
            (-0.0063, "-0.01"),
            (-0.0046, "-0.005"),
            (-0.00012, "-0.0001"),
            (0.0, "+0.00"),
            (-0.0, "+0.00"),
            (-0.00004, "-0.00"),
            (1e-10, "+0.00"),
            (f64::NAN, "NaN"),
            (f64::INFINITY, "+inf"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(signed_change(x, 2), text, "{x}");
        }
        // The cap: where the search reads a digit only past `from` + CHANGE_EXTRA_DECIMALS
        // (-0.00004 at 5 decimals, 1e-10 at 10, 1e-20 at none up to MOST_DECIMALS, where it
        // falls back to MOST_DECIMALS), the change reads `from`.
        for (x, past_the_cap) in [(-0.00004, 5), (1e-10, 10), (1e-20, MOST_DECIMALS)] {
            let (found, _) = fewest_decimals_apart(2, |d| fixed(x, d), |t| reads_not_zero(t));
            assert_eq!(found, past_the_cap, "{x}");
            assert_eq!(change_decimals(x, 2), 2, "{x}");
        }
        assert_eq!(change_decimals(-0.0063, 2), 2);
        assert_eq!(change_decimals(-0.0046, 2), 3);
        assert_eq!(change_decimals(-0.00012, 2), 4, "at the cap itself");
        // A size change starts at one decimal (the sliders' 0.1 mm): 0.0046 mm needs three,
        // 6e-5 mm is past the cap there.
        assert_eq!(change_decimals(0.0046, 1), 3);
        assert_eq!(change_decimals(6e-5, 1), 1);
        // Zero and numbers that are not finite read `from`, whatever `from` is; a `from` past
        // what the search covers reads itself, never fewer decimals.
        for x in [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(change_decimals(x, 1), 1, "{x}");
        }
        assert_eq!(change_decimals(0.5, MOST_DECIMALS + 1), MOST_DECIMALS + 1);
        assert_eq!(change_decimals(1e-17, MOST_DECIMALS - 1), MOST_DECIMALS - 1);
        // signed_fixed: exactly `decimals` places with the true sign; an exact zero reads "+".
        assert_eq!(signed_fixed(-0.0063, 3), "-0.006");
        assert_eq!(signed_fixed(-0.0046, 3), "-0.005");
        assert_eq!(signed_fixed(-0.00004, 2), "-0.00");
        assert_eq!(signed_fixed(0.0, 3), "+0.000");
        assert_eq!(signed_fixed(-0.0, 3), "+0.000");
        assert_eq!(signed_fixed(f64::NAN, 2), "NaN");
        assert_eq!(signed_fixed(f64::NEG_INFINITY, 2), "-inf");
        // A row's two N·m at one precision (T-14): the larger of the sides' decimals. (e)'s
        // Outside diameter row, -0.0063 and -0.0046 N·m, reads "-0.006" and "-0.005" (each on
        // its own: "-0.01" and "-0.005"); (e2)+Auto length's, -0.00477 and -0.00502, reads
        // "-0.005" both. A row with one number side keeps that side's own; a row of words has
        // none.
        assert_eq!(shared_change_decimals(&[-0.0063, -0.0046], 2), Some(3));
        assert_eq!(shared_change_decimals(&[-0.00477, -0.00502], 2), Some(3));
        assert_eq!(signed_fixed(-0.00477, 3), signed_fixed(-0.00502, 3));
        assert_eq!(shared_change_decimals(&[-0.0046], 2), Some(3));
        assert_eq!(shared_change_decimals(&[0.2008], 2), Some(2));
        assert_eq!(shared_change_decimals(&[], 2), None);
    }

    #[test]
    fn the_decimals_search_takes_the_fewest_from_its_start_else_the_most() {
        // The one search for more decimals (BL-059 NB-1; the guidance's band and window quotes,
        // a typed input below its slider's bottom, and the Size tab's change decimals).
        let digit = |text: &String| reads_not_zero(text);
        assert_eq!(
            fewest_decimals_apart(2, |d| fixed(-0.0046, d), digit),
            (3, "-0.005".to_owned())
        );
        // Never fewer than its start, even where fewer would read apart.
        assert_eq!(
            fewest_decimals_apart(2, |d| fixed(0.5, d), digit),
            (2, "0.50".to_owned())
        );
        // None up to MOST_DECIMALS: MOST_DECIMALS and its quote.
        assert_eq!(
            fewest_decimals_apart(2, |d| fixed(1e-20, d), digit),
            (MOST_DECIMALS, fixed(1e-20, MOST_DECIMALS))
        );
        // What reads apart here: a finite number that is not zero, of either sign.
        for (text, apart) in [
            ("-0.005", true),
            ("0.50", true),
            ("-0.00", false),
            ("0.000", false),
            ("NaN", false),
            ("+inf", false),
            ("", false),
        ] {
            assert_eq!(reads_not_zero(text), apart, "{text:?}");
        }
    }
}
