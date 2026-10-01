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

/// `text` followed by the unit, unless the unit is empty or `-` (dimensionless).
pub fn with_unit(text: String, unit: &str) -> String {
    if unit.is_empty() || unit == "-" {
        text
    } else {
        format!("{text} {unit}")
    }
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
}
