//! Python and Excel semantics the port reproduces exactly.
//!
//! The Python engine transcribes the workbook. These helpers cover the places
//! where Python (or Excel, through Python) behaves differently from the obvious
//! Rust. Each is checked against the Python original on a generated corpus
//! (`tests/data/differential/helpers.json`, checked by `tests/differential.rs`).
//!
//! Where Python raises (division by zero, `math.log` of a non-positive number,
//! `int()` of NaN or an infinity), Rust arithmetic returns NaN or an infinity
//! instead. The helpers never panic: text helpers return an Excel-style error
//! string for inputs Python rejects. Slider ranges keep the engine inside
//! Python's domain, and the differential generator fails loudly if Python
//! raises on any generated case.

use super::meta::Value;

/// Relative tolerance of the parity rule (`tests/test_parity.py`).
pub const PARITY_REL_TOL: f64 = 1e-9;
/// Absolute tolerance of the parity rule (`tests/test_parity.py`).
pub const PARITY_ABS_TOL: f64 = 1e-12;

/// Python `min(a, b)`: `b` if `b < a`, else `a`. Unlike `f64::min`, NaN and
/// signed-zero handling follow Python (the result depends on argument order).
/// For three or more arguments nest the calls left to right, as Python folds.
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Python `max(a, b)`: `b` if `b > a`, else `a`. See [`py_min`].
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Excel CEILING for positive numbers, as `_fields.ceiling`: round `x` up to a
/// multiple of `significance`, with Python's 1e-12 guard on the quotient.
pub fn ceiling(x: f64, significance: f64) -> f64 {
    let q = x / significance;
    (q - 1e-12).ceil() * significance
}

/// Excel FLOOR for positive numbers, as `_fields.floor_`.
pub fn floor_(x: f64, significance: f64) -> f64 {
    let q = x / significance;
    (q + 1e-12).floor() * significance
}

/// Excel `TEXT(x, "0")` as `temperature._text0`: round half away from zero and
/// print the integer, with no negative zero (`-0.3` gives `"0"`).
///
/// Python raises for NaN and infinities; this returns `"#NUM!"`.
pub fn text0(x: f64) -> String {
    if !x.is_finite() {
        return "#NUM!".to_owned();
    }
    let magnitude = (x.abs() + 0.5).floor();
    if magnitude == 0.0 {
        "0".to_owned()
    } else if x >= 0.0 {
        format!("{magnitude:.0}")
    } else {
        format!("-{magnitude:.0}")
    }
}

/// Python `repr(x)` (and `str(x)`) of a float: the shortest digits that round
/// trip, in fixed notation for decimal exponents -4..=15 (`"0.0001"`,
/// `"1000000000000000.0"`) and scientific otherwise (`"1e-05"`, `"1e+16"`).
pub fn py_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    // Rust's `{:e}` gives the same shortest round-trip digits: "-1.2345e-7".
    let sci = format!("{x:e}");
    let (mantissa, exponent) = sci
        .split_once('e')
        .expect("LowerExp output always has an exponent");
    let exponent: i32 = exponent.parse().expect("LowerExp exponent is an integer");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();

    if !(-4..16).contains(&exponent) {
        let (lead, rest) = digits.split_at(1);
        let fraction = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let exp_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{lead}{fraction}e{exp_sign}{:02}", exponent.abs())
    } else if exponent >= 0 {
        let point = exponent as usize + 1;
        if digits.len() <= point {
            format!("{sign}{digits}{}.0", "0".repeat(point - digits.len()))
        } else {
            let (int_part, frac_part) = digits.split_at(point);
            format!("{sign}{int_part}.{frac_part}")
        }
    } else {
        let zeros = "0".repeat((-exponent - 1) as usize);
        format!("{sign}0.{zeros}{digits}")
    }
}

/// `clamps._fmt_num`: Excel-style number to text in a concatenation. Whole
/// numbers print without a decimal point (`12.0` gives `"12"`, `-0.0` gives
/// `"0"`); anything else prints as Python `repr`.
pub fn fmt_num(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 {
        if x == 0.0 {
            "0".to_owned()
        } else {
            format!("{x:.0}")
        }
    } else {
        py_repr(x)
    }
}

/// Python `f"{x:.{decimals}f}"`: correctly rounded, ties to even on the exact
/// binary value (`0.125` to 2 places gives `"0.12"`), `"-0.00"` kept for tiny
/// negatives, and `"nan"`/`"inf"`/`"-inf"` for non-finite values.
pub fn fmt_fixed(x: f64, decimals: usize) -> String {
    if x.is_nan() {
        // Rust prints "NaN"; Python prints "nan" whatever the sign bit.
        return "nan".to_owned();
    }
    format!("{x:.decimals$}")
}

/// Python `str()` of a dynamic value, as the parity rule applies it to text.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::Num(x) => py_repr(*x),
        Value::Int(i) => i.to_string(),
        Value::Text(s) => s.clone(),
        Value::None => "None".to_owned(),
    }
}

/// Python `math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)`.
pub fn is_close(a: f64, b: f64, rel_tol: f64, abs_tol: f64) -> bool {
    if a == b {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    let diff = (b - a).abs();
    diff <= (rel_tol * b).abs() || diff <= (rel_tol * a).abs() || diff <= abs_tol
}

/// The parity rule of `tests/test_parity.py::_close`: if either side is text,
/// compare Python `str()` exactly; if either is `None`, both must be; otherwise
/// numbers agree to 1e-9 relative or 1e-12 absolute.
pub fn parity_close(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Text(_), _) | (_, Value::Text(_)) => py_str(a) == py_str(b),
        (Value::None, _) | (_, Value::None) => matches!((a, b), (Value::None, Value::None)),
        _ => is_close(as_f64(a), as_f64(b), PARITY_REL_TOL, PARITY_ABS_TOL),
    }
}

fn as_f64(value: &Value) -> f64 {
    match value {
        Value::Num(x) => *x,
        Value::Int(i) => *i as f64,
        Value::Text(_) | Value::None => unreachable!("parity_close handles text and None first"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_min_max_keep_python_argument_order() {
        assert_eq!(py_min(1.0, 2.0), 1.0);
        assert_eq!(py_min(2.0, 1.0), 1.0);
        assert_eq!(py_min(1.0, f64::NAN), 1.0);
        assert!(py_min(f64::NAN, 1.0).is_nan());
        assert_eq!(py_max(1.0, 2.0), 2.0);
        assert_eq!(py_max(1.0, f64::NAN), 1.0);
        assert!(py_max(f64::NAN, 1.0).is_nan());
        assert!(
            py_min(0.0, -0.0).is_sign_positive(),
            "a tie keeps the first argument"
        );
    }

    #[test]
    fn ceiling_and_floor_follow_excel_with_the_python_guard() {
        assert_eq!(ceiling(13.3, 2.0), 14.0);
        assert_eq!(ceiling(12.0, 2.0), 12.0);
        assert_eq!(ceiling(1.90415, 0.1), 2.0);
        assert_eq!(
            ceiling(0.3, 0.1),
            0.30000000000000004,
            "same float as Python"
        );
        assert_eq!(ceiling(2.0, 1.0), 2.0);
        assert_eq!(floor_(3.9999999999999, 1.0), 4.0, "within the 1e-12 guard");
        assert_eq!(floor_(3.7, 1.0), 3.0);
    }

    #[test]
    fn text0_rounds_half_away_from_zero() {
        let cases = [
            (0.5, "1"),
            (1.5, "2"),
            (2.5, "3"),
            (-0.5, "-1"),
            (-2.5, "-3"),
            (-0.3, "0"),
            (7.99, "8"),
            (0.0, "0"),
            (-0.0, "0"),
            (1e20, "100000000000000000000"),
        ];
        for (x, want) in cases {
            assert_eq!(text0(x), want, "text0({x})");
        }
        assert_eq!(text0(f64::NAN), "#NUM!");
        assert_eq!(text0(f64::INFINITY), "#NUM!");
    }

    #[test]
    fn py_repr_matches_python() {
        let cases = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.5, "1.5"),
            (123.0, "123.0"),
            (0.1 + 0.2, "0.30000000000000004"),
            (1e-05, "1e-05"),
            (1.5e-05, "1.5e-05"),
            (0.0001, "0.0001"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (-1.25e20, "-1.25e+20"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (1.256637e-06, "1.256637e-06"),
        ];
        for (x, want) in cases {
            assert_eq!(py_repr(x), want, "py_repr({x:e})");
        }
        assert_eq!(py_repr(f64::NAN), "nan");
        assert_eq!(py_repr(f64::NEG_INFINITY), "-inf");
    }

    #[test]
    fn fmt_num_prints_whole_numbers_without_a_point() {
        assert_eq!(fmt_num(12.0), "12");
        assert_eq!(fmt_num(-3.0), "-3");
        assert_eq!(fmt_num(-0.0), "0");
        assert_eq!(fmt_num(12.5), "12.5");
        assert_eq!(fmt_num(1e16), "10000000000000000");
        assert_eq!(fmt_num(1e23), "99999999999999991611392");
        assert_eq!(fmt_num(f64::INFINITY), "inf");
    }

    #[test]
    fn fmt_fixed_rounds_ties_to_even_like_python() {
        assert_eq!(fmt_fixed(0.125, 2), "0.12");
        assert_eq!(fmt_fixed(0.375, 2), "0.38");
        assert_eq!(fmt_fixed(2.5, 0), "2");
        assert_eq!(fmt_fixed(0.25, 1), "0.2");
        assert_eq!(fmt_fixed(-0.001, 2), "-0.00");
        assert_eq!(fmt_fixed(f64::NAN, 2), "nan");
        assert_eq!(fmt_fixed(-f64::NAN, 1), "nan");
        assert_eq!(fmt_fixed(f64::INFINITY, 2), "inf");
        assert_eq!(fmt_fixed(f64::NEG_INFINITY, 2), "-inf");
    }

    #[test]
    fn is_close_matches_math_isclose() {
        assert!(is_close(1.0, 1.0 + 1e-10, 1e-9, 0.0));
        assert!(!is_close(1.0, 1.0 + 1e-8, 1e-9, 0.0));
        assert!(is_close(0.0, 1e-13, 1e-9, 1e-12));
        assert!(!is_close(0.0, 1e-11, 1e-9, 1e-12));
        assert!(is_close(f64::INFINITY, f64::INFINITY, 1e-9, 0.0));
        assert!(!is_close(f64::INFINITY, 1e308, 1e-9, 0.0));
        assert!(!is_close(f64::NAN, f64::NAN, 1e-9, 1e-12));
    }

    #[test]
    fn parity_close_follows_test_parity() {
        let num = |x| Value::Num(x);
        let text = |s: &str| Value::Text(s.to_owned());
        assert!(parity_close(&num(1.0), &Value::Int(1)));
        assert!(parity_close(&num(2.0), &num(2.0 * (1.0 + 5e-10))));
        assert!(!parity_close(&num(2.0), &num(2.0 * (1.0 + 5e-9))));
        assert!(parity_close(&text("OK"), &text("OK")));
        assert!(!parity_close(&text("OK"), &text("OK ")));
        assert!(
            parity_close(&num(12.5), &text("12.5")),
            "str(12.5) == '12.5'"
        );
        assert!(parity_close(&Value::Int(20), &text("20")));
        assert!(
            !parity_close(&num(20.0), &text("20")),
            "str(20.0) is '20.0'"
        );
        assert!(parity_close(&Value::None, &Value::None));
        assert!(!parity_close(&Value::None, &num(0.0)));
        assert!(!parity_close(&num(0.0), &Value::None));
    }
}
