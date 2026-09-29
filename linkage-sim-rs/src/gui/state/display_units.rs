// ── Display units ─────────────────────────────────────────────────────────────

/// Length unit preference for display. Solvers always use SI (meters) internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LengthUnit {
    Meters,
    Millimeters,
}

/// Angle unit preference for display. Solvers always use radians internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AngleUnit {
    Radians,
    Degrees,
}

/// Display unit preferences. All conversion happens at the display boundary —
/// solvers never see converted values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DisplayUnits {
    pub length: LengthUnit,
    pub angle: AngleUnit,
}

impl Default for DisplayUnits {
    fn default() -> Self {
        Self {
            length: LengthUnit::Millimeters, // default to mm for engineering use
            angle: AngleUnit::Degrees,       // default to degrees
        }
    }
}

impl DisplayUnits {
    /// Convert meters to display length.
    pub fn length(&self, meters: f64) -> f64 {
        match self.length {
            LengthUnit::Meters => meters,
            LengthUnit::Millimeters => meters * 1000.0,
        }
    }

    /// Convert display length back to meters.
    pub fn length_to_si(&self, display: f64) -> f64 {
        match self.length {
            LengthUnit::Meters => display,
            LengthUnit::Millimeters => display / 1000.0,
        }
    }

    /// Length unit suffix string.
    pub fn length_suffix(&self) -> &'static str {
        match self.length {
            LengthUnit::Meters => " m",
            LengthUnit::Millimeters => " mm",
        }
    }

    /// Convert radians to display angle.
    pub fn angle(&self, radians: f64) -> f64 {
        match self.angle {
            AngleUnit::Radians => radians,
            AngleUnit::Degrees => radians.to_degrees(),
        }
    }

    /// Convert display angle back to radians.
    pub fn angle_to_si(&self, display: f64) -> f64 {
        match self.angle {
            AngleUnit::Radians => display,
            AngleUnit::Degrees => display.to_radians(),
        }
    }

    /// Angle unit suffix string.
    pub fn angle_suffix(&self) -> &'static str {
        match self.angle {
            AngleUnit::Radians => " rad",
            AngleUnit::Degrees => "\u{00b0}",
        }
    }

    /// X/Y axis label for length plots.
    pub fn length_axis_label(&self) -> &'static str {
        match self.length {
            LengthUnit::Meters => "m",
            LengthUnit::Millimeters => "mm",
        }
    }
}

/// Decimal places [`format_decimal`] keeps: 1 um in mm, 1 mg in kg. Finer
/// than anything a user types, coarse enough to read.
pub const DISPLAY_DECIMALS: usize = 6;

/// `value` with at most [`DISPLAY_DECIMALS`] decimals and no trailing
/// zeros: "2", "0.5", "1.234568". Never "-0". The text parses back to the
/// value rounded to those decimals, so a field that shows it and reads it
/// back can tell "unchanged" from "edited".
pub fn format_decimal(value: f64) -> String {
    let fixed = format!("{value:.DISPLAY_DECIMALS$}");
    let trimmed = if fixed.contains('.') { fixed.trim_end_matches('0').trim_end_matches('.') } else { fixed.as_str() };
    match trimmed {
        "-0" => "0".to_string(),
        text => text.to_string(),
    }
}

/// A mass in kg for display: "2 kg", "0.125 kg" ([`format_decimal`]).
pub fn format_mass_kg(kg: f64) -> String {
    format!("{} kg", format_decimal(kg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_decimal_keeps_up_to_six_decimals_without_trailing_zeros() {
        assert_eq!(format_decimal(2.0), "2");
        assert_eq!(format_decimal(0.5), "0.5");
        assert_eq!(format_decimal(50.25), "50.25");
        assert_eq!(format_decimal(1.23456789), "1.234568");
        assert_eq!(format_decimal(30.0004), "30.0004");
        assert_eq!(format_decimal(-12.5), "-12.5");
        assert_eq!(format_decimal(1500.0), "1500");
    }

    #[test]
    fn format_decimal_never_prints_negative_zero() {
        assert_eq!(format_decimal(0.0), "0");
        assert_eq!(format_decimal(-0.0), "0");
        assert_eq!(format_decimal(-0.0000001), "0");
    }

    #[test]
    fn format_decimal_passes_non_finite_values_through() {
        assert_eq!(format_decimal(f64::NAN), "NaN");
        assert_eq!(format_decimal(f64::INFINITY), "inf");
    }

    /// What the text reads back as is what `format_decimal` rounds to.
    #[test]
    fn format_decimal_round_trips_through_parse() {
        for v in [2.0, 0.5, 0.031234567, 30.0000004, 999.9999996] {
            let shown: f64 = format_decimal(v).parse().unwrap();
            assert!((shown - v).abs() <= 5e-7, "{v} shows as {shown}");
            assert_eq!(format_decimal(shown), format_decimal(v), "{v}: the shown text is stable");
        }
    }

    #[test]
    fn format_mass_kg_appends_the_unit() {
        assert_eq!(format_mass_kg(2.0), "2 kg");
        assert_eq!(format_mass_kg(0.125), "0.125 kg");
    }
}
