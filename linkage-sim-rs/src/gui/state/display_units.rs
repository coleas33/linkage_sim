// ── Display units ─────────────────────────────────────────────────────────────

/// Length unit preference for display. Solvers always use SI (meters) internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LengthUnit {
    Meters,
    Millimeters,
}

/// Angle unit preference for display. Solvers always use radians internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleUnit {
    Radians,
    Degrees,
}

/// Display unit preferences. All conversion happens at the display boundary —
/// solvers never see converted values.
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
