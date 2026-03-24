// ── Grid settings ─────────────────────────────────────────────────────────────

/// Grid display and snap-to-grid settings. Spacing is stored in meters (SI);
/// the UI converts to/from the active display unit at the display boundary.
pub struct GridSettings {
    /// Whether snap-to-grid is active during drag operations.
    pub snap_enabled: bool,
    /// Whether to draw the grid on the canvas.
    pub show_grid: bool,
    /// Grid spacing in meters (SI).
    pub spacing_m: f64,
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            snap_enabled: true,
            show_grid: true,
            spacing_m: 1.0,
        }
    }
}

impl GridSettings {
    /// Snap a single world-coordinate value to the nearest grid point.
    ///
    /// Returns the value unchanged when snapping is disabled or spacing is
    /// non-positive.
    pub fn snap(&self, value: f64) -> f64 {
        if !self.snap_enabled || self.spacing_m <= 0.0 {
            return value;
        }
        (value / self.spacing_m).round() * self.spacing_m
    }

    /// Snap an (x, y) world-coordinate pair to the nearest grid point.
    pub fn snap_point(&self, x: f64, y: f64) -> (f64, f64) {
        (self.snap(x), self.snap(y))
    }
}
