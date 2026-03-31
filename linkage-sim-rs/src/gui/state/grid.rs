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

/// Clean spacing values for zoom-adaptive grid (down to 0.1mm).
const CLEAN_SPACINGS: [f64; 16] = [
    10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1,
    0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
    0.0005, 0.0002, 0.0001,
];

impl GridSettings {
    /// Compute the grid spacing from the current view scale.
    /// Targets ~20 grid cells across the given screen width.
    /// The snap and display grid always match.
    pub fn zoom_spacing(&self, screen_width: f32, view_scale: f32) -> f64 {
        if view_scale <= 0.0 || screen_width <= 0.0 {
            return self.spacing_m;
        }
        let world_width = screen_width as f64 / view_scale as f64;
        let raw = world_width / 20.0;
        CLEAN_SPACINGS
            .iter()
            .copied()
            .find(|&c| c <= raw)
            .unwrap_or(0.0001)
    }

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
