// ── View transform ────────────────────────────────────────────────────────────

/// Maps between world coordinates (meters) and screen coordinates (pixels).
///
/// Screen Y grows downward, so world Y is flipped.
pub struct ViewTransform {
    /// Screen-space offset of the world origin, in pixels.
    pub offset: [f32; 2],
    /// Pixels per meter.
    pub scale: f32,
}

impl Default for ViewTransform {
    fn default() -> Self {
        Self {
            offset: [400.0, 400.0],
            scale: 5000.0,
        }
    }
}

impl ViewTransform {
    /// Convert world coordinates (meters) to screen coordinates (pixels).
    ///
    /// Screen Y is flipped relative to world Y.
    pub fn world_to_screen(&self, wx: f64, wy: f64) -> [f32; 2] {
        let sx = self.offset[0] + (wx as f32) * self.scale;
        let sy = self.offset[1] - (wy as f32) * self.scale;
        [sx, sy]
    }

    /// Convert screen coordinates (pixels) back to world coordinates (meters).
    pub fn screen_to_world(&self, sx: f32, sy: f32) -> [f64; 2] {
        let wx = ((sx - self.offset[0]) / self.scale) as f64;
        let wy = (-(sy - self.offset[1]) / self.scale) as f64;
        [wx, wy]
    }
}
