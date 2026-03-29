// ── View transform ────────────────────────────────────────────────────────────

/// Maps between world coordinates (meters) and screen coordinates (pixels).
///
/// Screen Y grows downward, so world Y is flipped.
pub struct ViewTransform {
    /// Screen-space offset of the world origin, in pixels.
    pub offset: [f32; 2],
    /// Pixels per meter.
    pub scale: f32,
    /// Mounting angle in radians; rotates world coordinates before pan/zoom.
    pub mounting_angle: f64,
}

impl Default for ViewTransform {
    fn default() -> Self {
        Self {
            offset: [400.0, 400.0],
            scale: 5000.0,
            mounting_angle: 0.0,
        }
    }
}

impl ViewTransform {
    /// Convert world coordinates (meters) to screen coordinates (pixels).
    ///
    /// Applies the mounting-angle rotation around the world origin before
    /// pan/zoom.  Screen Y is flipped relative to world Y.
    pub fn world_to_screen(&self, wx: f64, wy: f64) -> [f32; 2] {
        // Rotate world coordinates by mounting angle around the origin.
        let (sin_a, cos_a) = self.mounting_angle.sin_cos();
        let rx = cos_a * wx - sin_a * wy;
        let ry = sin_a * wx + cos_a * wy;

        let sx = self.offset[0] + (rx as f32) * self.scale;
        let sy = self.offset[1] - (ry as f32) * self.scale;
        [sx, sy]
    }

    /// Convert screen coordinates (pixels) back to world coordinates (meters).
    ///
    /// Reverses pan/zoom, then applies the inverse mounting-angle rotation.
    pub fn screen_to_world(&self, sx: f32, sy: f32) -> [f64; 2] {
        let rx = ((sx - self.offset[0]) / self.scale) as f64;
        let ry = (-(sy - self.offset[1]) / self.scale) as f64;

        // Inverse rotation (transpose of 2D rotation matrix).
        let (sin_a, cos_a) = self.mounting_angle.sin_cos();
        let wx = cos_a * rx + sin_a * ry;
        let wy = -sin_a * rx + cos_a * ry;
        [wx, wy]
    }
}
