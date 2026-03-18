//! Transmission angle computation for 4-bar linkage mechanisms.
//!
//! The transmission angle mu is the angle between the coupler and the output
//! link at their connecting joint. It measures how effectively force is
//! transmitted through the mechanism -- mu = 90 deg is ideal, mu near 0 deg or 180 deg
//! indicates poor force transmission (near toggle/singularity).
//!
//! For a 4-bar with links a (crank), b (coupler), c (rocker), d (ground):
//!     cos mu = (b^2 + c^2 - a^2 - d^2 + 2*a*d*cos(theta)) / (2*b*c)

/// Transmission angle at a configuration.
#[derive(Debug, Clone)]
pub struct TransmissionAngleResult {
    /// Transmission angle in radians, in (0, pi).
    pub angle_rad: f64,
    /// Transmission angle in degrees, in (0, 180).
    pub angle_deg: f64,
    /// |mu - 90 deg| in degrees.
    pub deviation_from_ideal: f64,
    /// True if deviation > 50 deg (mu < 40 deg or mu > 140 deg).
    pub is_poor: bool,
}

/// Compute transmission angle for a 4-bar linkage analytically.
///
/// Uses the closed-form formula:
///     cos mu = (b^2 + c^2 - a^2 - d^2 + 2*a*d*cos(theta)) / (2*b*c)
///
/// # Arguments
/// * `a` - Crank (input) length.
/// * `b` - Coupler length.
/// * `c` - Rocker (output) length.
/// * `d` - Ground (frame) length.
/// * `theta` - Crank angle in radians.
pub fn transmission_angle_fourbar(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    theta: f64,
) -> TransmissionAngleResult {
    let cos_mu =
        (b * b + c * c - a * a - d * d + 2.0 * a * d * theta.cos()) / (2.0 * b * c);
    // Clamp to [-1, 1] for numerical safety
    let cos_mu = cos_mu.clamp(-1.0, 1.0);
    let mu = cos_mu.acos();

    let deg = mu.to_degrees();
    let deviation = (deg - 90.0).abs();

    TransmissionAngleResult {
        angle_rad: mu,
        angle_deg: deg,
        deviation_from_ideal: deviation,
        is_poor: deviation > 50.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    fn transmission_angle_at_theta_zero() {
        // a=1, b=3, c=2, d=4, theta=0
        // cos mu = (9 + 4 - 1 - 16 + 2*1*4*1) / (2*3*2)
        //        = (9 + 4 - 1 - 16 + 8) / 12
        //        = 4 / 12 = 1/3
        let result = transmission_angle_fourbar(1.0, 3.0, 2.0, 4.0, 0.0);
        let expected_rad = (1.0_f64 / 3.0).acos();
        assert_abs_diff_eq!(result.angle_rad, expected_rad, epsilon = 1e-12);
        assert_abs_diff_eq!(result.angle_deg, expected_rad.to_degrees(), epsilon = 1e-10);
    }

    #[test]
    fn transmission_angle_ideal_at_90_degrees() {
        // For a 4-bar where mu = 90 deg, cos mu = 0.
        // That means: b^2 + c^2 - a^2 - d^2 + 2*a*d*cos(theta) = 0
        // With a=1, b=2, c=2, d=1: b^2+c^2-a^2-d^2 = 4+4-1-1 = 6
        // Need 2*a*d*cos(theta) = -6 => cos(theta) = -3
        // That's out of range, so let's pick values that work.
        // a=2, b=2, c=2, d=2, theta=pi/2: cos(pi/2)=0
        // cos mu = (4+4-4-4+0)/(2*2*2) = 0/8 = 0
        // => mu = 90 deg
        let result = transmission_angle_fourbar(2.0, 2.0, 2.0, 2.0, PI / 2.0);
        assert_abs_diff_eq!(result.angle_deg, 90.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.deviation_from_ideal, 0.0, epsilon = 1e-10);
        assert!(!result.is_poor);
    }

    #[test]
    fn transmission_angle_poor_near_toggle() {
        // Pick parameters that produce mu < 40 or mu > 140
        // a=1, b=4, c=4, d=1, theta=pi
        // cos mu = (16+16-1-1 + 2*1*1*cos(pi)) / (2*4*4)
        //        = (30 - 2) / 32 = 28/32 = 0.875
        // mu = acos(0.875) ~ 28.96 deg => deviation ~ 61, is_poor = true
        let result = transmission_angle_fourbar(1.0, 4.0, 4.0, 1.0, PI);
        assert!(result.is_poor);
        assert!(result.deviation_from_ideal > 50.0);
    }

    #[test]
    fn transmission_angle_clamping() {
        // Edge case: parameters that would yield |cos mu| > 1 without clamping.
        // a=10, b=1, c=1, d=10, theta=0
        // cos mu = (1+1-100-100+200) / (2*1*1) = 2/2 = 1.0
        // Exactly 1.0, so mu = 0 deg
        let result = transmission_angle_fourbar(10.0, 1.0, 1.0, 10.0, 0.0);
        assert_abs_diff_eq!(result.angle_rad, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.angle_deg, 0.0, epsilon = 1e-10);
    }
}
