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

// ── Mechanical advantage ──────────────────────────────────────────────────────

/// Result of instantaneous mechanical advantage computation.
#[derive(Debug, Clone)]
pub struct MechanicalAdvantageResult {
    /// Instantaneous mechanical advantage (dimensionless for angular/angular).
    /// MA = omega_output / omega_input.
    pub ma: f64,
    /// Input (driver) angular velocity (rad/s).
    pub input_velocity: f64,
    /// Output angular velocity (rad/s).
    pub output_velocity: f64,
}

/// Coordinate component to extract from the velocity vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocityCoord {
    /// Translational velocity in X.
    X,
    /// Translational velocity in Y.
    Y,
    /// Angular velocity (theta-dot).
    Theta,
}

/// Compute instantaneous mechanical advantage (velocity ratio) between two bodies.
///
/// For angular/angular MA the result is dimensionless. For translational/angular
/// the result has units of m/rad.
///
/// Returns `None` if either body is ground, a body is not found, or the input
/// velocity is effectively zero (|omega_in| < 1e-15).
///
/// # Arguments
/// * `state` - Mechanism state (provides body index mapping).
/// * `q_dot` - Solved velocity vector.
/// * `input_body_id` - Input (driver) body ID.
/// * `output_body_id` - Output body ID.
/// * `input_coord` - Which velocity component of the input body.
/// * `output_coord` - Which velocity component of the output body.
pub fn mechanical_advantage(
    state: &crate::core::state::State,
    q_dot: &nalgebra::DVector<f64>,
    input_body_id: &str,
    output_body_id: &str,
    input_coord: VelocityCoord,
    output_coord: VelocityCoord,
) -> Option<MechanicalAdvantageResult> {
    let v_in = get_velocity_component(state, q_dot, input_body_id, input_coord)?;
    let v_out = get_velocity_component(state, q_dot, output_body_id, output_coord)?;

    if v_in.abs() < 1e-15 {
        return None; // input stationary -- MA undefined
    }

    Some(MechanicalAdvantageResult {
        ma: v_out / v_in,
        input_velocity: v_in,
        output_velocity: v_out,
    })
}

/// Extract a single velocity component from q_dot for the given body.
///
/// Returns `None` if the body is ground or not found in the state.
fn get_velocity_component(
    state: &crate::core::state::State,
    q_dot: &nalgebra::DVector<f64>,
    body_id: &str,
    coord: VelocityCoord,
) -> Option<f64> {
    if state.is_ground(body_id) {
        return Some(0.0);
    }
    let idx = state.get_index(body_id).ok()?;
    Some(match coord {
        VelocityCoord::X => q_dot[idx.x_idx()],
        VelocityCoord::Y => q_dot[idx.y_idx()],
        VelocityCoord::Theta => q_dot[idx.theta_idx()],
    })
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

    // ── Mechanical advantage tests ──────────────────────────────────────

    use crate::core::state::State;
    use nalgebra::DVector;

    #[test]
    fn mechanical_advantage_basic_ratio() {
        // Two bodies: "crank" (indices 0,1,2) and "rocker" (indices 3,4,5).
        let mut state = State::new();
        state.register_body("crank").unwrap();
        state.register_body("rocker").unwrap();

        // q_dot: crank omega = 2.0, rocker omega = 1.0 => MA = 0.5
        let q_dot = DVector::from_vec(vec![0.0, 0.0, 2.0, 0.0, 0.0, 1.0]);
        let result = mechanical_advantage(
            &state, &q_dot, "crank", "rocker",
            VelocityCoord::Theta, VelocityCoord::Theta,
        );
        assert!(result.is_some());
        let r = result.unwrap();
        assert_abs_diff_eq!(r.ma, 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(r.input_velocity, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.output_velocity, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn mechanical_advantage_zero_input_returns_none() {
        let mut state = State::new();
        state.register_body("crank").unwrap();
        state.register_body("rocker").unwrap();

        // crank omega = 0.0 => MA undefined
        let q_dot = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let result = mechanical_advantage(
            &state, &q_dot, "crank", "rocker",
            VelocityCoord::Theta, VelocityCoord::Theta,
        );
        assert!(result.is_none());
    }

    #[test]
    fn mechanical_advantage_ground_input_returns_none() {
        // Ground always has velocity 0 => MA undefined.
        let mut state = State::new();
        state.register_body("crank").unwrap();

        let q_dot = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        let result = mechanical_advantage(
            &state, &q_dot, "ground", "crank",
            VelocityCoord::Theta, VelocityCoord::Theta,
        );
        assert!(result.is_none());
    }

    #[test]
    fn mechanical_advantage_ground_output_is_zero() {
        // Output on ground => v_out = 0 => MA = 0.
        let mut state = State::new();
        state.register_body("crank").unwrap();

        let q_dot = DVector::from_vec(vec![0.0, 0.0, 2.0]);
        let result = mechanical_advantage(
            &state, &q_dot, "crank", "ground",
            VelocityCoord::Theta, VelocityCoord::Theta,
        );
        assert!(result.is_some());
        assert_abs_diff_eq!(result.unwrap().ma, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn mechanical_advantage_unknown_body_returns_none() {
        let mut state = State::new();
        state.register_body("crank").unwrap();

        let q_dot = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        let result = mechanical_advantage(
            &state, &q_dot, "crank", "nonexistent",
            VelocityCoord::Theta, VelocityCoord::Theta,
        );
        assert!(result.is_none());
    }

    #[test]
    fn mechanical_advantage_translational_output() {
        // Test with x-velocity output (m/rad units).
        let mut state = State::new();
        state.register_body("crank").unwrap();
        state.register_body("slider").unwrap();

        // crank omega = 4.0, slider x_dot = 2.0 => MA = 0.5 m/rad
        let q_dot = DVector::from_vec(vec![0.0, 0.0, 4.0, 2.0, 0.0, 0.0]);
        let result = mechanical_advantage(
            &state, &q_dot, "crank", "slider",
            VelocityCoord::Theta, VelocityCoord::X,
        );
        assert!(result.is_some());
        assert_abs_diff_eq!(result.unwrap().ma, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn mechanical_advantage_negative_ratio() {
        // Bodies moving in opposite angular directions.
        let mut state = State::new();
        state.register_body("crank").unwrap();
        state.register_body("rocker").unwrap();

        // crank omega = 3.0, rocker omega = -1.5 => MA = -0.5
        let q_dot = DVector::from_vec(vec![0.0, 0.0, 3.0, 0.0, 0.0, -1.5]);
        let result = mechanical_advantage(
            &state, &q_dot, "crank", "rocker",
            VelocityCoord::Theta, VelocityCoord::Theta,
        );
        assert!(result.is_some());
        assert_abs_diff_eq!(result.unwrap().ma, -0.5, epsilon = 1e-12);
    }
}
