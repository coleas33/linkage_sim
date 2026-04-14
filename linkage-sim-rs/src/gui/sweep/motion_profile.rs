//! Trapezoidal motion profile for sweep analysis.

use super::SweepData;
use crate::gui::state::MotionProfile;

/// Compute (omega_profile, alpha_profile) for a trapezoidal velocity profile
/// at a given fraction [0, 1] of the sweep cycle.
///
/// The profile ramps from 0 to `omega_peak` during the accel phase, holds
/// `omega_peak` during the cruise phase, and ramps back to 0 during decel.
/// `omega_peak` is chosen so that the area under the velocity curve equals
/// `total_angle` (2*pi for a full revolution).
///
/// Returns `(omega_at_fraction, alpha_at_fraction)`.
fn trapezoidal_profile_at(
    fraction: f64,
    total_angle: f64,
    cycle_time: f64,
    accel_frac: f64,
    decel_frac: f64,
) -> (f64, f64) {
    let cruise_frac = 1.0 - accel_frac - decel_frac;
    debug_assert!(cruise_frac >= 0.0);

    let t_accel = accel_frac * cycle_time;
    let t_cruise = cruise_frac * cycle_time;
    let t_decel = decel_frac * cycle_time;

    // Area under trapezoidal velocity curve = total_angle
    // = 0.5 * omega_peak * t_accel + omega_peak * t_cruise + 0.5 * omega_peak * t_decel
    // = omega_peak * (0.5*t_accel + t_cruise + 0.5*t_decel)
    let denom = 0.5 * t_accel + t_cruise + 0.5 * t_decel;
    if denom.abs() < 1e-15 {
        return (0.0, 0.0);
    }
    let omega_peak = total_angle / denom;

    let t = fraction * cycle_time;
    if t < t_accel {
        // Acceleration phase: omega ramps linearly from 0 to omega_peak.
        let alpha = omega_peak / t_accel;
        let omega = alpha * t;
        (omega, alpha)
    } else if t < t_accel + t_cruise {
        // Cruise phase: constant omega_peak, zero acceleration.
        (omega_peak, 0.0)
    } else {
        // Deceleration phase: omega ramps linearly from omega_peak to 0.
        let alpha = -omega_peak / t_decel;
        let t_in_decel = t - t_accel - t_cruise;
        let omega = omega_peak + alpha * t_in_decel;
        (omega.max(0.0), alpha)
    }
}

/// Map a crank angle (radians, relative to the sweep start) to a cycle fraction
/// under a trapezoidal profile.
///
/// For a trapezoidal velocity profile, the angle-to-fraction mapping is
/// piecewise quadratic. Given a target angle, we solve for the cycle fraction
/// by inverting the angle(t) curve numerically (bisection, since angle(t) is
/// monotonically increasing).
fn angle_to_fraction_trapezoidal(
    angle_rad: f64,
    total_angle: f64,
    accel_frac: f64,
    decel_frac: f64,
) -> f64 {
    if total_angle.abs() < 1e-15 {
        return 0.0;
    }
    // Normalise: what fraction of total_angle is this?
    let target_frac = (angle_rad / total_angle).clamp(0.0, 1.0);

    let cruise_frac = (1.0 - accel_frac - decel_frac).max(0.0);

    // Angle accumulated by the end of each phase (as fraction of total_angle).
    // Using cycle_time = 1 for simplicity since we only need fractions.
    let denom = 0.5 * accel_frac + cruise_frac + 0.5 * decel_frac;
    if denom.abs() < 1e-15 {
        return target_frac;
    }
    let omega_peak_norm = 1.0 / denom; // normalised omega_peak

    // With cycle_time = 1: theta(t) during accel = 0.5 * alpha * t^2
    // where alpha = omega_peak_norm / accel_frac
    // At t = accel_frac: theta = 0.5 * (omega_peak_norm/accel_frac) * accel_frac^2 = 0.5 * omega_peak_norm * accel_frac
    let theta_end_accel = 0.5 * omega_peak_norm * accel_frac;

    // Angle at end of cruise phase:
    let theta_end_cruise = theta_end_accel + omega_peak_norm * cruise_frac;

    // Total should be total_angle (normalised to 1):
    // theta_end_decel = theta_end_cruise + 0.5 * omega_peak_norm * decel_frac = 1.0

    let target_theta = target_frac; // normalised angle [0,1]

    if target_theta <= theta_end_accel {
        // In accel phase: theta = 0.5 * alpha * t^2, alpha = omega_peak_norm / accel_frac
        // t = sqrt(2 * theta / alpha) = sqrt(2 * theta * accel_frac / omega_peak_norm)
        if omega_peak_norm < 1e-15 {
            return 0.0;
        }
        let t = (2.0 * target_theta * accel_frac / omega_peak_norm).sqrt();
        t.clamp(0.0, 1.0)
    } else if target_theta <= theta_end_cruise {
        // In cruise phase: theta = theta_end_accel + omega_peak_norm * (t - accel_frac)
        // t = accel_frac + (theta - theta_end_accel) / omega_peak_norm
        let t = accel_frac + (target_theta - theta_end_accel) / omega_peak_norm;
        t.clamp(0.0, 1.0)
    } else {
        // In decel phase: theta = theta_end_cruise + omega_peak_norm * dt - 0.5 * a * dt^2
        // where dt = t - (accel_frac + cruise_frac), a = omega_peak_norm / decel_frac.
        // Rearranging: (a/2) * dt^2 - omega_peak_norm * dt + delta_theta = 0
        let a = omega_peak_norm / decel_frac.max(1e-15);
        let half_a = 0.5 * a;
        let delta_theta = target_theta - theta_end_cruise;
        let discriminant = omega_peak_norm * omega_peak_norm - 4.0 * half_a * delta_theta;
        let dt = if discriminant < 0.0 {
            decel_frac // fallback: end of cycle
        } else {
            // Take the smaller root (first time we reach this angle).
            (omega_peak_norm - discriminant.sqrt()) / (2.0 * half_a)
        };
        let t = accel_frac + cruise_frac + dt.clamp(0.0, decel_frac);
        t.clamp(0.0, 1.0)
    }
}

/// Apply a motion profile to existing sweep data, computing profile-adjusted
/// torques, angular velocities, and angular accelerations.
///
/// For `ConstantSpeed` this is a no-op (profile fields stay `None`).
/// For `Trapezoidal`, the inverse dynamics torque is rescaled at each sweep
/// angle to reflect the varying omega and alpha of the profile.
pub(crate) fn apply_motion_profile(data: &mut SweepData, omega: f64, profile: MotionProfile) {
    match profile {
        MotionProfile::ConstantSpeed => {
            data.profile_torques = None;
            data.profile_omega = None;
            data.profile_alpha = None;
        }
        MotionProfile::Trapezoidal {
            accel_fraction,
            decel_fraction,
        } => {
            let n = data.angles_deg.len();
            if n == 0 || omega.abs() < 1e-15 {
                return;
            }

            let total_angle = 2.0 * std::f64::consts::PI;
            let cycle_time = total_angle / omega;

            let mut prof_omega = Vec::with_capacity(n);
            let mut prof_alpha = Vec::with_capacity(n);
            let mut prof_torques = Vec::with_capacity(n);

            for i in 0..n {
                let angle_rad = data.angles_deg[i].to_radians();
                // Map this angle to a cycle fraction via the trapezoidal profile.
                let frac = angle_to_fraction_trapezoidal(
                    angle_rad,
                    total_angle,
                    accel_fraction,
                    decel_fraction,
                );
                let (omega_p, alpha_p) = trapezoidal_profile_at(
                    frac,
                    total_angle,
                    cycle_time,
                    accel_fraction,
                    decel_fraction,
                );
                prof_omega.push(omega_p);
                prof_alpha.push(alpha_p);

                // Compute profile torque from constant-speed sweep data:
                // T_profile = T_statics + (omega_p / omega)^2 * T_inertia_const + I_eff * alpha_p
                // where T_inertia_const = T_id_const - T_statics
                // and I_eff = T_inertia_const / omega^2
                let id_torque = if i < data.inverse_dynamics_torques.len() {
                    data.inverse_dynamics_torques[i]
                } else {
                    f64::NAN
                };
                let statics_torque = data
                    .driver_torques
                    .as_ref()
                    .and_then(|t| t.get(i).copied())
                    .unwrap_or(f64::NAN);

                if id_torque.is_finite() && statics_torque.is_finite() {
                    let t_inertia = id_torque - statics_torque;
                    let omega_ratio = omega_p / omega;
                    // I_eff = T_inertia / omega^2
                    let i_eff = t_inertia / (omega * omega);
                    let t_profile =
                        statics_torque + t_inertia * omega_ratio * omega_ratio + i_eff * alpha_p;
                    prof_torques.push(t_profile);
                } else {
                    prof_torques.push(f64::NAN);
                }
            }

            data.profile_omega = Some(prof_omega);
            data.profile_alpha = Some(prof_alpha);
            data.profile_torques = Some(prof_torques);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trapezoidal_profile_symmetric_integrates_to_total_angle() {
        // A symmetric trapezoidal profile (25% accel, 50% cruise, 25% decel)
        // should sweep exactly 2*PI radians over one cycle.
        let total_angle = 2.0 * std::f64::consts::PI;
        let omega = 2.0 * std::f64::consts::PI; // 1 rev/s
        let cycle_time = total_angle / omega; // 1.0 s
        let accel_frac = 0.25;
        let decel_frac = 0.25;

        // Numerically integrate omega(t) over the cycle using many small steps.
        let n = 10_000;
        let dt = 1.0 / n as f64;
        let mut integrated_angle = 0.0;
        for i in 0..n {
            let frac = (i as f64 + 0.5) * dt;
            let (omega_p, _alpha_p) =
                trapezoidal_profile_at(frac, total_angle, cycle_time, accel_frac, decel_frac);
            integrated_angle += omega_p * (dt * cycle_time);
        }
        assert!(
            (integrated_angle - total_angle).abs() < 1e-4,
            "Integrated angle should be 2*PI, got {}",
            integrated_angle
        );
    }

    #[test]
    fn trapezoidal_profile_zero_at_endpoints() {
        let total_angle = 2.0 * std::f64::consts::PI;
        let cycle_time = 1.0;
        let accel_frac = 0.25;
        let decel_frac = 0.25;

        let (omega_start, alpha_start) =
            trapezoidal_profile_at(0.0, total_angle, cycle_time, accel_frac, decel_frac);
        assert!(
            omega_start.abs() < 1e-10,
            "omega at start should be 0, got {}",
            omega_start
        );
        assert!(alpha_start > 0.0, "alpha at start should be positive");

        let (omega_end, alpha_end) =
            trapezoidal_profile_at(1.0, total_angle, cycle_time, accel_frac, decel_frac);
        assert!(
            omega_end.abs() < 1e-10,
            "omega at end should be 0, got {}",
            omega_end
        );
        assert!(alpha_end < 0.0, "alpha at end should be negative");
    }

    #[test]
    fn angle_to_fraction_roundtrip() {
        let total_angle = 2.0 * std::f64::consts::PI;
        let accel_frac = 0.25;
        let decel_frac = 0.25;

        // Test several fractions: convert fraction -> angle -> fraction.
        for &orig_frac in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let cycle_time = 1.0;
            let (omega, _) = trapezoidal_profile_at(
                orig_frac,
                total_angle,
                cycle_time,
                accel_frac,
                decel_frac,
            );
            // Numerically integrate to get angle at this fraction.
            let n = 10_000;
            let dt = 1.0 / n as f64;
            let mut angle = 0.0;
            let steps = (orig_frac * n as f64) as usize;
            for i in 0..steps {
                let f = (i as f64 + 0.5) * dt;
                let (om, _) = trapezoidal_profile_at(
                    f,
                    total_angle,
                    cycle_time,
                    accel_frac,
                    decel_frac,
                );
                angle += om * dt;
            }
            let recovered = angle_to_fraction_trapezoidal(
                angle,
                total_angle,
                accel_frac,
                decel_frac,
            );
            assert!(
                (recovered - orig_frac).abs() < 0.02,
                "Roundtrip failed: orig={}, angle={}, recovered={}",
                orig_frac,
                angle,
                recovered
            );
        }
    }
}
