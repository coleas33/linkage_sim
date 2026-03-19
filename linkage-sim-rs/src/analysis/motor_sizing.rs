//! Motor sizing feasibility check.
//!
//! Given required (omega, T) operating points from inverse dynamics and a
//! DC motor's linear T-omega envelope, determines feasibility and margin
//! at each operating point.
//!
//! Motor envelope: T_available = T_stall * (1 - |speed| / omega_no_load)

/// Result of a motor sizing feasibility analysis.
#[derive(Debug, Clone)]
pub struct MotorSizingResult {
    /// Motor stall torque (N*m).
    pub stall_torque: f64,
    /// Motor no-load speed (rad/s).
    pub no_load_speed: f64,
    /// Per-step feasibility (true if the motor can deliver the required torque).
    pub feasible: Vec<bool>,
    /// Whether all operating points are feasible.
    pub all_feasible: bool,
    /// Margin at each step: (T_available - |T_required|) / T_available.
    /// Negative means infeasible.
    pub margins: Vec<f64>,
    /// Worst (minimum) margin across all operating points.
    pub worst_margin: f64,
    /// Index of the worst operating point.
    pub worst_index: usize,
}

/// Check motor feasibility at each operating point.
///
/// The motor envelope is a linear T-omega droop:
///     T_available = T_stall * max(0, 1 - |speed| / omega_no_load)
///
/// An operating point is feasible if |T_required| <= T_available.
///
/// # Arguments
/// * `speeds` - Angular velocities at each operating point (rad/s).
/// * `torques` - Required torques at each operating point (N*m).
/// * `stall_torque` - Motor stall torque (N*m).
/// * `no_load_speed` - Motor no-load speed (rad/s).
///
/// # Panics
/// Panics if `speeds` and `torques` have different lengths.
pub fn check_motor_sizing(
    speeds: &[f64],
    torques: &[f64],
    stall_torque: f64,
    no_load_speed: f64,
) -> MotorSizingResult {
    assert_eq!(
        speeds.len(),
        torques.len(),
        "speeds and torques must have the same length"
    );

    let n = speeds.len();
    let mut feasible = Vec::with_capacity(n);
    let mut margins = Vec::with_capacity(n);

    for i in 0..n {
        let speed = speeds[i].abs();
        let torque_required = torques[i].abs();

        let torque_available = if no_load_speed > 0.0 && speed < no_load_speed {
            stall_torque * (1.0 - speed / no_load_speed)
        } else {
            0.0
        };

        let margin = if torque_available > 1e-15 {
            (torque_available - torque_required) / torque_available
        } else if torque_required < 1e-15 {
            1.0 // no torque needed, any motor works
        } else {
            -1.0 // need torque but none available
        };

        feasible.push(margin >= 0.0);
        margins.push(margin);
    }

    // Find worst margin (minimum).
    let (worst_index, worst_margin) = if margins.is_empty() {
        (0, 1.0)
    } else {
        margins
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &m)| (i, m))
            .unwrap()
    };

    let all_feasible = feasible.iter().all(|&f| f);

    MotorSizingResult {
        stall_torque,
        no_load_speed,
        feasible,
        all_feasible,
        margins,
        worst_margin,
        worst_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn all_feasible_within_envelope() {
        // Motor: stall_torque=10 N*m, no_load_speed=100 rad/s
        // Operating points well within envelope.
        let speeds = [10.0, 20.0, 50.0];
        let torques = [5.0, 4.0, 2.0];

        let result = check_motor_sizing(&speeds, &torques, 10.0, 100.0);

        assert!(result.all_feasible);
        assert_eq!(result.feasible, vec![true, true, true]);
        // At speed=10: T_avail = 10*(1 - 10/100) = 9.0, margin = (9-5)/9 = 0.444
        assert_abs_diff_eq!(result.margins[0], (9.0 - 5.0) / 9.0, epsilon = 1e-12);
        // At speed=20: T_avail = 10*(1 - 0.2) = 8.0, margin = (8-4)/8 = 0.5
        assert_abs_diff_eq!(result.margins[1], 0.5, epsilon = 1e-12);
        // At speed=50: T_avail = 10*(1 - 0.5) = 5.0, margin = (5-2)/5 = 0.6
        assert_abs_diff_eq!(result.margins[2], 0.6, epsilon = 1e-12);
        assert!(result.worst_margin > 0.0);
    }

    #[test]
    fn one_infeasible_operating_point() {
        // Motor: stall_torque=10 N*m, no_load_speed=100 rad/s
        // Third point requires more torque than available.
        let speeds = [10.0, 20.0, 50.0];
        let torques = [5.0, 4.0, 8.0]; // at speed=50, T_avail=5.0, need 8.0

        let result = check_motor_sizing(&speeds, &torques, 10.0, 100.0);

        assert!(!result.all_feasible);
        assert_eq!(result.feasible, vec![true, true, false]);
        // At speed=50: T_avail=5.0, margin=(5-8)/5 = -0.6
        assert_abs_diff_eq!(result.margins[2], -0.6, epsilon = 1e-12);
        assert_eq!(result.worst_index, 2);
        assert!(result.worst_margin < 0.0);
    }

    #[test]
    fn edge_at_no_load_speed() {
        // At exactly no-load speed, T_available = 0. Any non-zero torque
        // requirement is infeasible.
        let speeds = [100.0];
        let torques = [0.1];

        let result = check_motor_sizing(&speeds, &torques, 10.0, 100.0);

        assert!(!result.all_feasible);
        assert_eq!(result.feasible, vec![false]);
        // T_avail = 0, T_required > 0 => margin = -1.0
        assert_abs_diff_eq!(result.margins[0], -1.0, epsilon = 1e-12);
    }

    #[test]
    fn zero_speed_gives_max_torque() {
        // At zero speed, the full stall torque is available.
        let speeds = [0.0];
        let torques = [7.0];

        let result = check_motor_sizing(&speeds, &torques, 10.0, 100.0);

        assert!(result.all_feasible);
        // T_avail = 10.0, margin = (10-7)/10 = 0.3
        assert_abs_diff_eq!(result.margins[0], 0.3, epsilon = 1e-12);
    }

    #[test]
    fn negative_speeds_handled_via_abs() {
        // Negative speeds should be treated the same as positive (|speed|).
        let speeds = [-30.0, 30.0];
        let torques = [3.0, 3.0];

        let result = check_motor_sizing(&speeds, &torques, 10.0, 100.0);

        assert!(result.all_feasible);
        // Both should produce identical margins.
        assert_abs_diff_eq!(result.margins[0], result.margins[1], epsilon = 1e-15);
        // T_avail = 10*(1 - 30/100) = 7.0, margin = (7-3)/7 = 4/7
        assert_abs_diff_eq!(result.margins[0], 4.0 / 7.0, epsilon = 1e-12);
    }

    #[test]
    fn empty_inputs_all_feasible() {
        let result = check_motor_sizing(&[], &[], 10.0, 100.0);

        assert!(result.all_feasible);
        assert!(result.feasible.is_empty());
        assert!(result.margins.is_empty());
        // Default worst_margin for empty input
        assert_abs_diff_eq!(result.worst_margin, 1.0, epsilon = 1e-15);
    }
}
