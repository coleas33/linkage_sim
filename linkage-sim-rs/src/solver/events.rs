//! Event detection for forward dynamics integration.
//!
//! Events are scalar functions g(t, q, q_dot) that trigger when they
//! cross zero. The integrator checks events at each step and records
//! trigger times via linear interpolation.

use nalgebra::DVector;

use crate::core::state::State;

/// An event that can be detected during forward dynamics integration.
#[derive(Debug, Clone)]
pub struct DynamicsEvent {
    /// Human-readable label.
    pub label: String,
    /// Which zero-crossing direction to detect:
    /// 0 = both, +1 = rising only, -1 = falling only.
    pub direction: i32,
    /// Whether this event terminates the simulation.
    pub terminal: bool,
    /// The event function type.
    pub kind: EventKind,
}

/// Type of event function.
#[derive(Debug, Clone)]
pub enum EventKind {
    /// Relative angle between two bodies reaches a threshold.
    AngleLimit {
        body_i: String,
        body_j: String,
        limit_angle: f64,
    },
    /// Velocity of a coordinate crosses zero (direction reversal).
    VelocityReversal {
        body_id: String,
        /// Which coordinate: 0=x, 1=y, 2=theta
        coord_offset: usize,
    },
}

/// A detected event occurrence.
#[derive(Debug, Clone)]
pub struct EventOccurrence {
    /// Index of the event in the events list.
    pub event_index: usize,
    /// Interpolated time of the zero crossing.
    pub time: f64,
    /// Value of the event function just before crossing.
    pub value_before: f64,
    /// Value of the event function just after crossing.
    pub value_after: f64,
}

impl DynamicsEvent {
    /// Evaluate this event function at a given state.
    ///
    /// `y` is the packed state vector `[q; q_dot]` of length `2 * n_coords`.
    pub fn evaluate(&self, state: &State, y: &DVector<f64>) -> f64 {
        let n = state.n_coords();
        match &self.kind {
            EventKind::AngleLimit {
                body_i,
                body_j,
                limit_angle,
            } => {
                let theta_i = if state.is_ground(body_i) {
                    0.0
                } else {
                    let idx = state.get_index(body_i).unwrap();
                    y[idx.theta_idx()]
                };
                let theta_j = if state.is_ground(body_j) {
                    0.0
                } else {
                    let idx = state.get_index(body_j).unwrap();
                    y[idx.theta_idx()]
                };
                theta_j - theta_i - limit_angle
            }
            EventKind::VelocityReversal {
                body_id,
                coord_offset,
            } => {
                if state.is_ground(body_id) {
                    return 0.0;
                }
                let idx = state.get_index(body_id).unwrap();
                let vel_idx = idx.q_start + coord_offset;
                y[n + vel_idx] // velocity is in the second half of y
            }
        }
    }

    /// Check if a zero crossing occurred between two evaluations,
    /// respecting the configured direction filter.
    fn check_crossing(&self, val_before: f64, val_after: f64) -> bool {
        let crossed =
            val_before * val_after < 0.0 || (val_before != 0.0 && val_after == 0.0);
        if !crossed {
            return false;
        }
        match self.direction {
            1 => val_after > val_before,  // rising
            -1 => val_after < val_before, // falling
            _ => true,                    // both
        }
    }
}

/// Check all events between two time steps, return any occurrences.
///
/// For each event whose function value crosses zero between `y_before`
/// and `y_after`, the crossing time is estimated via linear interpolation.
pub fn check_events(
    events: &[DynamicsEvent],
    state: &State,
    t_before: f64,
    y_before: &DVector<f64>,
    t_after: f64,
    y_after: &DVector<f64>,
) -> Vec<EventOccurrence> {
    let mut occurrences = Vec::new();
    for (i, event) in events.iter().enumerate() {
        let val_before = event.evaluate(state, y_before);
        let val_after = event.evaluate(state, y_after);
        if event.check_crossing(val_before, val_after) {
            // Linear interpolation for crossing time
            let frac = val_before.abs() / (val_before.abs() + val_after.abs());
            let t_cross = t_before + frac * (t_after - t_before);
            occurrences.push(EventOccurrence {
                event_index: i,
                time: t_cross,
                value_before: val_before,
                value_after: val_after,
            });
        }
    }
    occurrences
}

// ── Constructor helpers ─────────────────────────────────────────────────────

/// Create an angle-limit event that fires when the relative angle
/// `theta_j - theta_i` crosses `limit_angle`.
pub fn angle_limit_event(
    label: &str,
    body_i: &str,
    body_j: &str,
    limit_angle: f64,
    direction: i32,
    terminal: bool,
) -> DynamicsEvent {
    DynamicsEvent {
        label: label.to_string(),
        direction,
        terminal,
        kind: EventKind::AngleLimit {
            body_i: body_i.to_string(),
            body_j: body_j.to_string(),
            limit_angle,
        },
    }
}

/// Create a velocity-reversal event that fires when the velocity
/// of a coordinate crosses zero.
pub fn velocity_reversal_event(
    label: &str,
    body_id: &str,
    coord_offset: usize,
) -> DynamicsEvent {
    DynamicsEvent {
        label: label.to_string(),
        direction: 0,
        terminal: false,
        kind: EventKind::VelocityReversal {
            body_id: body_id.to_string(),
            coord_offset,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::Body;
    use crate::core::mechanism::Mechanism;
    use crate::core::state::GROUND_ID;
    use crate::forces::elements::{ForceElement, GravityElement};
    use crate::solver::forward_dynamics::{
        simulate, simulate_with_events, ForwardDynamicsConfig,
    };
    use nalgebra::Vector2;
    use std::f64::consts::PI;

    /// Build a simple pendulum: single bar pinned to ground.
    /// CG at tip (point mass), L=1, m=1.
    fn build_pendulum() -> Mechanism {
        let ground = crate::core::body::make_ground(&[("O", 0.0, 0.0)]);

        let mut bar = Body::new("bar");
        bar.add_attachment_point("A", 0.0, 0.0).unwrap();
        bar.mass = 1.0;
        bar.cg_local = Vector2::new(1.0, 0.0);
        bar.izz_cg = 0.0;

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", GROUND_ID, "O", "bar", "A")
            .unwrap();
        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        mech.build().unwrap();
        mech
    }

    fn pendulum_initial_state(
        mech: &Mechanism,
        theta0: f64,
    ) -> (DVector<f64>, DVector<f64>) {
        let state = mech.state();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, theta0);
        let q_dot = DVector::zeros(state.n_coords());
        (q, q_dot)
    }

    #[test]
    fn angle_limit_event_detects_crossing() {
        // Pendulum released from above horizontal (theta=0, i.e. bar pointing right).
        // It will swing down past -PI/2 (hanging). We set an angle-limit event
        // at theta = -PI/2 relative to ground and expect it to fire as the
        // pendulum swings through that angle.
        let mech = build_pendulum();
        let theta0 = 0.0; // horizontal, pointing right
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.002,
            ..Default::default()
        };

        // Event: relative angle bar-ground crosses -PI/2 (falling direction)
        let events = vec![angle_limit_event(
            "bar reaches hanging",
            GROUND_ID,
            "bar",
            -PI / 2.0,
            -1, // falling
            false,
        )];

        let result = simulate_with_events(
            &mech,
            &q0,
            &qd0,
            (0.0, 2.0),
            Some(&config),
            None,
            &events,
        )
        .unwrap();

        assert!(result.success);
        assert!(
            !result.detected_events.is_empty(),
            "Expected at least one angle-limit event, got none"
        );

        // The crossing time should be between 0 and 2 seconds
        let occ = &result.detected_events[0];
        assert!(
            occ.time > 0.0 && occ.time < 2.0,
            "Event time {} out of expected range",
            occ.time
        );
        assert_eq!(occ.event_index, 0);

        // val_before should be positive (theta > -PI/2),
        // val_after should be negative (theta < -PI/2),
        // because the event function is (theta_bar - 0) - (-PI/2) = theta_bar + PI/2.
        assert!(
            occ.value_before > 0.0,
            "Expected positive value_before, got {}",
            occ.value_before
        );
        assert!(
            occ.value_after < 0.0,
            "Expected negative value_after, got {}",
            occ.value_after
        );
    }

    #[test]
    fn velocity_reversal_detects_zero_crossing() {
        // Pendulum released from rest swings down, then back up. Its angular
        // velocity starts at 0, goes negative as it swings down, then reverses
        // (passes through zero) at the bottom of the swing when it reaches
        // maximum speed... wait, that's not right. The velocity reversal
        // happens when the pendulum reaches its lowest point (maximum speed)
        // -- no, the velocity reverses at the *turning points*.
        //
        // Actually: pendulum released from theta=0.3 (above hanging at -PI/2).
        // theta_dot starts at 0, goes negative, reaches min, goes back to 0
        // at the other turning point. So the angular velocity (theta_dot)
        // crosses zero at the turning points.
        //
        // We use coord_offset=2 to track theta_dot.
        let mech = build_pendulum();
        let theta0 = -PI / 2.0 + 0.5; // offset from hanging
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.002,
            ..Default::default()
        };

        // Detect theta_dot crossing zero (coord_offset=2 for theta)
        let events = vec![velocity_reversal_event("theta_dot reversal", "bar", 2)];

        let result = simulate_with_events(
            &mech,
            &q0,
            &qd0,
            (0.0, 3.0),
            Some(&config),
            None,
            &events,
        )
        .unwrap();

        assert!(result.success);

        // The pendulum should reverse direction multiple times in 3 seconds.
        // Period ~ 2*pi*sqrt(1/9.81) ~ 2.0 s, so we expect at least 2 reversals.
        assert!(
            result.detected_events.len() >= 2,
            "Expected at least 2 velocity reversals, got {}",
            result.detected_events.len()
        );

        // All events should have index 0 (only one event defined)
        for occ in &result.detected_events {
            assert_eq!(occ.event_index, 0);
            assert!(occ.time > 0.0 && occ.time < 3.0);
        }
    }

    #[test]
    fn terminal_event_stops_simulation() {
        // Set a terminal angle-limit event. The simulation should stop
        // when the event fires, producing fewer time steps than a
        // non-terminal simulation.
        let mech = build_pendulum();
        let theta0 = 0.0; // horizontal
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.002,
            ..Default::default()
        };

        // Terminal event: bar crosses -PI/4 (45 deg below horizontal)
        let terminal_events = vec![angle_limit_event(
            "terminal at -PI/4",
            GROUND_ID,
            "bar",
            -PI / 4.0,
            -1, // falling
            true,
        )];

        let result_terminal = simulate_with_events(
            &mech,
            &q0,
            &qd0,
            (0.0, 2.0),
            Some(&config),
            None,
            &terminal_events,
        )
        .unwrap();

        // Also run without events for comparison
        let result_full =
            simulate(&mech, &q0, &qd0, (0.0, 2.0), Some(&config), None).unwrap();

        assert!(result_terminal.success);
        assert!(result_full.success);

        // Terminal simulation should have fewer time steps
        assert!(
            result_terminal.t.len() < result_full.t.len(),
            "Terminal sim has {} steps, full sim has {} steps -- terminal should be shorter",
            result_terminal.t.len(),
            result_full.t.len()
        );

        // Terminal simulation should have detected at least one event
        assert!(
            !result_terminal.detected_events.is_empty(),
            "Expected terminal event to fire"
        );

        // The last recorded time should be near the event time
        let last_t = *result_terminal.t.last().unwrap();
        let event_t = result_terminal.detected_events[0].time;
        assert!(
            (last_t - event_t).abs() < config.max_step * 2.0,
            "Last time {} should be near event time {}",
            last_t,
            event_t
        );
    }

    #[test]
    fn direction_filtering_rising_only_misses_falling() {
        // Set a rising-only angle event. The pendulum swings through the
        // threshold in both directions; the rising-only event should only
        // fire on the return swing.
        let mech = build_pendulum();
        let theta0 = 0.0; // horizontal
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.002,
            ..Default::default()
        };

        let threshold = -PI / 4.0;

        // Rising-only event
        let rising_events = vec![angle_limit_event(
            "rising through -PI/4",
            GROUND_ID,
            "bar",
            threshold,
            1, // rising only
            false,
        )];

        // Both-directions event (for comparison)
        let both_events = vec![angle_limit_event(
            "both through -PI/4",
            GROUND_ID,
            "bar",
            threshold,
            0, // both
            false,
        )];

        let result_rising = simulate_with_events(
            &mech,
            &q0,
            &qd0,
            (0.0, 3.0),
            Some(&config),
            None,
            &rising_events,
        )
        .unwrap();

        let result_both = simulate_with_events(
            &mech,
            &q0,
            &qd0,
            (0.0, 3.0),
            Some(&config),
            None,
            &both_events,
        )
        .unwrap();

        assert!(result_rising.success);
        assert!(result_both.success);

        // The "both" result should have more events than "rising only"
        assert!(
            result_both.detected_events.len() > result_rising.detected_events.len(),
            "Both-direction events ({}) should outnumber rising-only events ({})",
            result_both.detected_events.len(),
            result_rising.detected_events.len()
        );

        // All rising-only events should have val_after > val_before (rising)
        for occ in &result_rising.detected_events {
            assert!(
                occ.value_after > occ.value_before,
                "Rising-only event should have val_after ({}) > val_before ({})",
                occ.value_after,
                occ.value_before
            );
        }
    }
}
