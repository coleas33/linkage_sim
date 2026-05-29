// ── Editor tool and small types ───────────────────────────────────────────────

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::gui::state::MotionProfile;
use crate::solver::reactions::ValidationState;

// ── Alignment guides ─────────────────────────────────────────────────────────

/// Axis for an alignment guide line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentAxis {
    /// Same y-value -- horizontal guide line.
    Horizontal,
    /// Same x-value -- vertical guide line.
    Vertical,
}

/// A snap-alignment guide shown when a dragged point lines up with another
/// attachment point horizontally or vertically.
#[derive(Debug, Clone)]
pub struct AlignmentGuide {
    /// Whether this guide is horizontal (same y) or vertical (same x).
    pub axis: AlignmentAxis,
    /// The world coordinate value: y for Horizontal, x for Vertical.
    pub world_value: f64,
    /// Label of the aligned attachment point (e.g. "ground/A").
    pub label: String,
}

/// Joint type for the two-click creation flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingJointType {
    Revolute,
    Prismatic,
    Fixed,
}

/// Identifies which trajectory-target field should be populated by the next
/// canvas click. Set by the trajectory_panel "Pick on canvas" buttons; consumed
/// by the canvas interaction layer on the next left-click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingCanvasPickKind {
    /// Body-local point (requires hitting a body to capture body-local coords).
    LocalPt,
    /// World-frame axis origin.
    AxisOrigin,
    /// World-frame axis direction (vector from current origin to click point;
    /// effectively the click coords are read as the direction vector).
    AxisDir,
    /// World-frame reference point.
    RefPt,
}

/// Active tab in the left sidebar's property/inspection area.
///
/// `Properties` is the original full property panel (link editor, mass, etc.).
/// `Equations` switches the same panel to the live loop-equations view
/// (constraint Φ_J* values + λ multipliers). Defaults to `Properties` so
/// existing user workflows remain the default landing tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyPanelTab {
    Properties,
    Equations,
}

impl Default for PropertyPanelTab {
    fn default() -> Self {
        PropertyPanelTab::Properties
    }
}

/// Active editor tool — determines what happens on canvas clicks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorTool {
    /// Default: click to select entities. Drag empty space to pan.
    Select,
    /// Draw Link: click a point (or empty space for ground pivot), drag to
    /// another point → creates bar + auto-creates revolute joints at both
    /// ends if they connect to existing points.
    DrawLink,
    /// Multi-click body placement: click to place attachment points, then
    /// confirm to create a body with those points.
    AddBody,
    /// Click canvas to place a new ground pivot.
    AddGroundPivot,
    /// Two-click placement of a two-point force element.
    PlaceForce,
    /// Drag-to-define force zone creation: click and drag a rectangle.
    CreateForceZone,
    /// Click on a link to place a point mass at that position.
    PlaceMass,
    /// Drag on canvas to draw body geometry for the selected body.
    DrawBodyGeometry,
}

// ── Context menu target ──────────────────────────────────────────────────────

/// Stores what was under the cursor when a right-click occurred.
///
/// egui's `context_menu()` closure runs every frame while the menu is open,
/// but `secondary_clicked()` is only true on the trigger frame. This struct
/// persists the hit-test result so the menu content stays correct.
#[derive(Debug, Clone, Default)]
pub struct ContextMenuTarget {
    /// Joint ID if right-click landed on a joint.
    pub joint_id: Option<String>,
    /// Attachment point under cursor (body_id, point_name).
    /// Takes priority over body_area.
    pub attachment_point: Option<(String, String)>,
    /// Body area under cursor (body_id) -- only set when no
    /// attachment point is within HIT_RADIUS.
    pub body_area: Option<String>,
    /// World coordinates of the right-click position.
    pub world_pos: Option<[f64; 2]>,
}

// ── Selection ─────────────────────────────────────────────────────────────────

/// Which entity in the mechanism is currently selected for inspection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SelectedEntity {
    Body(String),
    Joint(String),
    Driver(String),
}

// ── Validation warnings ──────────────────────────────────────────────────────

/// Lightweight validation warnings computed after each rebuild.
#[derive(Clone, Debug, Default)]
pub struct ValidationWarnings {
    /// Grubler DOF mismatch: expected 0 for a fully-constrained driven mechanism.
    pub dof_warning: Option<String>,
    /// Bodies not connected by any joint.
    pub disconnected_bodies: Vec<String>,
    /// Mechanism has no driver.
    pub missing_driver: bool,
}

// ── Solver status ─────────────────────────────────────────────────────────────

/// Summary of the most recent solver call.
#[derive(Debug, Clone)]
pub struct SolverStatus {
    pub converged: bool,
    pub residual_norm: f64,
    pub iterations: usize,
}

impl Default for SolverStatus {
    fn default() -> Self {
        Self {
            converged: false,
            residual_norm: 0.0,
            iterations: 0,
        }
    }
}

// ── Force results ─────────────────────────────────────────────────────────────

/// Results from static/inverse-dynamics force computation at the current pose.
#[derive(Debug, Clone, Default)]
pub struct ForceResults {
    /// Driver torque (N*m) -- the required input effort from the driver constraint.
    pub driver_torque: Option<f64>,
    /// Per-joint reaction forces in global frame: joint_id -> (Fx, Fy) in Newtons.
    pub joint_reactions: HashMap<String, (f64, f64)>,
    /// Condition number of the constraint Jacobian at this pose.
    pub condition_number: Option<f64>,
    /// True if the statics solver detected an overconstrained system.
    pub is_overconstrained: bool,
    /// Mechanical advantage (output/input angular velocity ratio) at current pose.
    pub mechanical_advantage: Option<f64>,
    /// Per-element force contribution norms at the current pose.
    pub force_contributions: Vec<(String, f64)>,
    /// Virtual work cross-check result: (vw_torque, lagrange_torque, agrees).
    pub virtual_work_check: Option<(f64, f64, bool)>,
    /// Trustworthiness verdict for the most recent reaction solve, from the
    /// INDEPENDENT per-body equilibrium check (see
    /// `solver::reactions::body_equilibrium_residual`). `None` when no solve
    /// has run yet. `Some(Verified)` = genuinely cross-checked and balances;
    /// `Some(Unverified)` = mechanism has element/joint types the independent
    /// check can't model yet, so no physics claim is made; `Some(Failed)` =
    /// equilibrium violated / ill-conditioned — do not trust the numbers.
    /// Surfaced as a tri-state badge in the property panel.
    pub reaction_validation: Option<ValidationState>,
}

// ── Trajectory profile ───────────────────────────────────────────────────────

/// Trajectory specification for `SweepMode::Trajectory`: either an analytic
/// motion profile or a user-defined keyframe table with linear interpolation.
///
/// Variants:
///   - `Profile` — closed-form `(h, ḣ, ḧ)` from `MotionProfile` (ConstantSpeed,
///     Trapezoidal, SCurve).
///   - `KeyframeTable` — piecewise-linear `h(t)` from user-provided waypoints
///     (or imported CSV). `ḣ` is the segment slope; `ḧ` is reported as 0
///     (linear interpolation between waypoints has zero curvature on each
///     segment; the velocity step at waypoint boundaries is small at typical
///     trajectory sample rates).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Trajectory {
    /// Analytic profile (ConstantSpeed / Trapezoidal / SCurve).
    Profile(TrajectoryProfile),
    /// User-defined (t, h) waypoints with linear interpolation.
    KeyframeTable(KeyframeTrajectory),
}

impl Trajectory {
    /// Evaluate `(h(t), ḣ(t), ḧ(t))` at trajectory time `t`.
    pub fn evaluate(&self, t: f64) -> (f64, f64, f64) {
        match self {
            Trajectory::Profile(p) => p.evaluate(t),
            Trajectory::KeyframeTable(kt) => kt.evaluate(t),
        }
    }
    /// Return `n` uniform sample times across `[0, duration]`.
    pub fn sample_times(&self, n: usize) -> Vec<f64> {
        match self {
            Trajectory::Profile(p) => p.sample_times(n),
            Trajectory::KeyframeTable(kt) => kt.sample_times(n),
        }
    }
    /// Trajectory duration in seconds.
    pub fn duration(&self) -> f64 {
        match self {
            Trajectory::Profile(p) => p.duration,
            Trajectory::KeyframeTable(kt) => kt.duration(),
        }
    }
}

/// User-defined (t, h) waypoint trajectory with linear interpolation.
/// Waypoints must be sorted by time and span `[0, duration]`. The constructor
/// sorts by time ascending and preserves duplicate-t entries (later evaluation
/// uses the first matching segment).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KeyframeTrajectory {
    /// Sorted (t, h) pairs. First entry's t = 0 (typically); last entry's t
    /// defines the trajectory duration.
    pub waypoints: Vec<(f64, f64)>,
}

impl KeyframeTrajectory {
    /// Construct a new keyframe trajectory, sorting waypoints by time.
    pub fn new(waypoints: Vec<(f64, f64)>) -> Self {
        let mut wps = waypoints;
        wps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self { waypoints: wps }
    }

    /// Linear interpolation between adjacent waypoints. Outside the table
    /// (`t < first` or `t > last`), clamps to the edge value.
    ///
    /// Returns `(h, h_dot, h_ddot)`. `h_dot` is the segment slope between
    /// the bracketing waypoints; `h_ddot` is `0` everywhere (piecewise
    /// linear). Edge clamps return `h_dot = 0` because `h` is constant
    /// outside the waypoint range.
    pub fn evaluate(&self, t: f64) -> (f64, f64, f64) {
        if self.waypoints.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        if self.waypoints.len() == 1 {
            return (self.waypoints[0].1, 0.0, 0.0);
        }
        // Below first waypoint
        if t <= self.waypoints[0].0 {
            return (self.waypoints[0].1, 0.0, 0.0);
        }
        // Above last waypoint
        if t >= self.waypoints[self.waypoints.len() - 1].0 {
            return (self.waypoints.last().unwrap().1, 0.0, 0.0);
        }
        // Linear search for the bracketing pair (small N, simpler than binary search).
        for w in self.waypoints.windows(2) {
            let (t0, h0) = w[0];
            let (t1, h1) = w[1];
            if t >= t0 && t <= t1 {
                let dt = t1 - t0;
                if dt < 1e-12 {
                    return (h0, 0.0, 0.0);
                }
                let frac = (t - t0) / dt;
                let h = h0 + frac * (h1 - h0);
                let h_dot = (h1 - h0) / dt;
                let h_ddot = 0.0;
                return (h, h_dot, h_ddot);
            }
        }
        // Shouldn't reach here given the above guards.
        (self.waypoints.last().unwrap().1, 0.0, 0.0)
    }

    /// Trajectory duration: time of the last waypoint.
    pub fn duration(&self) -> f64 {
        self.waypoints.last().map(|w| w.0).unwrap_or(0.0)
    }

    /// Return `n` uniform sample times across `[0, duration]`.
    pub fn sample_times(&self, n: usize) -> Vec<f64> {
        assert!(n >= 2, "sample_times requires n >= 2");
        let d = self.duration();
        if d <= 0.0 {
            return vec![0.0; n];
        }
        (0..n).map(|i| (i as f64) * d / ((n - 1) as f64)).collect()
    }
}

/// Wraps an existing `MotionProfile` shape with absolute units (start, end, duration).
/// Used by `SweepMode::Trajectory` to define a target observable trajectory `h(t)`.
///
/// See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §5.5
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrajectoryProfile {
    /// Shape of the velocity profile (constant / trapezoidal / S-curve).
    pub shape: MotionProfile,
    /// `h(0)` in target units.
    pub start_value: f64,
    /// `h(duration)` in target units.
    pub end_value: f64,
    /// Trajectory duration in seconds. Must be > 0.
    pub duration: f64,
}

impl TrajectoryProfile {
    /// Evaluate `(h(t), ḣ(t), ḧ(t))` at trajectory time `t`.
    /// `t` is clamped to `[0, duration]`.
    pub fn evaluate(&self, t: f64) -> (f64, f64, f64) {
        assert!(self.duration > 0.0, "TrajectoryProfile.duration must be > 0");
        let t_clamped = t.clamp(0.0, self.duration);
        match self.shape {
            MotionProfile::ConstantSpeed => {
                let frac = t_clamped / self.duration;
                let span = self.end_value - self.start_value;
                let h = self.start_value + frac * span;
                let h_dot = span / self.duration;
                let h_ddot = 0.0;
                (h, h_dot, h_ddot)
            }
            MotionProfile::Trapezoidal { accel_fraction, decel_fraction } => {
                trapezoidal_value(
                    t_clamped, self.duration, self.start_value, self.end_value,
                    accel_fraction, decel_fraction,
                )
            }
            MotionProfile::SCurve { .. } => {
                scurve_value(t_clamped, self.duration, self.start_value, self.end_value)
            }
        }
    }

    /// Return `n` uniform sample times across `[0, duration]`.
    pub fn sample_times(&self, n: usize) -> Vec<f64> {
        assert!(n >= 2, "sample_times requires n >= 2");
        (0..n).map(|i| (i as f64) * self.duration / ((n - 1) as f64)).collect()
    }
}

fn trapezoidal_value(
    t: f64, duration: f64,
    start: f64, end: f64,
    accel_frac: f64, decel_frac: f64,
) -> (f64, f64, f64) {
    let cruise_frac = 1.0 - accel_frac - decel_frac;
    debug_assert!(cruise_frac >= 0.0);
    let span = end - start;

    let t_a = accel_frac * duration;
    let t_c = cruise_frac * duration;
    let t_d = decel_frac * duration;

    // Peak velocity v_peak: ∫velocity dt = span ⇒ v_peak (t_a/2 + t_c + t_d/2) = span
    let denom = 0.5 * t_a + t_c + 0.5 * t_d;
    if denom.abs() < 1e-15 {
        return (start, 0.0, 0.0);
    }
    let v_peak = span / denom;

    if t < t_a {
        // Accelerate
        let a = v_peak / t_a;
        let h = start + 0.5 * a * t * t;
        let h_dot = a * t;
        let h_ddot = a;
        (h, h_dot, h_ddot)
    } else if t < t_a + t_c {
        // Cruise
        let h = start + 0.5 * v_peak * t_a + v_peak * (t - t_a);
        (h, v_peak, 0.0)
    } else {
        // Decelerate
        let a = -v_peak / t_d;
        let dt = t - (t_a + t_c);
        let h_at_decel_start = start + 0.5 * v_peak * t_a + v_peak * t_c;
        let h = h_at_decel_start + v_peak * dt + 0.5 * a * dt * dt;
        let h_dot = v_peak + a * dt;
        let h_ddot = a;
        (h, h_dot.max(0.0), h_ddot)
    }
}

/// Pure quintic ease-in-out S-curve evaluation.
///
/// Uses the smooth-step polynomial `s(τ) = τ³(10 − 15τ + 6τ²)` with
/// `τ = t/duration ∈ [0, 1]`. The first derivative is `30τ²(1−τ)²`
/// (zero at endpoints, peak at τ=0.5) and the second derivative is
/// `60τ(1−2τ)(1−τ)` (zero at τ=0, 0.5, 1). The endpoint behaviour gives
/// zero velocity AND zero acceleration at both `t=0` and `t=duration`,
/// which is the defining property of a jerk-limited motion profile.
fn scurve_value(t: f64, duration: f64, start: f64, end: f64) -> (f64, f64, f64) {
    let span = end - start;
    let tau = (t / duration).clamp(0.0, 1.0);
    let s = tau.powi(3) * (10.0 - 15.0 * tau + 6.0 * tau * tau);
    let s_prime = 30.0 * tau.powi(2) * (1.0 - tau).powi(2);
    let s_doubleprime = 60.0 * tau * (1.0 - 2.0 * tau) * (1.0 - tau);
    let h = start + s * span;
    let h_dot = s_prime * span / duration;
    let h_ddot = s_doubleprime * span / (duration * duration);
    (h, h_dot, h_ddot)
}

/// Sensor configuration for state-estimation / closed-loop control.
///
/// Drives the §5 sensor-fusion derivation in the HTML report. Independent
/// of the mechanism core (just stores which joints / actuators have
/// sensors, not the actual readings). When `encoder_joint` is `Some`,
/// an encoder is assumed to be mounted on that revolute joint and
/// measures the relative angle between its two bodies; for joints
/// involving ground, that's just the moving body's orientation.
///
/// `noise_std_*` fields are 1-σ standard deviations of the sensor
/// noise, in SI units (rad for the encoder, m for the actuator). Used
/// only by the EKF derivation in the report (process / measurement
/// covariance matrices). Defaults are conservative (encoder ≈ 0.001 rad
/// ≈ 12 bits resolution; actuator ≈ 50 µm = LVDT-typical).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SensorConfig {
    /// Joint where an encoder is mounted. `None` = no encoder. Joint ID
    /// must reference an existing revolute joint in the mechanism for
    /// the sensor to be exercised.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_joint: Option<String>,
    /// 1-σ encoder noise [rad].
    #[serde(default = "SensorConfig::default_encoder_noise")]
    pub encoder_noise_std: f64,
    /// Whether the linear actuator's stroke is measured. Only meaningful
    /// when the mechanism has a LinearActuator force element or
    /// LinearDriver constraint; otherwise treated as `false`.
    #[serde(default)]
    pub actuator_position_enabled: bool,
    /// 1-σ actuator-position noise [m].
    #[serde(default = "SensorConfig::default_actuator_noise")]
    pub actuator_noise_std: f64,
}

impl SensorConfig {
    fn default_encoder_noise() -> f64 {
        0.001 // ~12-bit encoder over 2π → 2π / 4096 ≈ 1.5 mrad
    }

    fn default_actuator_noise() -> f64 {
        50e-6 // 50 µm: typical LVDT / linear encoder
    }

    /// Returns the number of active sensors (0, 1, or 2). Drives the
    /// EKF section's branching — 2 sensors → full fusion, 1 → single-
    /// sensor estimator, 0 → open-loop note.
    pub fn n_active_sensors(&self) -> usize {
        let mut n = 0;
        if self.encoder_joint.is_some() {
            n += 1;
        }
        if self.actuator_position_enabled {
            n += 1;
        }
        n
    }
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            encoder_joint: None,
            encoder_noise_std: Self::default_encoder_noise(),
            actuator_position_enabled: false,
            actuator_noise_std: Self::default_actuator_noise(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_profile_constant_velocity_evaluates_linearly() {
        let profile = TrajectoryProfile {
            shape: MotionProfile::ConstantSpeed,
            start_value: 0.0,
            end_value: 1.0,
            duration: 2.0,
        };
        let (h, _, _) = profile.evaluate(0.0);
        assert!((h - 0.0).abs() < 1e-12);
        let (h, _, _) = profile.evaluate(1.0);
        assert!((h - 0.5).abs() < 1e-12);
        let (h, h_dot, h_ddot) = profile.evaluate(2.0);
        assert!((h - 1.0).abs() < 1e-12);
        assert!((h_dot - 0.5).abs() < 1e-12); // (end-start)/duration = 0.5
        assert!(h_ddot.abs() < 1e-12);
    }

    #[test]
    fn trajectory_profile_sample_times_are_uniform() {
        let profile = TrajectoryProfile {
            shape: MotionProfile::ConstantSpeed,
            start_value: 0.0,
            end_value: 1.0,
            duration: 1.0,
        };
        let samples = profile.sample_times(5);
        assert_eq!(samples.len(), 5);
        assert!((samples[0] - 0.0).abs() < 1e-12);
        assert!((samples[4] - 1.0).abs() < 1e-12);
        assert!((samples[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn trajectory_profile_scurve_endpoints_match_target() {
        let profile = TrajectoryProfile {
            shape: MotionProfile::SCurve { jerk_fraction: 0.2 },
            start_value: 0.0,
            end_value: 1.0,
            duration: 1.0,
        };
        let (h, h_dot, h_ddot) = profile.evaluate(0.0);
        assert!((h - 0.0).abs() < 1e-12);
        assert!(h_dot.abs() < 1e-12, "jerk-limited: zero velocity at t=0");
        assert!(h_ddot.abs() < 1e-12, "jerk-limited: zero acceleration at t=0");

        let (h, _, _) = profile.evaluate(0.5);
        assert!(
            (h - 0.5).abs() < 1e-3,
            "midpoint of symmetric profile ~ midpoint span: got {}",
            h
        );

        let (h, h_dot, h_ddot) = profile.evaluate(1.0);
        assert!((h - 1.0).abs() < 1e-12);
        assert!(h_dot.abs() < 1e-12, "jerk-limited: zero velocity at t=duration");
        assert!(h_ddot.abs() < 1e-12, "jerk-limited: zero acceleration at t=duration");
    }

    #[test]
    fn keyframe_trajectory_linear_interpolation() {
        let kt = KeyframeTrajectory::new(vec![(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)]);
        let (h, _, _) = kt.evaluate(0.5);
        assert!((h - 1.0).abs() < 1e-12);
        let (h, _, _) = kt.evaluate(1.5);
        assert!((h - 1.5).abs() < 1e-12);
    }

    #[test]
    fn keyframe_trajectory_clamps_outside_range() {
        let kt = KeyframeTrajectory::new(vec![(0.5, 1.0), (1.5, 3.0)]);
        let (h, _, _) = kt.evaluate(0.0);
        assert!((h - 1.0).abs() < 1e-12); // clamps to first value
        let (h, _, _) = kt.evaluate(2.0);
        assert!((h - 3.0).abs() < 1e-12); // clamps to last value
    }

    #[test]
    fn keyframe_trajectory_constant_velocity_segment_returns_correct_h_dot() {
        let kt = KeyframeTrajectory::new(vec![(0.0, 0.0), (2.0, 4.0)]);
        let (_, h_dot, h_ddot) = kt.evaluate(1.0);
        assert!((h_dot - 2.0).abs() < 1e-12); // slope = 4/2 = 2
        assert_eq!(h_ddot, 0.0);
    }

    #[test]
    fn trajectory_profile_scurve_integrates_to_span() {
        // Numerical integration check: ∫₀^duration h_dot(t) dt = span
        let profile = TrajectoryProfile {
            shape: MotionProfile::SCurve { jerk_fraction: 0.2 },
            start_value: 0.5,
            end_value: 2.5,
            duration: 1.0,
        };
        let n = 1000;
        let dt = profile.duration / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let t = (i as f64 + 0.5) * dt;
            let (_, h_dot, _) = profile.evaluate(t);
            integral += h_dot * dt;
        }
        assert!(
            (integral - 2.0).abs() < 1e-3,
            "integral of h_dot should equal span (2.0): got {}",
            integral
        );
    }
}
