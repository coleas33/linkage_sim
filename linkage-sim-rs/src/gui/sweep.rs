//! Sweep analysis: full-rotation driver sweep and 4-bar link detection.

use nalgebra::DVector;
use std::collections::HashMap;

use crate::analysis::coupler::eval_coupler_point;
use crate::analysis::energy::compute_energy_state_mech;
use crate::analysis::transmission::{
    mechanical_advantage, transmission_angle_fourbar, VelocityCoord,
};
use crate::analysis::validation::check_toggle;
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::forces::elements::ForceElement;
use crate::solver::inverse_dynamics::solve_inverse_dynamics;
use crate::solver::kinematics::{solve_acceleration, solve_position, solve_velocity};
use crate::solver::statics::{extract_reactions, get_driver_reactions, solve_statics};

// ── Sweep data ───────────────────────────────────────────────────────────────

/// Sweep mode. Currently only angle-based sweeps are supported (all mechanisms
/// use revolute drivers). The enum is retained for API compatibility and future
/// use.
#[derive(Debug, Clone)]
pub enum SweepMode {
    /// Revolute driver: x-axis is angle in degrees (0-360).
    Angle,
}

impl SweepMode {
    /// Returns true if this is a stroke-based sweep (linear driver).
    /// Always false now that all mechanisms use revolute drivers.
    pub fn is_stroke(&self) -> bool {
        false
    }
}

/// Pre-computed sweep results for the full driver rotation (0-360 degrees).
///
/// Computed once when a mechanism is loaded or the driver changes. Cached
/// to avoid recomputing every frame. Used by the plot panel and canvas.
#[derive(Debug, Clone)]
pub struct SweepData {
    /// Driver angles in degrees at which solutions were obtained.
    pub angles_deg: Vec<f64>,
    /// Body orientation angles (degrees) keyed by body ID.
    pub body_angles: HashMap<String, Vec<f64>>,
    /// Coupler point traces keyed by "body_id.point_name", each entry
    /// is a sequence of [x, y] world-coordinate pairs.
    pub coupler_traces: HashMap<String, Vec<[f64; 2]>>,
    /// Transmission angle (degrees) at each step, if the mechanism is a
    /// 4-bar linkage with identifiable link lengths.
    pub transmission_angles: Option<Vec<f64>>,
    /// Driver torque (N*m) at each step, computed from statics.
    pub driver_torques: Option<Vec<f64>>,
    /// Kinetic energy at each sweep step (Joules). Requires velocity solve.
    pub kinetic_energy: Vec<f64>,
    /// Gravitational potential energy at each sweep step (Joules).
    pub potential_energy: Vec<f64>,
    /// Total mechanical energy (KE + PE) at each sweep step (Joules).
    pub total_energy: Vec<f64>,
    /// Inverse dynamics driver torque at each sweep step (N·m).
    /// Includes inertial effects (unlike the statics-based driver_torques).
    pub inverse_dynamics_torques: Vec<f64>,
    /// Mechanical advantage (output/input angular velocity ratio) at each
    /// sweep step. Requires velocity solve and a detectable driver body pair.
    pub mechanical_advantage: Vec<f64>,
    /// Per-joint reaction force magnitudes (N) over the sweep.
    /// Key: joint_id, Value: vec of resultant force magnitudes at each step.
    pub joint_reaction_magnitudes: HashMap<String, Vec<f64>>,
    /// Coupler point velocity magnitudes over the sweep.
    /// Key: trace name (same as coupler_traces), Value: velocity magnitude (m/s) at each step.
    pub coupler_velocities: HashMap<String, Vec<f64>>,
    /// Coupler point acceleration magnitudes over the sweep.
    /// Key: trace name, Value: acceleration magnitude (m/s^2) at each step.
    pub coupler_accelerations: HashMap<String, Vec<f64>>,
    /// Required actuator force (N) at each sweep angle.
    /// Computed from driver torque and actuator velocity via power balance:
    /// F_actuator = driver_torque * omega / (dL/dt).
    /// `None` when no LinearActuator force element is present.
    pub actuator_forces: Option<Vec<f64>>,
    /// Required actuator force from inverse dynamics (includes inertial loads).
    /// More accurate than statics-based `actuator_forces` at high speed.
    /// Uses the same power-balance formula but with the inverse dynamics torque.
    /// `None` when no LinearActuator force element is present.
    pub actuator_forces_id: Option<Vec<f64>>,
    /// Actuator length (m) at each sweep angle.
    /// Distance between actuator attachment points A and B.
    /// `None` when no LinearActuator force element is present.
    pub actuator_lengths: Option<Vec<f64>>,
    /// Angles (degrees) at which toggle/dead points were detected.
    pub toggle_angles: Vec<f64>,
    /// Index range of the active sweep region within the full 0-360° data.
    /// `None` means the full range is active (no sweep limit).
    /// When `Some((start_idx, end_idx))`, both indices are inclusive.
    pub active_range: Option<(usize, usize)>,
    /// Whether this sweep was over angle (revolute driver) or stroke (linear driver).
    /// Determines X-axis labelling in plots and CSV exports.
    pub sweep_mode: SweepMode,
}

pub(crate) fn compute_sweep_data(
    mech: &Mechanism,
    q_start: &DVector<f64>,
    omega: f64,
    theta_0: f64,
    gravity_magnitude: f64,
    sweep_range: Option<(f64, f64)>,
) -> (SweepData, DVector<f64>) {
    // All mechanisms use revolute drivers: sweep 0-360 degrees in 1-degree steps.
    let num_steps = 360_i32;
    let sweep_mode = SweepMode::Angle;

    let capacity = (num_steps.max(0) + 1) as usize;

    // Detect the first LinearActuator force element for actuator force computation.
    let actuator_info: Option<(String, [f64; 2], String, [f64; 2])> =
        mech.forces().iter().find_map(|f| {
            if let ForceElement::LinearActuator(act) = f {
                Some((
                    act.body_a.clone(),
                    act.point_a,
                    act.body_b.clone(),
                    act.point_b,
                ))
            } else {
                None
            }
        });

    let mut data = SweepData {
        angles_deg: Vec::with_capacity(capacity),
        body_angles: HashMap::new(),
        coupler_traces: HashMap::new(),
        transmission_angles: None,
        driver_torques: Some(Vec::with_capacity(capacity)),
        kinetic_energy: Vec::with_capacity(capacity),
        potential_energy: Vec::with_capacity(capacity),
        total_energy: Vec::with_capacity(capacity),
        inverse_dynamics_torques: Vec::with_capacity(capacity),
        mechanical_advantage: Vec::with_capacity(capacity),
        joint_reaction_magnitudes: HashMap::new(),
        coupler_velocities: HashMap::new(),
        coupler_accelerations: HashMap::new(),
        actuator_forces: if actuator_info.is_some() {
            Some(Vec::with_capacity(capacity))
        } else {
            None
        },
        actuator_forces_id: if actuator_info.is_some() {
            Some(Vec::with_capacity(capacity))
        } else {
            None
        },
        actuator_lengths: if actuator_info.is_some() {
            Some(Vec::with_capacity(capacity))
        } else {
            None
        },
        toggle_angles: Vec::new(),
        active_range: None, // computed after sweep loop
        sweep_mode: sweep_mode.clone(),
    };

    // Temporary accumulator for reaction data (filled during sweep,
    // then moved into `data` at the end).
    let mut reaction_data: HashMap<String, Vec<f64>> = HashMap::new();

    // Pre-allocate body angle vectors.
    let body_order: Vec<String> = mech.body_order().to_vec();
    for body_id in &body_order {
        data.body_angles
            .insert(body_id.clone(), Vec::with_capacity(capacity));
    }

    // Pre-allocate coupler trace vectors.
    // Collect coupler point keys: "body_id.point_name"
    let mut coupler_keys: Vec<(String, String, nalgebra::Vector2<f64>)> = Vec::new();
    for (body_id, body) in mech.bodies() {
        if body_id == GROUND_ID {
            continue;
        }
        for (point_name, local) in &body.coupler_points {
            let key = format!("{}.{}", body_id, point_name);
            coupler_keys.push((key.clone(), body_id.clone(), *local));
            data.coupler_traces.insert(key, Vec::with_capacity(capacity));
        }
        // Also trace attachment points on non-ground bodies (useful
        // for visualization even if no explicit coupler points exist).
        for (point_name, local) in &body.attachment_points {
            let key = format!("{}.{}", body_id, point_name);
            if !data.coupler_traces.contains_key(&key) {
                coupler_keys.push((key.clone(), body_id.clone(), *local));
                data.coupler_traces.insert(key, Vec::with_capacity(capacity));
            }
        }
    }

    // Pre-allocate coupler velocity and acceleration vectors.
    let mut coupler_vel_data: HashMap<String, Vec<f64>> = HashMap::new();
    let mut coupler_accel_data: HashMap<String, Vec<f64>> = HashMap::new();
    for (key, _, _) in &coupler_keys {
        coupler_vel_data.insert(key.clone(), Vec::with_capacity(capacity));
        coupler_accel_data.insert(key.clone(), Vec::with_capacity(capacity));
    }

    // Detect 4-bar link lengths for transmission angle.
    let fourbar_links = detect_fourbar_links(mech);
    if fourbar_links.is_some() {
        data.transmission_angles = Some(Vec::with_capacity(capacity));
    }

    // Detect driver/output body pair for mechanical advantage.
    let ma_bodies: Option<(String, String)> = mech.driver_body_pair().and_then(|(_bi, driver)| {
        let output = mech.body_order().iter()
            .filter(|b| b.as_str() != driver)
            .last()
            .cloned();
        output.map(|out| (driver.to_string(), out))
    });

    // Sweep loop: iterate over num_steps+1 positions (0-360 degrees).
    let mut q = q_start.clone();
    let mut q_at_zero = q_start.clone();

    for i in 0..=num_steps.max(0) {
        let angle_deg = i as f64; // 0, 1, 2, ... 360
        let t = (angle_deg.to_radians() - theta_0) / omega;

        match solve_position(mech, &q, t, 1e-10, 50) {
            Ok(result) if result.converged => {
                q = result.q.clone();
                if i == 0 {
                    q_at_zero = q.clone();
                }

                data.angles_deg.push(angle_deg);

                // Toggle/dead-point detection.
                let toggle = check_toggle(mech, &q, t, 1e-6);
                if toggle.is_near_toggle {
                    data.toggle_angles.push(angle_deg);
                }

                let mech_state = mech.state();

                // Extract body angles.
                for body_id in &body_order {
                    let theta = mech_state.get_angle(body_id, &q);
                    data.body_angles
                        .get_mut(body_id)
                        .unwrap()
                        .push(theta.to_degrees());
                }

                // Extract coupler traces.
                for (key, body_id, local) in &coupler_keys {
                    let global = mech_state.body_point_global(body_id, local, &q);
                    data.coupler_traces
                        .get_mut(key)
                        .unwrap()
                        .push([global.x, global.y]);
                }

                // Actuator length (position-only, no velocity needed).
                if let Some(ref act_info) = actuator_info {
                    let (ref body_a, ref pt_a, ref body_b, ref pt_b) = *act_info;
                    let local_a = nalgebra::Vector2::new(pt_a[0], pt_a[1]);
                    let local_b = nalgebra::Vector2::new(pt_b[0], pt_b[1]);
                    let p_a = mech_state.body_point_global(body_a, &local_a, &q);
                    let p_b = mech_state.body_point_global(body_b, &local_b, &q);
                    let act_length = (p_b - p_a).norm();
                    data.actuator_lengths.as_mut().unwrap().push(act_length);
                }

                // Transmission angle (4-bar only).
                if let Some((a, b, c, d)) = fourbar_links {
                    let theta_crank = angle_deg.to_radians();
                    let ta = transmission_angle_fourbar(a, b, c, d, theta_crank);
                    data.transmission_angles.as_mut().unwrap().push(ta.angle_deg);
                }

                // Driver torque and joint reactions from statics solve.
                let driver_torque_val;
                if let Ok(statics) = solve_statics(mech, &q, t) {
                    let reactions = extract_reactions(mech, &statics);
                    let torque = get_driver_reactions(&reactions)
                        .first()
                        .map(|r| r.effort)
                        .unwrap_or(0.0);
                    driver_torque_val = torque;
                    data.driver_torques.as_mut().unwrap().push(torque);

                    // Per-joint reaction magnitudes.
                    for jr in &reactions {
                        if jr.n_equations > 1 {
                            reaction_data
                                .entry(jr.joint_id.clone())
                                .or_insert_with(|| Vec::with_capacity(capacity))
                                .push(jr.resultant);
                        }
                    }
                } else {
                    driver_torque_val = 0.0;
                    data.driver_torques.as_mut().unwrap().push(0.0);

                    // Push NaN for all tracked joints when statics fails.
                    for values in reaction_data.values_mut() {
                        values.push(f64::NAN);
                    }
                }

                // Velocity solve for energy and mechanical advantage.
                if let Ok(q_dot) = solve_velocity(mech, &q, t) {
                    let energy = compute_energy_state_mech(mech, &q, &q_dot, gravity_magnitude);
                    data.kinetic_energy.push(energy.kinetic);
                    data.potential_energy.push(energy.potential_gravity);
                    data.total_energy.push(energy.total);

                    // Mechanical advantage from velocity ratio.
                    if let Some((ref input_id, ref output_id)) = ma_bodies {
                        let ma_val = mechanical_advantage(
                            mech.state(), &q_dot,
                            input_id, output_id,
                            VelocityCoord::Theta, VelocityCoord::Theta,
                        ).map(|r| r.ma).unwrap_or(f64::NAN);
                        data.mechanical_advantage.push(ma_val);
                    } else {
                        data.mechanical_advantage.push(f64::NAN);
                    }

                    // Acceleration solve + inverse dynamics for torque including inertial effects
                    let accel_result = solve_acceleration(mech, &q, &q_dot, t);
                    if let Ok(ref q_ddot) = accel_result {
                        if let Ok(inv_dyn) = solve_inverse_dynamics(mech, &q, &q_dot, q_ddot, t) {
                            // Extract driver torque from the last lambda (driver is last constraint)
                            let n_lam = inv_dyn.lambdas.len();
                            if n_lam > 0 {
                                data.inverse_dynamics_torques.push(inv_dyn.lambdas[n_lam - 1]);
                            } else {
                                data.inverse_dynamics_torques.push(f64::NAN);
                            }
                        } else {
                            data.inverse_dynamics_torques.push(f64::NAN);
                        }
                    } else {
                        data.inverse_dynamics_torques.push(f64::NAN);
                    }

                    // Coupler point velocities and accelerations.
                    let n = q.len();
                    for (key, body_id, local) in &coupler_keys {
                        if let Ok(ref q_ddot) = accel_result {
                            let (_pos, vel, acc) = eval_coupler_point(
                                mech.state(), body_id, local, &q, &q_dot, q_ddot,
                            );
                            coupler_vel_data.get_mut(key).unwrap().push(vel.norm());
                            coupler_accel_data.get_mut(key).unwrap().push(acc.norm());
                        } else {
                            // No acceleration -- still store velocity.
                            let zero = DVector::zeros(n);
                            let (_pos, vel, _acc) = eval_coupler_point(
                                mech.state(), body_id, local, &q, &q_dot, &zero,
                            );
                            coupler_vel_data.get_mut(key).unwrap().push(vel.norm());
                            coupler_accel_data.get_mut(key).unwrap().push(f64::NAN);
                        }
                    }

                    // Actuator force from power balance:
                    // F_actuator = driver_torque * omega / dL_dt
                    if let Some(ref act_info) = actuator_info {
                        let (ref body_a, ref pt_a, ref body_b, ref pt_b) = *act_info;
                        let local_a = nalgebra::Vector2::new(pt_a[0], pt_a[1]);
                        let local_b = nalgebra::Vector2::new(pt_b[0], pt_b[1]);
                        let p_a = mech_state.body_point_global(body_a, &local_a, &q);
                        let p_b = mech_state.body_point_global(body_b, &local_b, &q);
                        let d_vec = p_b - p_a;
                        let length = d_vec.norm();
                        if length > 1e-12 {
                            let unit = d_vec / length;
                            let v_a = mech_state.body_point_velocity(body_a, &local_a, &q, &q_dot);
                            let v_b = mech_state.body_point_velocity(body_b, &local_b, &q, &q_dot);
                            let dl_dt = (v_b - v_a).dot(&unit);
                            let actuator_force = if dl_dt.abs() > 1e-12 {
                                driver_torque_val * omega / dl_dt
                            } else {
                                f64::NAN // singular -- actuator perpendicular to motion
                            };
                            data.actuator_forces.as_mut().unwrap().push(actuator_force);

                            // Inverse dynamics actuator force: same formula but
                            // using the ID torque (includes inertial loads).
                            let id_torque = *data.inverse_dynamics_torques.last().unwrap_or(&f64::NAN);
                            let id_force = if dl_dt.abs() > 1e-12 && id_torque.is_finite() {
                                id_torque * omega / dl_dt
                            } else {
                                f64::NAN
                            };
                            data.actuator_forces_id.as_mut().unwrap().push(id_force);
                        } else {
                            data.actuator_forces.as_mut().unwrap().push(f64::NAN);
                            data.actuator_forces_id.as_mut().unwrap().push(f64::NAN);
                        }
                    }
                } else {
                    data.kinetic_energy.push(f64::NAN);
                    data.potential_energy.push(f64::NAN);
                    data.total_energy.push(f64::NAN);
                    data.inverse_dynamics_torques.push(f64::NAN);
                    data.mechanical_advantage.push(f64::NAN);

                    // No velocity solve -- push NaN for coupler vel/accel.
                    for (key, _, _) in &coupler_keys {
                        coupler_vel_data.get_mut(key).unwrap().push(f64::NAN);
                        coupler_accel_data.get_mut(key).unwrap().push(f64::NAN);
                    }

                    // No velocity solve -- push NaN for actuator force.
                    if actuator_info.is_some() {
                        data.actuator_forces.as_mut().unwrap().push(f64::NAN);
                        data.actuator_forces_id.as_mut().unwrap().push(f64::NAN);
                    }
                }
            }
            _ => {
                // Solver failed at this angle -- stop sweep.
                // The mechanism likely cannot complete a full rotation.
                break;
            }
        }
    }

    data.joint_reaction_magnitudes = reaction_data;
    data.coupler_velocities = coupler_vel_data;
    data.coupler_accelerations = coupler_accel_data;

    // Compute active range indices from the sweep_range parameter.
    // The full 0-360 degree data is always present; active_range marks
    // the user-selected sub-range for highlighted rendering.
    data.active_range = sweep_range.and_then(|(min_val, max_val)| {
        if data.angles_deg.is_empty() {
            return None;
        }
        let start_idx = data.angles_deg.iter().position(|&a| a >= min_val).unwrap_or(0);
        let end_idx = data
            .angles_deg
            .iter()
            .rposition(|&a| a <= max_val)
            .unwrap_or(data.angles_deg.len().saturating_sub(1));
        Some((start_idx, end_idx))
    });

    (data, q_at_zero)
}

/// Try to detect a classic 4-bar linkage and return (crank, coupler, rocker,
/// ground) link lengths for transmission angle computation.
///
/// A 4-bar is identified by:
/// - Exactly 3 moving bodies
/// - Exactly 4 revolute joints
/// - Each moving body is a binary bar (exactly 2 attachment points)
/// - One of the moving bodies is the driven body (crank)
///
/// Returns `None` for non-4-bar mechanisms.
pub(crate) fn detect_fourbar_links(mech: &Mechanism) -> Option<(f64, f64, f64, f64)> {
    use crate::core::constraint::Constraint;

    let body_order = mech.body_order();
    if body_order.len() != 3 {
        return None;
    }

    let joints = mech.joints();
    let revolute_joints: Vec<_> = joints.iter().filter(|j| j.is_revolute()).collect();
    if revolute_joints.len() != 4 {
        return None;
    }

    // Identify which body is the crank (driven body).
    let driver_pair = mech.driver_body_pair()?;
    let driven_body = if driver_pair.0 == GROUND_ID {
        driver_pair.1
    } else {
        driver_pair.0
    };

    let bodies = mech.bodies();

    // Find the crank length (distance between its two attachment points).
    let crank_body = bodies.get(driven_body)?;
    if crank_body.attachment_points.len() != 2 {
        return None;
    }
    let crank_pts: Vec<_> = crank_body.attachment_points.values().collect();
    let crank_len = (crank_pts[0] - crank_pts[1]).norm();

    // Find the coupler and rocker. The coupler connects to the crank at a
    // non-ground joint, and the rocker connects the coupler to ground.
    // We identify them by finding which bodies connect to the crank vs ground.
    let other_bodies: Vec<&str> = body_order
        .iter()
        .map(|s| s.as_str())
        .filter(|s| *s != driven_body)
        .collect();

    if other_bodies.len() != 2 {
        return None;
    }

    // Check which of the two bodies connects to ground (rocker).
    let mut coupler_id = None;
    let mut rocker_id = None;
    for &body_id in &other_bodies {
        let connects_to_ground = revolute_joints.iter().any(|j| {
            (j.body_i_id() == body_id && j.body_j_id() == GROUND_ID)
                || (j.body_j_id() == body_id && j.body_i_id() == GROUND_ID)
        });
        let connects_to_crank = revolute_joints.iter().any(|j| {
            (j.body_i_id() == body_id && j.body_j_id() == driven_body)
                || (j.body_j_id() == body_id && j.body_i_id() == driven_body)
        });

        if connects_to_ground && !connects_to_crank {
            rocker_id = Some(body_id);
        } else if connects_to_crank && !connects_to_ground {
            coupler_id = Some(body_id);
        } else if connects_to_crank && connects_to_ground {
            // This body connects to both -- could be either in a parallelogram.
            // Treat as rocker if we haven't assigned one yet.
            if rocker_id.is_none() {
                rocker_id = Some(body_id);
            } else {
                coupler_id = Some(body_id);
            }
        }
    }

    let coupler_body = bodies.get(coupler_id?)?;
    let rocker_body = bodies.get(rocker_id?)?;

    if coupler_body.attachment_points.len() != 2 || rocker_body.attachment_points.len() != 2 {
        return None;
    }

    let coupler_pts: Vec<_> = coupler_body.attachment_points.values().collect();
    let coupler_len = (coupler_pts[0] - coupler_pts[1]).norm();

    let rocker_pts: Vec<_> = rocker_body.attachment_points.values().collect();
    let rocker_len = (rocker_pts[0] - rocker_pts[1]).norm();

    // Ground length: distance between the two ground pivots.
    let ground = bodies.get(GROUND_ID)?;
    if ground.attachment_points.len() != 2 {
        return None;
    }
    let ground_pts: Vec<_> = ground.attachment_points.values().collect();
    let ground_len = (ground_pts[0] - ground_pts[1]).norm();

    Some((crank_len, coupler_len, rocker_len, ground_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::samples::{build_sample, SampleMechanism};
    use crate::gui::state::AppState;

    #[test]
    fn detect_fourbar_links_returns_some_for_fourbar() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let mech = state.mechanism.as_ref().unwrap();
        let links = detect_fourbar_links(mech);
        assert!(links.is_some(), "Should detect 4-bar link lengths");
        let (a, b, c, d) = links.unwrap();
        assert!(a > 0.0 && b > 0.0 && c > 0.0 && d > 0.0);
    }

    /// Reproduces bug: repeatedly toggling sweep range causes progressive degradation.
    /// Each cycle should produce the same number of sweep angles (361 = full rotation).
    #[test]
    fn sweep_range_toggle_does_not_degrade() {
        let (mech, q0) = build_sample(SampleMechanism::ParallelogramPress);
        let omega = 2.0 * std::f64::consts::PI;
        let theta_0 = 0.0;
        let gravity = 9.81;

        // Initial sweep
        let (data1, q_zero1) = compute_sweep_data(&mech, &q0, omega, theta_0, gravity, None);
        let count1 = data1.angles_deg.len();
        assert!(count1 > 300, "Initial sweep should cover most of 360°, got {}", count1);

        // Toggle ON
        let (data2, q_zero2) = compute_sweep_data(&mech, &q_zero1, omega, theta_0, gravity, Some((150.0, 210.0)));
        assert_eq!(data2.angles_deg.len(), count1, "Sweep 2 should have same count");

        // Toggle OFF
        let (data3, q_zero3) = compute_sweep_data(&mech, &q_zero2, omega, theta_0, gravity, None);
        assert_eq!(data3.angles_deg.len(), count1, "Sweep 3 should have same count as sweep 1");

        // Toggle ON
        let (data4, q_zero4) = compute_sweep_data(&mech, &q_zero3, omega, theta_0, gravity, Some((150.0, 210.0)));
        assert_eq!(data4.angles_deg.len(), count1, "Sweep 4 should have same count");

        // Toggle OFF
        let (data5, q_zero5) = compute_sweep_data(&mech, &q_zero4, omega, theta_0, gravity, None);
        assert_eq!(data5.angles_deg.len(), count1, "Sweep 5 should have same count");

        // Toggle ON
        let (_data6, q_zero6) = compute_sweep_data(&mech, &q_zero5, omega, theta_0, gravity, Some((150.0, 210.0)));

        // Toggle OFF
        let (data7, _) = compute_sweep_data(&mech, &q_zero6, omega, theta_0, gravity, None);
        assert_eq!(data7.angles_deg.len(), count1, "Sweep 7 should have same count");

        // Also verify q_zero hasn't drifted
        let diff = (&q_zero1 - &q_zero3).norm();
        assert!(diff < 1e-6, "q_zero should be stable across toggles, drift = {}", diff);
    }

    /// Test using AppState.compute_sweep() — the real code path.
    /// Simulates the user toggling "Limit Sweep Range" checkbox repeatedly,
    /// including moving the driver angle slider between toggles.
    #[test]
    fn appstate_sweep_toggle_stable() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramPress);

        // Initial sweep happens in load_sample
        let count0 = state.sweep_data.as_ref().unwrap().angles_deg.len();
        assert!(count0 > 300, "Initial sweep should be full, got {}", count0);

        // Simulate toggling the checkbox 6 times
        for cycle in 0..3 {
            // Toggle ON
            state.sweep_range_enabled = true;
            state.sweep_angle_min_deg = 150.0;
            state.sweep_angle_max_deg = 210.0;
            state.compute_sweep();
            let count_on = state.sweep_data.as_ref().unwrap().angles_deg.len();
            assert_eq!(count_on, count0, "Cycle {} ON: expected {} angles, got {}", cycle, count0, count_on);

            // Simulate user moving driver angle slider between toggles
            state.driver_angle = (90.0 + cycle as f64 * 45.0).to_radians();
            // In the real app, this would update last_good_q via position solve.
            // Simulate that by solving at the new angle.
            if let Some(mech) = &state.mechanism {
                if let Ok(result) = crate::solver::kinematics::solve_position(
                    mech, &state.last_good_q,
                    (state.driver_angle - state.driver_theta_0) / state.driver_omega,
                    1e-10, 50,
                ) {
                    if result.converged {
                        state.q = result.q.clone();
                        state.last_good_q = result.q;
                    }
                }
            }

            // Toggle OFF
            state.sweep_range_enabled = false;
            state.compute_sweep();
            let count_off = state.sweep_data.as_ref().unwrap().angles_deg.len();
            assert_eq!(count_off, count0, "Cycle {} OFF: expected {} angles, got {}", cycle, count0, count_off);
        }
    }

    #[test]
    fn detect_fourbar_links_returns_none_for_sixbar() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::SixBarB1);
        let mech = state.mechanism.as_ref().unwrap();
        let links = detect_fourbar_links(mech);
        assert!(links.is_none(), "Should not detect 4-bar links for 6-bar");
    }

    /// Build a simple mechanism with a revolute joint + linear driver:
    /// bar pinned to ground at O/A. A separate ground point P is offset from O.
    /// The linear driver prescribes the distance from P to bar B.
    ///
    /// Geometry: ground has O=(0,0) and P=(0, -0.5) (below the pivot).
    /// Bar has A=(0,0) and B=(0.1, 0), length 0.1m.
    /// Bar starts at angle=60 degrees, so B=(0.05, 0.0866).
    /// Distance from P=(0, -0.5) to B = sqrt(0.05^2 + 0.5866^2) ~= 0.5887m.
    ///
    /// The bar starts at a non-extremal angle to avoid the branch-point
    /// singularity that occurs when P, O, B are collinear (at 0 or pi/2).
    fn build_linear_driver_mech(velocity: f64, length_0: f64) -> (Mechanism, DVector<f64>) {
        use crate::core::body::{make_bar, make_ground};
        use crate::core::linear_driver::constant_velocity_linear_driver;

        let ground = make_ground(&[("O", 0.0, 0.0), ("P", 0.0, -0.5)]);
        let bar = make_bar("bar", "A", "B", 0.1, 1.0, 0.01);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A").unwrap();

        // Linear driver: prescribes distance from ground P=(0, -0.5) to bar B.
        let ld = constant_velocity_linear_driver(
            "LD1", "ground", [0.0, -0.5], "bar", [0.1, 0.0], velocity, length_0,
        );
        mech.add_linear_driver(ld).unwrap();
        mech.build().unwrap();

        // Set initial pose: bar at angle=pi/3 (60 deg).
        // B global = (0.1*cos(60), 0.1*sin(60)) = (0.05, 0.0866).
        // Distance from P=(0,-0.5) to B = sqrt(0.05^2 + 0.5866^2) ~= 0.5887m.
        let mut q0 = mech.state().make_q();
        mech.state().set_pose("bar", &mut q0, 0.0, 0.0, std::f64::consts::FRAC_PI_3);
        (mech, q0)
    }

    /// Compute the initial distance from P to bar B at the starting angle.
    fn initial_distance() -> f64 {
        let theta = std::f64::consts::FRAC_PI_3;
        let b_x = 0.1 * theta.cos();
        let b_y = 0.1 * theta.sin();
        ((b_x - 0.0_f64).powi(2) + (b_y - (-0.5_f64)).powi(2)).sqrt()
    }

    #[test]
    fn linear_driver_position_solve_step_by_step() {
        use crate::solver::kinematics::solve_position;

        let length_0 = initial_distance(); // ~0.5887
        let velocity = -0.01; // retracting
        let (mech, q0) = build_linear_driver_mech(velocity, length_0);

        // Step 0: t=0
        let r0 = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(r0.converged, "Step 0 should converge, residual = {}", r0.residual_norm);

        // Step 1: small time step
        let t1 = 0.1; // prescribed distance = length_0 - 0.001
        let r1 = solve_position(&mech, &r0.q, t1, 1e-10, 50);
        assert!(r1.as_ref().is_ok(), "Step 1 should not error");
        let r1 = r1.unwrap();
        assert!(r1.converged, "Step 1 should converge, residual = {}", r1.residual_norm);
    }

    #[test]
    fn sweep_angle_mode_unchanged_for_revolute_driver() {
        // Verify angle mode still works correctly for revolute drivers.
        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let omega = 2.0 * std::f64::consts::PI;
        let theta_0 = 0.0;

        let (data, _) = compute_sweep_data(&mech, &q0, omega, theta_0, 0.0, None);

        assert!(
            matches!(data.sweep_mode, SweepMode::Angle),
            "Expected Angle mode for revolute driver"
        );
        assert_eq!(data.angles_deg.len(), 361, "Full 0-360 sweep");
        assert!((data.angles_deg[0] - 0.0).abs() < 1e-10);
        assert!((data.angles_deg[360] - 360.0).abs() < 1e-10);
    }
}
