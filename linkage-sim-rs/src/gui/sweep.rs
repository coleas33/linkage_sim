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
use crate::solver::inverse_dynamics::solve_inverse_dynamics;
use crate::solver::kinematics::{solve_acceleration, solve_position, solve_velocity};
use crate::solver::statics::{extract_reactions, get_driver_reactions, solve_statics};

// ── Sweep data ───────────────────────────────────────────────────────────────

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
    /// Angles (degrees) at which toggle/dead points were detected.
    pub toggle_angles: Vec<f64>,
}

pub(crate) fn compute_sweep_data(
    mech: &Mechanism,
    q_start: &DVector<f64>,
    omega: f64,
    theta_0: f64,
    gravity_magnitude: f64,
) -> (SweepData, DVector<f64>) {
    let mut data = SweepData {
        angles_deg: Vec::with_capacity(361),
        body_angles: HashMap::new(),
        coupler_traces: HashMap::new(),
        transmission_angles: None,
        driver_torques: Some(Vec::with_capacity(361)),
        kinetic_energy: Vec::with_capacity(361),
        potential_energy: Vec::with_capacity(361),
        total_energy: Vec::with_capacity(361),
        inverse_dynamics_torques: Vec::with_capacity(361),
        mechanical_advantage: Vec::with_capacity(361),
        joint_reaction_magnitudes: HashMap::new(),
        coupler_velocities: HashMap::new(),
        coupler_accelerations: HashMap::new(),
        toggle_angles: Vec::new(),
    };

    // Temporary accumulator for reaction data (filled during sweep,
    // then moved into `data` at the end).
    let mut reaction_data: HashMap<String, Vec<f64>> = HashMap::new();

    // Pre-allocate body angle vectors.
    let body_order: Vec<String> = mech.body_order().to_vec();
    for body_id in &body_order {
        data.body_angles
            .insert(body_id.clone(), Vec::with_capacity(361));
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
            data.coupler_traces.insert(key, Vec::with_capacity(361));
        }
        // Also trace attachment points on non-ground bodies (useful
        // for visualization even if no explicit coupler points exist).
        for (point_name, local) in &body.attachment_points {
            let key = format!("{}.{}", body_id, point_name);
            if !data.coupler_traces.contains_key(&key) {
                coupler_keys.push((key.clone(), body_id.clone(), *local));
                data.coupler_traces.insert(key, Vec::with_capacity(361));
            }
        }
    }

    // Pre-allocate coupler velocity and acceleration vectors.
    let mut coupler_vel_data: HashMap<String, Vec<f64>> = HashMap::new();
    let mut coupler_accel_data: HashMap<String, Vec<f64>> = HashMap::new();
    for (key, _, _) in &coupler_keys {
        coupler_vel_data.insert(key.clone(), Vec::with_capacity(361));
        coupler_accel_data.insert(key.clone(), Vec::with_capacity(361));
    }

    // Detect 4-bar link lengths for transmission angle.
    let fourbar_links = detect_fourbar_links(mech);
    if fourbar_links.is_some() {
        data.transmission_angles = Some(Vec::with_capacity(361));
    }

    // Detect driver/output body pair for mechanical advantage.
    let ma_bodies: Option<(String, String)> = mech.driver_body_pair().and_then(|(_bi, driver)| {
        let output = mech.body_order().iter()
            .filter(|b| b.as_str() != driver)
            .last()
            .cloned();
        output.map(|out| (driver.to_string(), out))
    });

    // Sweep from 0 to 360 degrees in 1-degree steps.
    let mut q = q_start.clone();
    let mut q_at_zero = q_start.clone();

    for i in 0..=360 {
        let angle_deg = i as f64;
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

                // Transmission angle (4-bar only).
                if let Some((a, b, c, d)) = fourbar_links {
                    let theta_crank = angle_deg.to_radians();
                    let ta = transmission_angle_fourbar(a, b, c, d, theta_crank);
                    data.transmission_angles.as_mut().unwrap().push(ta.angle_deg);
                }

                // Driver torque and joint reactions from statics solve.
                if let Ok(statics) = solve_statics(mech, &q, t) {
                    let reactions = extract_reactions(mech, &statics);
                    let torque = get_driver_reactions(&reactions)
                        .first()
                        .map(|r| r.effort)
                        .unwrap_or(0.0);
                    data.driver_torques.as_mut().unwrap().push(torque);

                    // Per-joint reaction magnitudes.
                    for jr in &reactions {
                        if jr.n_equations > 1 {
                            reaction_data
                                .entry(jr.joint_id.clone())
                                .or_insert_with(|| Vec::with_capacity(361))
                                .push(jr.resultant);
                        }
                    }
                } else {
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

    #[test]
    fn detect_fourbar_links_returns_none_for_sixbar() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::SixBarB1);
        let mech = state.mechanism.as_ref().unwrap();
        let links = detect_fourbar_links(mech);
        assert!(links.is_none(), "Should not detect 4-bar links for 6-bar");
    }
}
