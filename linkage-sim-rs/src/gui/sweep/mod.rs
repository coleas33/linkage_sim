//! Sweep analysis: full-rotation driver sweep and 4-bar link detection.

mod fourbar;
mod motion_profile;

pub(crate) use fourbar::detect_fourbar_links;
pub(crate) use motion_profile::apply_motion_profile;

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
use crate::forces::elements::{
    evaluate_linear_actuator, force_zone_overlap_ratio, ForceElement, LinearActuatorElement,
};
use crate::solver::assembly::assemble_jacobian;
use crate::solver::inverse_dynamics::solve_inverse_dynamics;
use crate::solver::kinematics::{solve_acceleration, solve_position, solve_velocity};
use crate::solver::statics::{
    extract_reactions, get_driver_reactions, solve_statics, JointReaction, StaticSolveResult,
};

use super::state::MotionProfile;

// ── Sweep data ───────────────────────────────────────────────────────────────

/// Sweep mode — selects how `compute_sweep_data` parameterises the
/// sweep and how plots interpret `SweepData.angles_deg`.
///
/// In Angle mode `angles_deg` carries degrees and the X-axis label
/// reads "Driver Angle"; in Stroke mode `angles_deg` carries metres
/// (despite the field name) and the X-axis label reads "Actuator
/// Stroke (mm)" — plot consumers branch on `is_stroke()` to format
/// the values correctly. The field name predates the multi-mode
/// design and is kept to avoid touching ~30 read sites; treat
/// `angles_deg` as "the X-axis values for this sweep" rather than
/// "degrees".
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SweepMode {
    /// Revolute driver: X-axis is angle (degrees, 0-360 default).
    Angle,
    /// Linear driver: X-axis is actuator stroke (metres in
    /// `angles_deg`, displayed in mm by plot consumers via the
    /// is_stroke branch).
    Stroke,
}

impl SweepMode {
    /// Returns true for stroke-based (linear driver) sweeps.
    pub fn is_stroke(&self) -> bool {
        matches!(self, SweepMode::Stroke)
    }
}

/// Pre-computed sweep results for the full driver rotation (0-360 degrees).
///
/// Computed once when a mechanism is loaded or the driver changes. Cached
/// to avoid recomputing every frame. Used by the plot panel and canvas.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Actuator extension rate (m/s) at each sweep angle.
    /// Computed as dL/dt from the velocity of the actuator attachment points.
    /// `None` when no LinearActuator force element is present.
    pub actuator_speeds: Option<Vec<f64>>,
    /// Required actuator power (W) at each sweep angle. P = F_actuator * dL/dt.
    /// Uses the statics-based actuator force.
    /// `None` when no LinearActuator force element is present.
    pub actuator_power: Option<Vec<f64>>,
    /// Required actuator power (W) from inverse dynamics. P = F_actuator_id * dL/dt.
    /// Uses the inverse-dynamics actuator force (includes inertial loads).
    /// `None` when no LinearActuator force element is present.
    pub actuator_power_id: Option<Vec<f64>>,
    /// Output force magnitude (N) at each sweep angle.
    /// Computed from force zone overlap: the net force the mechanism exerts
    /// on the output (= force_zone.force * overlap_ratio) at each crank angle.
    /// `None` when no ForceZone force element is present.
    pub output_forces: Option<Vec<f64>>,
    /// Driver torque with the active motion profile applied (N*m).
    /// For ConstantSpeed this equals `inverse_dynamics_torques`.
    /// For Trapezoidal it rescales inertial contributions by the profile's
    /// omega(theta) and adds alpha(theta) inertial loads.
    /// `None` when the profile is ConstantSpeed (no extra data needed).
    pub profile_torques: Option<Vec<f64>>,
    /// Profile angular velocity (rad/s) at each sweep angle.
    /// `None` when ConstantSpeed.
    pub profile_omega: Option<Vec<f64>>,
    /// Profile angular acceleration (rad/s^2) at each sweep angle.
    /// `None` when ConstantSpeed.
    pub profile_alpha: Option<Vec<f64>>,
    /// Target value h(t_k) at each sample. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_values: Option<Vec<f64>>,
    /// Achieved value g(q_k) at each sample. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub achieved_values: Option<Vec<f64>>,
    /// Tracking residual achieved - target. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracking_residual: Option<Vec<f64>>,
    /// Back-solved input parameter u_k. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub u_values: Option<Vec<f64>>,
    /// Back-solved input rate u̇_k. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub u_dot_values: Option<Vec<f64>>,
    /// Back-solved input accel ü_k. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub u_ddot_values: Option<Vec<f64>>,
    /// Per-sample inverse-solve diagnostic. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inverse_solve_statuses:
        Option<Vec<crate::solver::inverse_kinematics::InverseSolveStatus>>,
    /// Angles (degrees) at which toggle/dead points were detected.
    pub toggle_angles: Vec<f64>,
    /// Index range of an "active" sub-slice within the sweep data, used
    /// by plots to render a faded full-cycle context curve plus a solid
    /// highlighted sub-range. Always `None` under the current sweep
    /// behaviour (the sweep covers exactly the user's range, so the
    /// whole dataset is the active portion). Kept as an `Option` to
    /// preserve the plot rendering code path without behavioural change.
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
    // Detect whether this sweep should iterate angle (revolute driver)
    // or stroke (linear driver). When linear drivers are present the
    // X-axis is stroke in metres; sweep_range is interpreted as
    // (start_m, end_m) and step granularity is 1 mm (matches the user's
    // mm-frame Stroke slider). When no linear driver is present the
    // X-axis is angle in degrees as before. The default range when
    // sweep_range is None depends on mode: 0..=360° for angle, or the
    // length_0 ± 100 mm window for stroke.
    let is_stroke = mech.n_linear_drivers() > 0;
    let sweep_mode = if is_stroke {
        SweepMode::Stroke
    } else {
        SweepMode::Angle
    };
    let (start_x, end_x) = sweep_range.unwrap_or_else(|| {
        if is_stroke {
            (theta_0 - 0.1, theta_0 + 0.1) // ±100 mm around length_0
        } else {
            (0.0, 360.0)
        }
    });
    let step_size = if is_stroke { 0.001 } else { 1.0 }; // 1 mm or 1 deg
    let num_steps = (((end_x - start_x) / step_size).round() as i32).max(0);

    // Aliases used below — the iteration variable carries degrees in
    // angle mode and metres in stroke mode. SweepData.angles_deg stores
    // them as-is and plot consumers branch on sweep_mode.is_stroke().
    let start_angle_deg = start_x;
    let _end_angle_deg = end_x;

    let capacity = (num_steps + 1) as usize;

    // Detect the first LinearActuator force element for actuator force computation.
    let actuator_element: Option<LinearActuatorElement> =
        mech.forces().iter().find_map(|f| {
            if let ForceElement::LinearActuator(act) = f {
                Some(act.clone())
            } else {
                None
            }
        });
    let actuator_info: Option<(String, [f64; 2], String, [f64; 2])> =
        actuator_element.as_ref().map(|act| {
            (act.body_a.clone(), act.point_a, act.body_b.clone(), act.point_b)
        });

    // Collect all ForceZone elements for output force computation.
    let force_zones: Vec<_> = mech.forces().iter().filter_map(|f| {
        if let ForceElement::ForceZone(fz) = f {
            Some(fz.clone())
        } else {
            None
        }
    }).collect();
    let has_force_zones = !force_zones.is_empty();

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
        actuator_speeds: if actuator_info.is_some() {
            Some(Vec::with_capacity(capacity))
        } else {
            None
        },
        actuator_power: if actuator_info.is_some() {
            Some(Vec::with_capacity(capacity))
        } else {
            None
        },
        actuator_power_id: if actuator_info.is_some() {
            Some(Vec::with_capacity(capacity))
        } else {
            None
        },
        output_forces: if has_force_zones {
            Some(Vec::with_capacity(capacity))
        } else {
            None
        },
        profile_torques: None,
        profile_omega: None,
        profile_alpha: None,
        target_values: None,
        achieved_values: None,
        tracking_residual: None,
        u_values: None,
        u_dot_values: None,
        u_ddot_values: None,
        inverse_solve_statuses: None,
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

    // Sweep loop: iterate over num_steps+1 positions from start_angle_deg
    // to end_angle_deg in 1-degree increments.
    let mut q = q_start.clone();
    let mut q_at_zero = q_start.clone();

    for i in 0..=num_steps {
        // X is degrees (angle mode) or metres (stroke mode); see comment
        // at the top of the function.
        let x_value = start_angle_deg + (i as f64) * step_size;
        let t = if is_stroke {
            // f(t) = length_0 + velocity * t  =>  t = (x - length_0) / velocity
            // (omega = velocity, theta_0 = length_0 in linear mode).
            if omega.abs() > f64::EPSILON {
                (x_value - theta_0) / omega
            } else {
                0.0
            }
        } else {
            (x_value.to_radians() - theta_0) / omega
        };
        let angle_deg = x_value;

        match solve_position(mech, &q, t, 1e-10, 50) {
            Ok(result) if result.converged => {
                q = result.q.clone();
                // Update q_at_zero only when the sweep actually visits
                // angle 0 as its first sample. For range-limited sweeps
                // that start elsewhere (e.g. 200..=365) we leave the
                // caller-supplied q_at_zero untouched so later full
                // sweeps re-seed from the correct angle-0 configuration.
                if i == 0 && start_angle_deg.abs() < 0.5 {
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

                // Output force from force zone overlap (position-only).
                if has_force_zones {
                    let mech_bodies = mech.bodies();
                    let mut total_force_mag = 0.0_f64;
                    for fz in &force_zones {
                        let ratio = force_zone_overlap_ratio(fz, mech_state, mech_bodies, &q);
                        let fx = fz.force[0] * ratio;
                        let fy = fz.force[1] * ratio;
                        total_force_mag += (fx * fx + fy * fy).sqrt();
                    }
                    data.output_forces.as_mut().unwrap().push(total_force_mag);
                }

                // Transmission angle (4-bar only).
                if let Some((a, b, c, d)) = fourbar_links {
                    let theta_crank = angle_deg.to_radians();
                    let ta = transmission_angle_fourbar(a, b, c, d, theta_crank);
                    data.transmission_angles.as_mut().unwrap().push(ta.angle_deg);
                }

                // Driver torque and joint reactions from statics solve.
                // Reactions are deferred: if a LinearActuator is present, we
                // re-solve statics with the computed actuator force so that
                // joint reactions reflect the actuator as the prime mover.
                let mut pending_reactions: Option<Vec<JointReaction>> = None;
                let mut statics_q_forces: Option<DVector<f64>> = None;
                if let Ok(statics) = solve_statics(mech, &q, t) {
                    let reactions = extract_reactions(mech, &statics);
                    let torque = get_driver_reactions(&reactions)
                        .first()
                        .map(|r| r.effort)
                        .unwrap_or(0.0);
                    data.driver_torques.as_mut().unwrap().push(torque);
                    statics_q_forces = Some(statics.q_forces);
                    pending_reactions = Some(reactions);
                } else {
                    data.driver_torques.as_mut().unwrap().push(0.0);
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
                            let driver_torque = *data.driver_torques.as_ref().unwrap().last().unwrap_or(&0.0);
                            let actuator_force = if dl_dt.abs() > 1e-6 {
                                driver_torque * omega / dl_dt
                            } else {
                                f64::NAN // singular -- actuator nearly perpendicular to motion
                            };
                            data.actuator_forces.as_mut().unwrap().push(actuator_force);

                            // Inverse dynamics actuator force: same formula but
                            // using the ID torque (includes inertial loads).
                            let id_torque = *data.inverse_dynamics_torques.last().unwrap_or(&f64::NAN);
                            let id_force = if dl_dt.abs() > 1e-6 && id_torque.is_finite() {
                                id_torque * omega / dl_dt
                            } else {
                                f64::NAN
                            };
                            data.actuator_forces_id.as_mut().unwrap().push(id_force);

                            // Actuator extension rate (m/s).
                            data.actuator_speeds.as_mut().unwrap().push(dl_dt);

                            // Actuator power: P = F * v (statics and ID).
                            let power_statics = actuator_force * dl_dt;
                            data.actuator_power.as_mut().unwrap().push(
                                if power_statics.is_finite() { power_statics } else { f64::NAN }
                            );
                            let power_id = id_force * dl_dt;
                            data.actuator_power_id.as_mut().unwrap().push(
                                if power_id.is_finite() { power_id } else { f64::NAN }
                            );
                        } else {
                            data.actuator_forces.as_mut().unwrap().push(f64::NAN);
                            data.actuator_forces_id.as_mut().unwrap().push(f64::NAN);
                            data.actuator_speeds.as_mut().unwrap().push(f64::NAN);
                            data.actuator_power.as_mut().unwrap().push(f64::NAN);
                            data.actuator_power_id.as_mut().unwrap().push(f64::NAN);
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

                    // No velocity solve -- push NaN for actuator force, speed, power.
                    if actuator_info.is_some() {
                        data.actuator_forces.as_mut().unwrap().push(f64::NAN);
                        data.actuator_forces_id.as_mut().unwrap().push(f64::NAN);
                        data.actuator_speeds.as_mut().unwrap().push(f64::NAN);
                        data.actuator_power.as_mut().unwrap().push(f64::NAN);
                        data.actuator_power_id.as_mut().unwrap().push(f64::NAN);
                    }
                }

                // ── Two-pass reaction solve ─────────────────────────────
                // If a LinearActuator is present and we computed its required
                // force, re-solve statics with that force applied.  This makes
                // joint reactions reflect the actuator as the prime mover
                // (driver torque drops to ~0, load flows through actuator).
                if let (Some(q_forces), Some(act_elem)) =
                    (&statics_q_forces, &actuator_element)
                {
                    let act_force = data.actuator_forces.as_ref()
                        .and_then(|v| v.last().copied())
                        .unwrap_or(f64::NAN);
                    if act_force.is_finite() && act_force.abs() > 1e-12 {
                        let mut act_mod = act_elem.clone();
                        act_mod.force = act_force;
                        let q_dot_zero = DVector::zeros(mech.state().n_coords());
                        let q_actuator = evaluate_linear_actuator(
                            &act_mod, mech.state(), &q, &q_dot_zero,
                        );
                        let q_new = q_forces + &q_actuator;
                        let phi_q = assemble_jacobian(mech, &q, t);
                        let rhs = -&q_new;
                        if let Ok(lambdas) = phi_q.transpose().svd(true, true).solve(&rhs, 1e-14) {
                            let result2 = StaticSolveResult {
                                lambdas,
                                q_forces: q_new,
                                residual_norm: 0.0,
                                is_overconstrained: false,
                                condition_number: 0.0,
                            };
                            pending_reactions = Some(extract_reactions(mech, &result2));
                        }
                    }
                }

                // Push the final reactions (pass-2 if available, else pass-1).
                if let Some(reactions) = &pending_reactions {
                    for jr in reactions {
                        if jr.n_equations > 1 {
                            reaction_data
                                .entry(jr.joint_id.clone())
                                .or_insert_with(|| Vec::with_capacity(capacity))
                                .push(jr.resultant);
                        }
                    }
                } else {
                    // Statics failed -- push NaN for all tracked joints.
                    for values in reaction_data.values_mut() {
                        values.push(f64::NAN);
                    }
                }
            }
            _ => {
                // Solver failed at this angle — push NaN for all channels.
                // Don't update `q`: keep last-good as the initial guess.
                push_nan_row(
                    &mut data,
                    angle_deg,
                    &body_order,
                    &coupler_keys,
                    &mut coupler_vel_data,
                    &mut coupler_accel_data,
                    &mut reaction_data,
                    actuator_info.is_some(),
                    has_force_zones,
                );
            }
        }
    }

    data.joint_reaction_magnitudes = reaction_data;
    data.coupler_velocities = coupler_vel_data;
    data.coupler_accelerations = coupler_accel_data;

    // `active_range` used to mark a sub-slice of the full 0-360 sweep
    // for "faded context + solid active" rendering. Now that the sweep
    // IS the user's range (when enabled), the whole dataset is the
    // active range — plots render it solid with no faded overlay.
    data.active_range = None;

    (data, q_at_zero)
}

/// Push NaN for all data channels at a given angle (solver failure case).
///
/// Used when the position solver fails at an angle (e.g. non-Grashof mechanisms
/// outside their reachable range). Keeps all data vectors aligned so plots show
/// gaps rather than crashing on mismatched lengths.
fn push_nan_row(
    data: &mut SweepData,
    angle_deg: f64,
    body_order: &[String],
    coupler_keys: &[(String, String, nalgebra::Vector2<f64>)],
    coupler_vel_data: &mut HashMap<String, Vec<f64>>,
    coupler_accel_data: &mut HashMap<String, Vec<f64>>,
    reaction_data: &mut HashMap<String, Vec<f64>>,
    has_actuator: bool,
    has_force_zones: bool,
) {
    data.angles_deg.push(angle_deg);

    for body_id in body_order {
        data.body_angles.get_mut(body_id).unwrap().push(f64::NAN);
    }

    for (key, _, _) in coupler_keys {
        data.coupler_traces
            .get_mut(key)
            .unwrap()
            .push([f64::NAN, f64::NAN]);
        coupler_vel_data.get_mut(key).unwrap().push(f64::NAN);
        coupler_accel_data.get_mut(key).unwrap().push(f64::NAN);
    }

    if data.transmission_angles.is_some() {
        data.transmission_angles.as_mut().unwrap().push(f64::NAN);
    }

    data.driver_torques.as_mut().unwrap().push(f64::NAN);
    for values in reaction_data.values_mut() {
        values.push(f64::NAN);
    }

    data.kinetic_energy.push(f64::NAN);
    data.potential_energy.push(f64::NAN);
    data.total_energy.push(f64::NAN);
    data.inverse_dynamics_torques.push(f64::NAN);
    data.mechanical_advantage.push(f64::NAN);

    if has_actuator {
        data.actuator_forces.as_mut().unwrap().push(f64::NAN);
        data.actuator_forces_id.as_mut().unwrap().push(f64::NAN);
        data.actuator_speeds.as_mut().unwrap().push(f64::NAN);
        data.actuator_power.as_mut().unwrap().push(f64::NAN);
        data.actuator_power_id.as_mut().unwrap().push(f64::NAN);
        data.actuator_lengths.as_mut().unwrap().push(f64::NAN);
    }

    if has_force_zones {
        data.output_forces.as_mut().unwrap().push(f64::NAN);
    }
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

    /// Reproduces bug: repeatedly toggling sweep range causes progressive
    /// degradation. Full sweeps (range=None) should consistently cover the
    /// same count; range sweeps should produce `max-min+1` samples.
    #[test]
    fn sweep_range_toggle_does_not_degrade() {
        let (mech, q0) = build_sample(SampleMechanism::ParallelogramPress);
        let omega = 2.0 * std::f64::consts::PI;
        let theta_0 = 0.0;
        let gravity = 9.81;
        let range = (150.0, 210.0);
        let range_len = (range.1 - range.0) as usize + 1; // 61

        // Initial sweep
        let (data1, q_zero1) = compute_sweep_data(&mech, &q0, omega, theta_0, gravity, None);
        let count_full = data1.angles_deg.len();
        assert!(count_full > 300, "Initial sweep should cover most of 360°, got {}", count_full);

        // Toggle ON — sweep literally covers 150..=210
        let (data2, q_zero2) = compute_sweep_data(&mech, &q_zero1, omega, theta_0, gravity, Some(range));
        assert_eq!(data2.angles_deg.len(), range_len, "ON sweep should cover exactly {} angles", range_len);

        // Toggle OFF — back to full count
        let (data3, q_zero3) = compute_sweep_data(&mech, &q_zero2, omega, theta_0, gravity, None);
        assert_eq!(data3.angles_deg.len(), count_full, "OFF sweep should match original full count");

        // Toggle ON/OFF a few more times
        let (data4, q_zero4) = compute_sweep_data(&mech, &q_zero3, omega, theta_0, gravity, Some(range));
        assert_eq!(data4.angles_deg.len(), range_len);

        let (data5, q_zero5) = compute_sweep_data(&mech, &q_zero4, omega, theta_0, gravity, None);
        assert_eq!(data5.angles_deg.len(), count_full);

        let (_data6, q_zero6) = compute_sweep_data(&mech, &q_zero5, omega, theta_0, gravity, Some(range));

        let (data7, _) = compute_sweep_data(&mech, &q_zero6, omega, theta_0, gravity, None);
        assert_eq!(data7.angles_deg.len(), count_full, "Final OFF sweep should still match original full count");

        // q_zero drift check (only meaningful across full sweeps)
        let diff = (&q_zero1 - &q_zero3).norm();
        assert!(diff < 1e-6, "q_zero should be stable across toggles, drift = {}", diff);
    }

    /// Verify that a sweep range extending past 360° (the "wrap" case the
    /// user originally ran into) produces a contiguous display-angle axis
    /// and the expected sample count.
    #[test]
    fn sweep_range_can_wrap_past_360() {
        let (mech, q0) = build_sample(SampleMechanism::ParallelogramPress);
        let omega = 2.0 * std::f64::consts::PI;
        let theta_0 = 0.0;
        let gravity = 9.81;

        let (data, _) =
            compute_sweep_data(&mech, &q0, omega, theta_0, gravity, Some((200.0, 365.0)));

        // 200..=365 inclusive in 1-degree steps = 166 samples
        assert_eq!(data.angles_deg.len(), 166);
        assert!((data.angles_deg.first().unwrap() - 200.0).abs() < 1e-9);
        assert!((data.angles_deg.last().unwrap() - 365.0).abs() < 1e-9);
        // X-axis is strictly monotonic (no wrap to 0)
        for pair in data.angles_deg.windows(2) {
            assert!(pair[1] > pair[0], "angles_deg must be monotonic for contiguous X-axis");
        }
        // active_range is None whenever the range is custom — the full
        // dataset IS the active range.
        assert!(data.active_range.is_none());
    }

    /// Test using AppState.compute_sweep() — the real code path.
    /// Simulates the user toggling "Limit Sweep Range" checkbox repeatedly,
    /// including moving the driver angle slider between toggles. With the
    /// new semantics, range-ON sweep covers exactly `max-min+1` samples;
    /// range-OFF sweep covers the full cycle (stable count across toggles).
    #[test]
    fn appstate_sweep_toggle_stable() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramPress);

        // Initial sweep happens in load_sample (range disabled -> full count)
        let count_full = state.sweep_data.as_ref().unwrap().angles_deg.len();
        assert!(count_full > 300, "Initial sweep should be full, got {}", count_full);

        let range_len = (210 - 150) + 1; // 61 samples for 150..=210

        for cycle in 0..3 {
            // Toggle ON
            state.sweep_range_enabled = true;
            state.sweep_angle_min_deg = 150.0;
            state.sweep_angle_max_deg = 210.0;
            state.compute_sweep();
            let count_on = state.sweep_data.as_ref().unwrap().angles_deg.len();
            assert_eq!(count_on, range_len, "Cycle {} ON: expected {} angles, got {}", cycle, range_len, count_on);

            // Simulate user moving driver angle slider between toggles
            state.driver_angle = (90.0 + cycle as f64 * 45.0).to_radians();
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
            assert_eq!(count_off, count_full, "Cycle {} OFF: expected {} angles, got {}", cycle, count_full, count_off);
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

    #[test]
    fn sweep_stroke_mode_for_linear_driver() {
        // A mechanism with a linear driver should sweep in stroke mode:
        // SweepMode::Stroke, X-axis values in metres, default range
        // length_0 ± 100mm in 1mm steps (201 samples).
        let length_0 = 0.5;
        let velocity = 0.01; // 10 mm/s
        let (mech, q0) = build_linear_driver_mech(velocity, length_0);

        let (data, _) = compute_sweep_data(&mech, &q0, velocity, length_0, 0.0, None);

        assert!(
            matches!(data.sweep_mode, SweepMode::Stroke),
            "Expected Stroke mode for linear driver"
        );
        assert_eq!(
            data.angles_deg.len(),
            201,
            "Default stroke sweep covers ±100 mm in 1mm steps (201 samples)"
        );
        // Values should be in metres, centred on length_0.
        assert!((data.angles_deg[0] - (length_0 - 0.1)).abs() < 1e-9);
        assert!((data.angles_deg[100] - length_0).abs() < 1e-9);
        assert!((data.angles_deg[200] - (length_0 + 0.1)).abs() < 1e-9);
    }

    #[test]
    fn sweep_stroke_mode_with_explicit_range() {
        // Caller-supplied range (in metres) should be honoured directly.
        let length_0 = 0.5;
        let velocity = 0.005;
        let (mech, q0) = build_linear_driver_mech(velocity, length_0);

        let (data, _) = compute_sweep_data(
            &mech,
            &q0,
            velocity,
            length_0,
            0.0,
            Some((length_0 - 0.05, length_0 + 0.05)), // ±50 mm
        );

        assert!(matches!(data.sweep_mode, SweepMode::Stroke));
        assert_eq!(data.angles_deg.len(), 101, "±50 mm in 1mm steps = 101 samples");
        assert!((data.angles_deg[0] - 0.45).abs() < 1e-9);
        assert!((data.angles_deg[100] - 0.55).abs() < 1e-9);
    }


    #[test]
    fn apply_motion_profile_constant_speed_is_noop() {
        let mut data = SweepData {
            angles_deg: vec![0.0, 90.0, 180.0, 270.0, 360.0],
            body_angles: HashMap::new(),
            coupler_traces: HashMap::new(),
            transmission_angles: None,
            driver_torques: Some(vec![1.0, 2.0, 1.5, 0.5, 1.0]),
            kinetic_energy: vec![],
            potential_energy: vec![],
            total_energy: vec![],
            inverse_dynamics_torques: vec![1.5, 2.5, 2.0, 1.0, 1.5],
            mechanical_advantage: vec![],
            joint_reaction_magnitudes: HashMap::new(),
            coupler_velocities: HashMap::new(),
            coupler_accelerations: HashMap::new(),
            actuator_forces: None,
            actuator_forces_id: None,
            actuator_lengths: None,
            actuator_speeds: None,
            actuator_power: None,
            actuator_power_id: None,
            output_forces: None,
            profile_torques: None,
            profile_omega: None,
            profile_alpha: None,
            target_values: None,
            achieved_values: None,
            tracking_residual: None,
            u_values: None,
            u_dot_values: None,
            u_ddot_values: None,
            inverse_solve_statuses: None,
            toggle_angles: Vec::new(),
            active_range: None,
            sweep_mode: SweepMode::Angle,
        };

        apply_motion_profile(&mut data, 2.0 * std::f64::consts::PI, MotionProfile::ConstantSpeed);
        assert!(data.profile_torques.is_none());
        assert!(data.profile_omega.is_none());
        assert!(data.profile_alpha.is_none());
    }

    #[test]
    fn apply_motion_profile_trapezoidal_produces_data() {
        let mut data = SweepData {
            angles_deg: (0..=360).map(|i| i as f64).collect(),
            body_angles: HashMap::new(),
            coupler_traces: HashMap::new(),
            transmission_angles: None,
            driver_torques: Some((0..=360).map(|i| (i as f64).to_radians().sin()).collect()),
            kinetic_energy: vec![],
            potential_energy: vec![],
            total_energy: vec![],
            inverse_dynamics_torques: (0..=360).map(|i| (i as f64).to_radians().sin() * 1.2).collect(),
            mechanical_advantage: vec![],
            joint_reaction_magnitudes: HashMap::new(),
            coupler_velocities: HashMap::new(),
            coupler_accelerations: HashMap::new(),
            actuator_forces: None,
            actuator_forces_id: None,
            actuator_lengths: None,
            actuator_speeds: None,
            actuator_power: None,
            actuator_power_id: None,
            output_forces: None,
            profile_torques: None,
            profile_omega: None,
            profile_alpha: None,
            target_values: None,
            achieved_values: None,
            tracking_residual: None,
            u_values: None,
            u_dot_values: None,
            u_ddot_values: None,
            inverse_solve_statuses: None,
            toggle_angles: Vec::new(),
            active_range: None,
            sweep_mode: SweepMode::Angle,
        };

        let omega = 2.0 * std::f64::consts::PI;
        let profile = MotionProfile::Trapezoidal {
            accel_fraction: 0.25,
            decel_fraction: 0.25,
        };
        apply_motion_profile(&mut data, omega, profile);

        let torques = data.profile_torques.as_ref().expect("profile_torques should be Some");
        assert_eq!(torques.len(), 361);

        let omegas = data.profile_omega.as_ref().expect("profile_omega should be Some");
        assert_eq!(omegas.len(), 361);

        let alphas = data.profile_alpha.as_ref().expect("profile_alpha should be Some");
        assert_eq!(alphas.len(), 361);

        // At 0 degrees, omega should be near zero (start of accel phase).
        assert!(omegas[0] < 1.0, "Omega at 0 deg should be small, got {}", omegas[0]);

        // At 180 degrees (mid-cycle), omega should be near peak (cruise phase).
        let peak_omega = omegas.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            (omegas[180] - peak_omega).abs() / peak_omega < 0.1,
            "Omega at 180 deg should be near peak"
        );

        // Alpha should be positive at the start (accelerating) and negative at the end.
        assert!(alphas[0] > 0.0, "Alpha at 0 deg should be positive");
        assert!(alphas[360] < 0.0, "Alpha at 360 deg should be negative");
    }

    #[test]
    fn trapezoidal_profile_on_fourbar_sweep() {
        // End-to-end test: compute sweep data for a 4-bar, then apply
        // trapezoidal profile and verify profile torques have same length.
        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let omega = 2.0 * std::f64::consts::PI;
        let theta_0 = 0.0;

        let (mut data, _) = compute_sweep_data(&mech, &q0, omega, theta_0, 9.81, None);
        let n = data.angles_deg.len();
        assert!(n > 300, "FourBar should produce a full sweep");

        apply_motion_profile(
            &mut data,
            omega,
            MotionProfile::Trapezoidal {
                accel_fraction: 0.2,
                decel_fraction: 0.3,
            },
        );

        let prof = data.profile_torques.as_ref().unwrap();
        assert_eq!(prof.len(), n, "Profile torques length should match sweep length");

        // At least some values should differ from the constant-speed ID torques
        // (unless inertia is zero, which it's not for FourBar).
        let differs = prof.iter().zip(data.inverse_dynamics_torques.iter())
            .any(|(p, c)| p.is_finite() && c.is_finite() && (p - c).abs() > 1e-12);
        assert!(differs, "Profile torques should differ from constant-speed ID torques");
    }
}
