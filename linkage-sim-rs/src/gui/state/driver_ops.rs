//! Driver reassignment, expression driver, load case operations, parametric/counterbalance studies.

use std::f64::consts::PI;

use crate::forces::elements::ForceElement;
use crate::io::{
    load_mechanism_unbuilt_from_json, mechanism_to_json,
    DriverJson,
};
use crate::solver::kinematics::solve_position;

use super::{AppState, LoadCaseManager, SolverStatus};
use super::blueprint_ops::{
    joint_body_ids, generate_unique_id,
};
use crate::gui::sweep::compute_sweep_data;

impl AppState {
    /// Rebuild the mechanism with a different driver joint.
    ///
    /// Works for both sample mechanisms (via sample builder) and blueprint-based
    /// mechanisms (via blueprint driver mutation + rebuild).
    pub fn reassign_driver(&mut self, joint_id: &str) {
        // Try sample-based reassignment first (preserves sample-specific builder logic).
        if let Some(sample) = self.current_sample {
            self.push_undo();
            match crate::gui::samples::build_sample_with_driver(sample, Some(joint_id)) {
                Ok((mech, q0)) => {
                    self.driver_omega = 2.0 * PI;
                    match solve_position(&mech, &q0, 0.0, 1e-10, 50) {
                        Ok(result) => {
                            self.solver_status = SolverStatus {
                                converged: result.converged,
                                residual_norm: result.residual_norm,
                                iterations: result.iterations,
                            };
                            if result.converged {
                                self.q = result.q.clone();
                                self.last_good_q = result.q;
                            } else {
                                self.q = q0.clone();
                                self.last_good_q = q0;
                            }
                        }
                        Err(_) => {
                            self.solver_status = SolverStatus {
                                converged: false,
                                residual_norm: f64::NAN,
                                iterations: 0,
                            };
                            self.q = q0.clone();
                            self.last_good_q = q0;
                        }
                    }
                    self.driver_theta_0 = 0.0;
                    self.driver_angle = 0.0;
                    self.q_at_zero = self.q.clone();
                    self.blueprint = mechanism_to_json(&mech).ok();
                    self.mechanism = Some(mech);
                    self.driver_joint_id = Some(joint_id.to_string());
                    self.selected = None;
                    self.load_cases = LoadCaseManager::new_default(
                        joint_id,
                        self.driver_omega,
                        self.driver_theta_0,
                    );
                    self.playing = false;
                    self.animation_direction = 1.0;
                    self.pending_driver_reassignment = None;
                    self.mark_sweep_dirty();
                    self.compute_validation();
                    return;
                }
                Err(msg) => {
                    log::warn!("Sample-based driver reassignment failed: {}", msg);
                    // Fall through to blueprint-based reassignment.
                }
            }
        }

        // Blueprint-based reassignment: modify the driver in the blueprint and rebuild.
        let Some(bp) = &self.blueprint else { return };

        // Find the joint and its body pair.
        let Some(joint) = bp.joints.get(joint_id) else {
            log::warn!("Joint '{}' not found in blueprint", joint_id);
            return;
        };
        let (body_i, body_j) = joint_body_ids(joint);
        let body_i = body_i.to_string();
        let body_j = body_j.to_string();

        self.push_undo();
        let bp = self.blueprint.as_mut().unwrap();

        // Remove all existing drivers.
        bp.drivers.clear();

        // Add new constant-speed driver for the target joint's body pair.
        let driver_id = generate_unique_id("D", &bp.drivers);
        bp.drivers.insert(
            driver_id,
            DriverJson::ConstantSpeed {
                body_i,
                body_j,
                omega: self.driver_omega,
                theta_0: 0.0,
            },
        );

        self.driver_theta_0 = 0.0;
        self.driver_angle = 0.0;
        self.driver_joint_id = Some(joint_id.to_string());
        self.selected = None;
        self.playing = false;
        self.animation_direction = 1.0;
        self.pending_driver_reassignment = None;

        self.load_cases = LoadCaseManager::new_default(
            joint_id,
            self.driver_omega,
            self.driver_theta_0,
        );

        self.rebuild();
        self.q_at_zero = self.q.clone();
        // rebuild() already calls mark_sweep_dirty()
    }

    /// Switch the current driver to an expression-based driver.
    ///
    /// Replaces the first driver in the blueprint with an `Expression` variant,
    /// preserving the body pair. Pushes undo and rebuilds.
    pub fn set_expression_driver(&mut self, expr: &str, expr_dot: &str, expr_ddot: &str) {
        let Some(bp) = &self.blueprint else { return };

        // Find the first driver and its ID + body pair
        let Some((id, existing)) = bp.drivers.iter().next() else { return };
        let id = id.clone();
        let (body_i, body_j) = match existing {
            DriverJson::ConstantSpeed { body_i, body_j, .. }
            | DriverJson::Expression { body_i, body_j, .. } => {
                (body_i.clone(), body_j.clone())
            }
        };

        self.push_undo();
        let bp = self.blueprint.as_mut().unwrap();
        bp.drivers.insert(
            id,
            DriverJson::Expression {
                body_i,
                body_j,
                expr: expr.to_string(),
                expr_dot: expr_dot.to_string(),
                expr_ddot: expr_ddot.to_string(),
            },
        );
        self.rebuild();
    }

    /// Switch the current driver back to a constant-speed driver.
    ///
    /// Replaces the first driver in the blueprint with a `ConstantSpeed` variant,
    /// preserving the body pair. Pushes undo and rebuilds.
    pub fn set_constant_speed_driver(&mut self, omega: f64, theta_0: f64) {
        let Some(bp) = &self.blueprint else { return };

        // Find the first driver and its ID + body pair
        let Some((id, existing)) = bp.drivers.iter().next() else { return };
        let id = id.clone();
        let (body_i, body_j) = match existing {
            DriverJson::ConstantSpeed { body_i, body_j, .. }
            | DriverJson::Expression { body_i, body_j, .. } => {
                (body_i.clone(), body_j.clone())
            }
        };

        self.push_undo();
        let bp = self.blueprint.as_mut().unwrap();
        bp.drivers.insert(
            id,
            DriverJson::ConstantSpeed {
                body_i,
                body_j,
                omega,
                theta_0,
            },
        );
        self.driver_omega = omega;
        self.driver_theta_0 = theta_0;
        self.rebuild();
    }

    // ── Load case operations ──────────────────────────────────────────────

    /// Add a new load case by copying the current driver settings.
    pub fn add_load_case(&mut self) {
        let driver_joint_id = self
            .driver_joint_id
            .clone()
            .unwrap_or_default();
        self.load_cases.add_case(&driver_joint_id, self.driver_omega, self.driver_theta_0);
    }

    /// Remove the currently active load case.
    ///
    /// No-op if only one case remains. After removal, applies the new active case.
    pub fn remove_active_load_case(&mut self) {
        let index = self.load_cases.active_index;
        if self.load_cases.remove_case(index) {
            self.apply_load_case(self.load_cases.active_index);
        }
    }

    /// Switch to and apply the load case at the given index.
    ///
    /// Updates driver settings from the load case. If the driver joint differs
    /// from the current one, triggers a driver reassignment via the pending
    /// mechanism. Otherwise just updates omega/theta_0 and re-solves.
    pub fn apply_load_case(&mut self, index: usize) {
        if index >= self.load_cases.cases.len() {
            return;
        }

        self.push_undo();
        self.load_cases.active_index = index;

        let case = self.load_cases.cases[index].clone();

        let current_joint = self.driver_joint_id.clone().unwrap_or_default();

        if case.driver_joint_id != current_joint {
            // Different driver joint -- need to reassign.
            // Store the load case driver params so they survive reassignment,
            // then trigger the rebuild via the pending reassignment path.
            self.driver_omega = case.omega;
            self.driver_theta_0 = case.theta_0;
            self.pending_driver_reassignment = Some(case.driver_joint_id.clone());
        } else {
            // Same driver joint -- just update speed and angle.
            self.driver_omega = case.omega;
            self.driver_theta_0 = case.theta_0;
            self.driver_angle = case.theta_0;
            self.solve_at_angle(case.theta_0);
            self.mark_sweep_dirty();
        }
    }

    /// Sync the active load case from the current driver state.
    ///
    /// Called when the user changes driver settings (omega, theta_0, or joint)
    /// so the active load case stays in sync.
    pub fn sync_active_load_case(&mut self) {
        if let Some(case) = self.load_cases.cases.get_mut(self.load_cases.active_index) {
            if let Some(ref joint_id) = self.driver_joint_id {
                case.driver_joint_id = joint_id.clone();
            }
            case.omega = self.driver_omega;
            case.theta_0 = self.driver_theta_0;
        }
    }

    // ── Parametric study ──────────────────────────────────────────────

    /// Run a parametric study: sweep a parameter across a range, compute a full
    /// kinematic/force sweep at each value, and extract the selected metric.
    ///
    /// Clones the blueprint for each parameter value — the user's current
    /// mechanism is not modified.
    pub fn run_parametric_study(&mut self) {
        let Some(ref base_bp) = self.blueprint else { return };
        let config = &self.parametric_config;
        if config.num_steps < 2 {
            return;
        }

        let step_size = (config.max_value - config.min_value) / (config.num_steps - 1) as f64;
        let mut param_values = Vec::with_capacity(config.num_steps);
        let mut metric_values = Vec::with_capacity(config.num_steps);

        for i in 0..config.num_steps {
            let value = config.min_value + i as f64 * step_size;
            param_values.push(value);

            // Clone blueprint and apply parameter
            let mut bp = base_bp.clone();
            let mut omega = self.driver_omega;
            if !Self::set_parameter_on_blueprint(&mut bp, &config.parameter, value, &mut omega) {
                metric_values.push(f64::NAN);
                continue;
            }

            // Build mechanism from modified blueprint
            let Ok(mut mech) = load_mechanism_unbuilt_from_json(&bp) else {
                metric_values.push(f64::NAN);
                continue;
            };
            if mech.build().is_err() {
                metric_values.push(f64::NAN);
                continue;
            }

            // Use current q as initial guess when dimensions match
            let q0 = if self.q.len() == mech.state().n_coords() { self.q.clone() } else { mech.state().make_q() };
            let theta_0 = self.driver_theta_0;
            let (sweep, _) = compute_sweep_data(&mech, &q0, omega, theta_0, self.gravity_magnitude, None);

            // Extract the selected metric
            metric_values.push(config.metric.extract(&sweep));
        }

        self.parametric_result = Some(super::ParametricStudyResult {
            config: config.clone(),
            parameter_values: param_values,
            metric_values,
            selected_sweep: None,
        });
    }

    /// Run a counterbalance optimization: grid search over spring (k, free_length)
    /// to minimize driver torque peak-to-peak variation.
    ///
    /// The baseline torque curve (without the spring) is computed first, then
    /// each (k, free_length) combination is evaluated.
    pub fn run_counterbalance_study(&mut self) {
        use crate::analysis::envelopes::compute_envelope;

        let Some(ref base_bp) = self.blueprint else { return };
        let config = &self.counterbalance_config;
        if config.k_steps < 2 || config.body_a.is_empty() || config.body_b.is_empty() {
            return;
        }

        let omega = self.driver_omega;
        let theta_0 = self.driver_theta_0;

        // Use current q as the initial guess (the geometric guess from sample loading
        // converges much better than all-zeros for the first step).
        let q_init = self.q.clone();

        // 1. Compute baseline (no spring)
        let baseline_sweep = {
            let Ok(mut mech) = load_mechanism_unbuilt_from_json(base_bp) else { return };
            if mech.build().is_err() { return; }
            let q0 = if q_init.len() == mech.state().n_coords() { q_init.clone() } else { mech.state().make_q() };
            let (sweep, _) = compute_sweep_data(&mech, &q0, omega, theta_0, self.gravity_magnitude, None);
            sweep
        };
        let baseline_torques = baseline_sweep.driver_torques.clone().unwrap_or_default();
        let baseline_pp = compute_envelope(&baseline_torques)
            .map(|e| e.peak_to_peak)
            .unwrap_or(0.0);

        // 2. Grid search
        let k_step = if config.k_steps > 1 {
            (config.k_max - config.k_min) / (config.k_steps - 1) as f64
        } else { 0.0 };
        let fl_steps = config.free_length_steps.max(1);
        let fl_step = if fl_steps > 1 {
            (config.free_length_max - config.free_length_min) / (fl_steps - 1) as f64
        } else { 0.0 };

        let mut k_values = Vec::with_capacity(config.k_steps);
        let mut fl_values = Vec::with_capacity(fl_steps);
        for i in 0..config.k_steps {
            k_values.push(config.k_min + i as f64 * k_step);
        }
        for j in 0..fl_steps {
            fl_values.push(config.free_length_min + j as f64 * fl_step);
        }

        let mut grid: Vec<Vec<f64>> = vec![vec![f64::INFINITY; fl_steps]; config.k_steps];
        let mut best_k = config.k_min;
        let mut best_fl = config.free_length_min;
        let mut best_pp = f64::INFINITY;
        let mut best_torques: Option<Vec<f64>> = None;

        for (ki, &k) in k_values.iter().enumerate() {
            for (fi, &fl) in fl_values.iter().enumerate() {
                let mut bp = base_bp.clone();
                // Add a linear spring to the blueprint
                bp.forces.push(ForceElement::LinearSpring(
                    crate::forces::elements::LinearSpringElement {
                        body_a: config.body_a.clone(),
                        point_a: Self::resolve_point_coords(&bp, &config.body_a, &config.point_a),
                        point_a_name: None,
                        body_b: config.body_b.clone(),
                        point_b: Self::resolve_point_coords(&bp, &config.body_b, &config.point_b),
                        point_b_name: None,
                        stiffness: k,
                        free_length: fl,
                    },
                ));

                let Ok(mut mech) = load_mechanism_unbuilt_from_json(&bp) else {
                    grid[ki][fi] = f64::NAN;
                    continue;
                };
                if mech.build().is_err() {
                    grid[ki][fi] = f64::NAN;
                    continue;
                }

                let q0 = if q_init.len() == mech.state().n_coords() { q_init.clone() } else { mech.state().make_q() };
                let (sweep, _) = compute_sweep_data(&mech, &q0, omega, theta_0, self.gravity_magnitude, None);
                let pp = sweep.driver_torques.as_ref()
                    .and_then(|t| compute_envelope(t))
                    .map(|e| e.peak_to_peak)
                    .unwrap_or(f64::INFINITY);
                grid[ki][fi] = pp;

                if pp < best_pp {
                    best_pp = pp;
                    best_k = k;
                    best_fl = fl;
                    best_torques = sweep.driver_torques.clone();
                }
            }
        }

        self.counterbalance_result = Some(super::CounterbalanceResult {
            best_k,
            best_free_length: best_fl,
            best_peak_to_peak: best_pp,
            baseline_peak_to_peak: baseline_pp,
            baseline_torques,
            optimized_torques: best_torques.unwrap_or_default(),
            angles_deg: baseline_sweep.angles_deg,
            grid,
            k_values,
            fl_values,
        });
    }
}
