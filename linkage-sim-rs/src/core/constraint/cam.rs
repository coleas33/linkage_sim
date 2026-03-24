//! Cam-follower joint: follower displacement prescribed by cam profile.

use nalgebra::{DMatrix, DVector, Vector2};

use crate::core::state::State;

use super::trait_def::Constraint;

/// Cam profile definition -- displacement s as a function of cam angle theta.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "profile_type", rename_all = "snake_case")]
pub enum CamProfile {
    /// s(theta) = a[0] + a[1]*theta + a[2]*theta^2 + ...
    Polynomial { coefficients: Vec<f64> },
    /// s(theta) = amplitude * sin(frequency * theta + phase) + offset
    Harmonic {
        amplitude: f64,
        frequency: f64,
        phase: f64,
        offset: f64,
    },
    /// Cubic spline through user-defined (theta, s) points.
    Spline { points: Vec<[f64; 2]> },
}

impl CamProfile {
    /// Evaluate s(theta).
    pub fn evaluate(&self, theta: f64) -> f64 {
        match self {
            Self::Polynomial { coefficients } => {
                let mut s = 0.0;
                let mut t_pow = 1.0;
                for &c in coefficients {
                    s += c * t_pow;
                    t_pow *= theta;
                }
                s
            }
            Self::Harmonic { amplitude, frequency, phase, offset } => {
                amplitude * (frequency * theta + phase).sin() + offset
            }
            Self::Spline { points } => spline_eval(points, theta),
        }
    }

    /// Evaluate ds/dtheta.
    pub fn derivative(&self, theta: f64) -> f64 {
        match self {
            Self::Polynomial { coefficients } => {
                let mut ds = 0.0;
                for (i, &c) in coefficients.iter().enumerate().skip(1) {
                    ds += (i as f64) * c * theta.powi(i as i32 - 1);
                }
                ds
            }
            Self::Harmonic { amplitude, frequency, phase, .. } => {
                amplitude * frequency * (frequency * theta + phase).cos()
            }
            Self::Spline { points } => spline_derivative(points, theta),
        }
    }

    /// Evaluate d^2s/dtheta^2.
    pub fn second_derivative(&self, theta: f64) -> f64 {
        match self {
            Self::Polynomial { coefficients } => {
                let mut d2s = 0.0;
                for (i, &c) in coefficients.iter().enumerate().skip(2) {
                    d2s += (i as f64) * ((i - 1) as f64) * c * theta.powi(i as i32 - 2);
                }
                d2s
            }
            Self::Harmonic { amplitude, frequency, phase, .. } => {
                -amplitude * frequency * frequency * (frequency * theta + phase).sin()
            }
            Self::Spline { points } => spline_second_derivative(points, theta),
        }
    }
}

/// Linear interpolation for spline evaluation (simple piecewise-linear for now).
fn spline_eval(points: &[[f64; 2]], theta: f64) -> f64 {
    if points.is_empty() { return 0.0; }
    if points.len() == 1 { return points[0][1]; }
    // Clamp to range
    if theta <= points[0][0] { return points[0][1]; }
    if theta >= points.last().unwrap()[0] { return points.last().unwrap()[1]; }
    // Binary search for segment
    for w in points.windows(2) {
        if theta >= w[0][0] && theta <= w[1][0] {
            let t = (theta - w[0][0]) / (w[1][0] - w[0][0]).max(1e-15);
            return w[0][1] + t * (w[1][1] - w[0][1]);
        }
    }
    points.last().unwrap()[1]
}

fn spline_derivative(points: &[[f64; 2]], theta: f64) -> f64 {
    if points.len() < 2 { return 0.0; }
    if theta <= points[0][0] || theta >= points.last().unwrap()[0] { return 0.0; }
    for w in points.windows(2) {
        if theta >= w[0][0] && theta <= w[1][0] {
            let dx = (w[1][0] - w[0][0]).max(1e-15);
            return (w[1][1] - w[0][1]) / dx;
        }
    }
    0.0
}

fn spline_second_derivative(_points: &[[f64; 2]], _theta: f64) -> f64 {
    // Piecewise-linear has zero second derivative (except at knots).
    0.0
}

/// Cam-follower joint: follower displacement prescribed by cam profile.
///
/// One constraint equation: projection of follower position onto cam-frame
/// direction minus the profile displacement s(theta_cam) = 0.
#[derive(Debug, Clone)]
pub struct CamFollowerJoint {
    id_: String,
    body_i_id_: String,
    body_j_id_: String,
    pub point_i_local: Vector2<f64>,
    pub point_j_local: Vector2<f64>,
    /// Follower motion direction in body_i (cam) local frame.
    pub follower_dir: Vector2<f64>,
    /// Cam profile: s(theta_cam).
    pub profile: CamProfile,
}

impl Constraint for CamFollowerJoint {
    fn id(&self) -> &str { &self.id_ }
    fn n_equations(&self) -> usize { 1 }
    fn dof_removed(&self) -> usize { 1 }
    fn body_i_id(&self) -> &str { &self.body_i_id_ }
    fn body_j_id(&self) -> &str { &self.body_j_id_ }

    fn constraint(&self, state: &State, q: &DVector<f64>, _t: f64) -> DVector<f64> {
        let theta_i = if state.is_ground(&self.body_i_id_) { 0.0 }
            else { state.get_angle(&self.body_i_id_, q) };
        let a_i = State::rotation_matrix(theta_i);
        let u_global = a_i * self.follower_dir;

        let pt_i = state.body_point_global(&self.body_i_id_, &self.point_i_local, q);
        let pt_j = state.body_point_global(&self.body_j_id_, &self.point_j_local, q);

        let disp = (pt_j - pt_i).dot(&u_global);
        let s = self.profile.evaluate(theta_i);

        DVector::from_element(1, disp - s)
    }

    fn phi_t(&self, _state: &State, _q: &DVector<f64>, _t: f64) -> DVector<f64> {
        DVector::zeros(1)
    }

    fn jacobian(&self, state: &State, q: &DVector<f64>, _t: f64) -> DMatrix<f64> {
        let n = state.n_coords();
        let mut jac = DMatrix::zeros(1, n);

        let theta_i = if state.is_ground(&self.body_i_id_) { 0.0 }
            else { state.get_angle(&self.body_i_id_, q) };
        let a_i = State::rotation_matrix(theta_i);
        let b_i = State::rotation_matrix_derivative(theta_i);
        let u_global = a_i * self.follower_dir;
        let du_dtheta = b_i * self.follower_dir;

        let pt_i = state.body_point_global(&self.body_i_id_, &self.point_i_local, q);
        let pt_j = state.body_point_global(&self.body_j_id_, &self.point_j_local, q);
        let diff = pt_j - pt_i;

        let ds_dtheta = self.profile.derivative(theta_i);

        // Body j terms
        if !state.is_ground(&self.body_j_id_) {
            if let Ok(bj) = state.get_index(&self.body_j_id_) {
                let theta_j = state.get_angle(&self.body_j_id_, q);
                let b_j = State::rotation_matrix_derivative(theta_j);
                // d/d(x_j, y_j): u_global
                jac[(0, bj.x_idx())] = u_global.x;
                jac[(0, bj.y_idx())] = u_global.y;
                // d/d(theta_j): (B_j * s_j_local) . u_global
                let d_ptj_dtheta = b_j * self.point_j_local;
                jac[(0, bj.theta_idx())] = d_ptj_dtheta.dot(&u_global);
            }
        }

        // Body i terms
        if !state.is_ground(&self.body_i_id_) {
            if let Ok(bi) = state.get_index(&self.body_i_id_) {
                // d/d(x_i, y_i): -u_global
                jac[(0, bi.x_idx())] = -u_global.x;
                jac[(0, bi.y_idx())] = -u_global.y;
                // d/d(theta_i): diff . du_dtheta - (B_i * s_i_local) . u_global - ds/dtheta
                let d_pti_dtheta = State::rotation_matrix_derivative(theta_i) * self.point_i_local;
                jac[(0, bi.theta_idx())] =
                    diff.dot(&du_dtheta) - d_pti_dtheta.dot(&u_global) - ds_dtheta;
            }
        }

        jac
    }

    fn gamma(
        &self,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        _t: f64,
    ) -> DVector<f64> {
        // Full gamma for Phi = diff . u - s(theta_i)
        // where diff = pt_j - pt_i, u = A_i . dir.
        //
        // gamma = -d^2Phi/dt^2 |_{q_ddot=0}
        //       = -[d_ddot_iff . u + 2 d_dot_iff . u_dot + diff . u_ddot] + s''(theta_i) theta_dot_i^2
        //
        // At q_ddot = 0:
        //   d_dot_iff = r_dot_j + B_j s_j theta_dot_j - r_dot_i - B_i s_i theta_dot_i
        //   d_ddot_iff = -A_j s_j theta_dot_j^2 + A_i s_i theta_dot_i^2   (centripetal only)
        //   u_dot   = B_i . dir . theta_dot_i
        //   u_ddot  = -A_i . dir . theta_dot_i^2                           (centripetal only)

        let theta_i = if state.is_ground(&self.body_i_id_) { 0.0 }
            else { state.get_angle(&self.body_i_id_, q) };
        let a_i = State::rotation_matrix(theta_i);
        let b_i = State::rotation_matrix_derivative(theta_i);
        let u_global = a_i * self.follower_dir;

        let pt_i = state.body_point_global(&self.body_i_id_, &self.point_i_local, q);
        let pt_j = state.body_point_global(&self.body_j_id_, &self.point_j_local, q);
        let diff = pt_j - pt_i;

        // Velocities for body_i (zero for ground)
        let (theta_dot_i, r_dot_i) = if !state.is_ground(&self.body_i_id_) {
            let bi = state.get_index(&self.body_i_id_).unwrap();
            (
                q_dot[bi.theta_idx()],
                Vector2::new(q_dot[bi.x_idx()], q_dot[bi.y_idx()]),
            )
        } else {
            (0.0, Vector2::zeros())
        };

        // Velocities for body_j (zero for ground)
        let (theta_dot_j, r_dot_j) = if !state.is_ground(&self.body_j_id_) {
            let bj = state.get_index(&self.body_j_id_).unwrap();
            (
                q_dot[bj.theta_idx()],
                Vector2::new(q_dot[bj.x_idx()], q_dot[bj.y_idx()]),
            )
        } else {
            (0.0, Vector2::zeros())
        };

        let theta_j = if state.is_ground(&self.body_j_id_) { 0.0 }
            else { state.get_angle(&self.body_j_id_, q) };
        let b_j = State::rotation_matrix_derivative(theta_j);
        let a_j = State::rotation_matrix(theta_j);

        // d_dot_iff = d(pt_j - pt_i)/dt
        let d_dot = (r_dot_j + b_j * self.point_j_local * theta_dot_j)
            - (r_dot_i + b_i * self.point_i_local * theta_dot_i);

        // d_ddot_iff at q_ddot=0 (centripetal terms only: -A.s.theta_dot^2)
        let d_ddot = -(a_j * self.point_j_local) * theta_dot_j.powi(2)
            + (a_i * self.point_i_local) * theta_dot_i.powi(2);

        // u_dot = B_i . dir . theta_dot_i
        let u_dot = b_i * self.follower_dir * theta_dot_i;

        // u_ddot at q_ddot=0: -A_i . dir . theta_dot_i^2
        let u_ddot = -(a_i * self.follower_dir) * theta_dot_i.powi(2);

        // Profile curvature: s''(theta_i) . theta_dot_i^2
        let d2s = self.profile.second_derivative(theta_i);

        // gamma = -[d_ddot_iff . u + 2 d_dot_iff . u_dot + diff . u_ddot] + s'' theta_dot^2
        let gamma = -(d_ddot.dot(&u_global) + 2.0 * d_dot.dot(&u_dot) + diff.dot(&u_ddot))
            + d2s * theta_dot_i.powi(2);

        DVector::from_element(1, gamma)
    }
}

/// Create a cam-follower joint.
pub fn make_cam_follower(
    id: &str,
    body_i_id: &str,
    body_j_id: &str,
    point_i_local: Vector2<f64>,
    point_j_local: Vector2<f64>,
    follower_dir: Vector2<f64>,
    profile: CamProfile,
) -> CamFollowerJoint {
    let norm = follower_dir.norm();
    let dir = if norm > 1e-12 { follower_dir / norm } else { Vector2::new(1.0, 0.0) };
    CamFollowerJoint {
        id_: id.to_string(),
        body_i_id_: body_i_id.to_string(),
        body_j_id_: body_j_id.to_string(),
        point_i_local,
        point_j_local,
        follower_dir: dir,
        profile,
    }
}
