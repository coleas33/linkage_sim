//! Joint constraint equations, Jacobians, and gamma (acceleration RHS).
//!
//! Each constraint type provides:
//!   - constraint(q, t) → residual vector Φ
//!   - jacobian(q, t)   → Jacobian matrix rows ∂Φ/∂q
//!   - gamma(q, q̇, t)  → acceleration RHS contribution
//!   - n_equations       → number of constraint rows
//!
//! Constraints connect two bodies at specified attachment points.
//! When one body is ground, its terms become constants (no q entries).

use nalgebra::{DMatrix, DVector, Vector2};

use crate::core::state::State;

/// Interface that all joint constraints must implement.
pub trait Constraint {
    fn id(&self) -> &str;
    fn n_equations(&self) -> usize;
    fn dof_removed(&self) -> usize;
    fn body_i_id(&self) -> &str;
    fn body_j_id(&self) -> &str;

    fn constraint(&self, state: &State, q: &DVector<f64>, t: f64) -> DVector<f64>;
    fn phi_t(&self, state: &State, q: &DVector<f64>, t: f64) -> DVector<f64>;
    fn jacobian(&self, state: &State, q: &DVector<f64>, t: f64) -> DMatrix<f64>;
    fn gamma(
        &self,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        t: f64,
    ) -> DVector<f64>;
}

// ---------------------------------------------------------------------------
// Shared translational helpers (used by Revolute and Fixed joints)
// ---------------------------------------------------------------------------

/// Compute the 2-row translational Jacobian block for the constraint
///     Phi = r_i + A_i * s_i - r_j - A_j * s_j = 0
///
/// Writes into rows 0..2 of a pre-allocated Jacobian matrix.
fn translational_jacobian_block(
    state: &State,
    body_i_id: &str,
    body_j_id: &str,
    point_i_local: &Vector2<f64>,
    point_j_local: &Vector2<f64>,
    q: &DVector<f64>,
    jac: &mut DMatrix<f64>,
) {
    if !state.is_ground(body_i_id) {
        let idx_i = state.get_index(body_i_id).unwrap();
        jac[(0, idx_i.x_idx())] = 1.0;
        jac[(1, idx_i.y_idx())] = 1.0;
        let b_si = state.body_point_global_derivative(body_i_id, point_i_local, q);
        jac[(0, idx_i.theta_idx())] = b_si.x;
        jac[(1, idx_i.theta_idx())] = b_si.y;
    }

    if !state.is_ground(body_j_id) {
        let idx_j = state.get_index(body_j_id).unwrap();
        jac[(0, idx_j.x_idx())] = -1.0;
        jac[(1, idx_j.y_idx())] = -1.0;
        let b_sj = state.body_point_global_derivative(body_j_id, point_j_local, q);
        jac[(0, idx_j.theta_idx())] = -b_sj.x;
        jac[(1, idx_j.theta_idx())] = -b_sj.y;
    }
}

/// Compute the 2-element translational gamma (centripetal terms) for the constraint
///     Phi = r_i + A_i * s_i - r_j - A_j * s_j = 0
///
/// gamma_trans = A_i * s_i * theta_dot_i^2 - A_j * s_j * theta_dot_j^2
fn translational_gamma(
    state: &State,
    body_i_id: &str,
    body_j_id: &str,
    point_i_local: &Vector2<f64>,
    point_j_local: &Vector2<f64>,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> Vector2<f64> {
    let mut result = Vector2::zeros();

    if !state.is_ground(body_i_id) {
        let theta_i = state.get_angle(body_i_id, q);
        let idx_i = state.get_index(body_i_id).unwrap();
        let theta_dot_i = q_dot[idx_i.theta_idx()];
        let a_i = State::rotation_matrix(theta_i);
        result += (a_i * point_i_local) * theta_dot_i.powi(2);
    }

    if !state.is_ground(body_j_id) {
        let theta_j = state.get_angle(body_j_id, q);
        let idx_j = state.get_index(body_j_id).unwrap();
        let theta_dot_j = q_dot[idx_j.theta_idx()];
        let a_j = State::rotation_matrix(theta_j);
        result -= (a_j * point_j_local) * theta_dot_j.powi(2);
    }

    result
}

// ---------------------------------------------------------------------------
// Revolute joint
// ---------------------------------------------------------------------------

/// Revolute joint: constrains two attachment points to be coincident.
///
/// Removes 2 translational DOF. Allows relative rotation.
///
/// Constraint (2 equations):
///     Φ = rᵢ + Aᵢ·sᵢ − rⱼ − Aⱼ·sⱼ = 0
#[derive(Debug, Clone)]
pub struct RevoluteJoint {
    id_: String,
    body_i_id_: String,
    body_j_id_: String,
    point_i_local: Vector2<f64>,
    point_j_local: Vector2<f64>,
}

impl RevoluteJoint {
    pub fn point_i_local(&self) -> &Vector2<f64> {
        &self.point_i_local
    }
    pub fn point_j_local(&self) -> &Vector2<f64> {
        &self.point_j_local
    }
}

impl Constraint for RevoluteJoint {
    fn id(&self) -> &str {
        &self.id_
    }
    fn n_equations(&self) -> usize {
        2
    }
    fn dof_removed(&self) -> usize {
        2
    }
    fn body_i_id(&self) -> &str {
        &self.body_i_id_
    }
    fn body_j_id(&self) -> &str {
        &self.body_j_id_
    }

    fn constraint(&self, state: &State, q: &DVector<f64>, _t: f64) -> DVector<f64> {
        let global_i = state.body_point_global(&self.body_i_id_, &self.point_i_local, q);
        let global_j = state.body_point_global(&self.body_j_id_, &self.point_j_local, q);
        let diff = global_i - global_j;
        DVector::from_column_slice(&[diff.x, diff.y])
    }

    fn phi_t(&self, _state: &State, _q: &DVector<f64>, _t: f64) -> DVector<f64> {
        DVector::zeros(2)
    }

    fn jacobian(&self, state: &State, q: &DVector<f64>, _t: f64) -> DMatrix<f64> {
        let n = state.n_coords();
        let mut jac = DMatrix::zeros(2, n);
        translational_jacobian_block(
            state,
            &self.body_i_id_,
            &self.body_j_id_,
            &self.point_i_local,
            &self.point_j_local,
            q,
            &mut jac,
        );
        jac
    }

    fn gamma(
        &self,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        _t: f64,
    ) -> DVector<f64> {
        let g = translational_gamma(
            state,
            &self.body_i_id_,
            &self.body_j_id_,
            &self.point_i_local,
            &self.point_j_local,
            q,
            q_dot,
        );
        DVector::from_column_slice(&[g.x, g.y])
    }
}

/// Create a revolute joint between two bodies at specified attachment points.
pub fn make_revolute_joint(
    joint_id: &str,
    body_i_id: &str,
    point_i_local: Vector2<f64>,
    body_j_id: &str,
    point_j_local: Vector2<f64>,
) -> RevoluteJoint {
    RevoluteJoint {
        id_: joint_id.to_string(),
        body_i_id_: body_i_id.to_string(),
        body_j_id_: body_j_id.to_string(),
        point_i_local,
        point_j_local,
    }
}

// ---------------------------------------------------------------------------
// Fixed joint
// ---------------------------------------------------------------------------

/// Fixed joint: constrains all relative motion to zero.
///
/// Removes 3 DOF (2 translational + 1 rotational).
///
/// Constraint (3 equations):
///     Φ[0:2] = rᵢ + Aᵢ·sᵢ − rⱼ − Aⱼ·sⱼ = 0   (coincident points)
///     Φ[2]   = θⱼ − θᵢ − Δθ₀ = 0                 (locked relative angle)
#[derive(Debug, Clone)]
pub struct FixedJoint {
    id_: String,
    body_i_id_: String,
    body_j_id_: String,
    point_i_local: Vector2<f64>,
    point_j_local: Vector2<f64>,
    delta_theta_0: f64,
}

impl FixedJoint {
    pub fn point_i_local(&self) -> &Vector2<f64> {
        &self.point_i_local
    }
    pub fn point_j_local(&self) -> &Vector2<f64> {
        &self.point_j_local
    }
    pub fn delta_theta_0(&self) -> f64 {
        self.delta_theta_0
    }
}

impl Constraint for FixedJoint {
    fn id(&self) -> &str {
        &self.id_
    }
    fn n_equations(&self) -> usize {
        3
    }
    fn dof_removed(&self) -> usize {
        3
    }
    fn body_i_id(&self) -> &str {
        &self.body_i_id_
    }
    fn body_j_id(&self) -> &str {
        &self.body_j_id_
    }

    fn constraint(&self, state: &State, q: &DVector<f64>, _t: f64) -> DVector<f64> {
        let global_i = state.body_point_global(&self.body_i_id_, &self.point_i_local, q);
        let global_j = state.body_point_global(&self.body_j_id_, &self.point_j_local, q);
        let diff = global_i - global_j;

        let theta_i = state.get_angle(&self.body_i_id_, q);
        let theta_j = state.get_angle(&self.body_j_id_, q);

        DVector::from_column_slice(&[diff.x, diff.y, theta_j - theta_i - self.delta_theta_0])
    }

    fn phi_t(&self, _state: &State, _q: &DVector<f64>, _t: f64) -> DVector<f64> {
        DVector::zeros(3)
    }

    fn jacobian(&self, state: &State, q: &DVector<f64>, _t: f64) -> DMatrix<f64> {
        let n = state.n_coords();
        let mut jac = DMatrix::zeros(3, n);

        // Rows 0-1: translational block (shared with revolute)
        translational_jacobian_block(
            state,
            &self.body_i_id_,
            &self.body_j_id_,
            &self.point_i_local,
            &self.point_j_local,
            q,
            &mut jac,
        );

        // Row 2: rotation lock
        if !state.is_ground(&self.body_i_id_) {
            let idx_i = state.get_index(&self.body_i_id_).unwrap();
            jac[(2, idx_i.theta_idx())] = -1.0;
        }
        if !state.is_ground(&self.body_j_id_) {
            let idx_j = state.get_index(&self.body_j_id_).unwrap();
            jac[(2, idx_j.theta_idx())] += 1.0;
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
        let pos_gamma = translational_gamma(
            state,
            &self.body_i_id_,
            &self.body_j_id_,
            &self.point_i_local,
            &self.point_j_local,
            q,
            q_dot,
        );

        // gamma[2] = 0 (rotation constraint is linear in theta)
        DVector::from_column_slice(&[pos_gamma.x, pos_gamma.y, 0.0])
    }
}

/// Create a fixed joint that locks all relative motion.
pub fn make_fixed_joint(
    joint_id: &str,
    body_i_id: &str,
    point_i_local: Vector2<f64>,
    body_j_id: &str,
    point_j_local: Vector2<f64>,
    delta_theta_0: f64,
) -> FixedJoint {
    FixedJoint {
        id_: joint_id.to_string(),
        body_i_id_: body_i_id.to_string(),
        body_j_id_: body_j_id.to_string(),
        point_i_local,
        point_j_local,
        delta_theta_0,
    }
}

// ---------------------------------------------------------------------------
// Prismatic joint
// ---------------------------------------------------------------------------

/// Prismatic (slider) joint: allows translation along one axis, locks rotation.
///
/// Removes 2 DOF (1 perpendicular translation + 1 rotation).
///
/// Constraint (2 equations):
///     Φ[0] = n̂ᵢ_global · d = 0        (no perpendicular displacement)
///     Φ[1] = θⱼ − θᵢ − Δθ₀ = 0        (no relative rotation)
#[derive(Debug, Clone)]
pub struct PrismaticJoint {
    id_: String,
    body_i_id_: String,
    body_j_id_: String,
    point_i_local: Vector2<f64>,
    point_j_local: Vector2<f64>,
    #[allow(dead_code)]
    axis_local_i: Vector2<f64>,
    n_hat_local_i: Vector2<f64>,
    delta_theta_0: f64,
}

impl PrismaticJoint {
    pub fn point_i_local(&self) -> &Vector2<f64> {
        &self.point_i_local
    }
    pub fn point_j_local(&self) -> &Vector2<f64> {
        &self.point_j_local
    }
    pub fn axis_local_i(&self) -> &Vector2<f64> {
        &self.axis_local_i
    }
    pub fn delta_theta_0(&self) -> f64 {
        self.delta_theta_0
    }
}

impl Constraint for PrismaticJoint {
    fn id(&self) -> &str {
        &self.id_
    }
    fn n_equations(&self) -> usize {
        2
    }
    fn dof_removed(&self) -> usize {
        2
    }
    fn body_i_id(&self) -> &str {
        &self.body_i_id_
    }
    fn body_j_id(&self) -> &str {
        &self.body_j_id_
    }

    fn constraint(&self, state: &State, q: &DVector<f64>, _t: f64) -> DVector<f64> {
        let theta_i = state.get_angle(&self.body_i_id_, q);
        let a_i = State::rotation_matrix(theta_i);
        let n_hat_g = a_i * self.n_hat_local_i;

        let pt_i_g = state.body_point_global(&self.body_i_id_, &self.point_i_local, q);
        let pt_j_g = state.body_point_global(&self.body_j_id_, &self.point_j_local, q);
        let d = pt_j_g - pt_i_g;

        let theta_j = state.get_angle(&self.body_j_id_, q);

        DVector::from_column_slice(&[
            n_hat_g.dot(&d),
            theta_j - theta_i - self.delta_theta_0,
        ])
    }

    fn phi_t(&self, _state: &State, _q: &DVector<f64>, _t: f64) -> DVector<f64> {
        DVector::zeros(2)
    }

    fn jacobian(&self, state: &State, q: &DVector<f64>, _t: f64) -> DMatrix<f64> {
        let n = state.n_coords();
        let mut jac = DMatrix::zeros(2, n);

        let theta_i = state.get_angle(&self.body_i_id_, q);
        let a_i = State::rotation_matrix(theta_i);
        let b_i = State::rotation_matrix_derivative(theta_i);
        let n_hat_g = a_i * self.n_hat_local_i;
        let b_n = b_i * self.n_hat_local_i;

        let pt_i_g = state.body_point_global(&self.body_i_id_, &self.point_i_local, q);
        let pt_j_g = state.body_point_global(&self.body_j_id_, &self.point_j_local, q);
        let d = pt_j_g - pt_i_g;

        if !state.is_ground(&self.body_i_id_) {
            let idx_i = state.get_index(&self.body_i_id_).unwrap();
            // Row 0: perpendicular constraint
            jac[(0, idx_i.x_idx())] = -n_hat_g.x;
            jac[(0, idx_i.y_idx())] = -n_hat_g.y;
            let b_si = b_i * self.point_i_local;
            jac[(0, idx_i.theta_idx())] = b_n.dot(&d) - n_hat_g.dot(&b_si);
            // Row 1: rotation constraint
            jac[(1, idx_i.theta_idx())] = -1.0;
        }

        if !state.is_ground(&self.body_j_id_) {
            let idx_j = state.get_index(&self.body_j_id_).unwrap();
            let theta_j = state.get_angle(&self.body_j_id_, q);
            let b_j = State::rotation_matrix_derivative(theta_j);
            // Row 0: perpendicular constraint
            jac[(0, idx_j.x_idx())] = n_hat_g.x;
            jac[(0, idx_j.y_idx())] = n_hat_g.y;
            let b_sj = b_j * self.point_j_local;
            jac[(0, idx_j.theta_idx())] = n_hat_g.dot(&b_sj);
            // Row 1: rotation constraint
            jac[(1, idx_j.theta_idx())] += 1.0;
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
        let theta_i = state.get_angle(&self.body_i_id_, q);
        let theta_j = state.get_angle(&self.body_j_id_, q);
        let a_i = State::rotation_matrix(theta_i);
        let b_i = State::rotation_matrix_derivative(theta_i);
        let a_j = State::rotation_matrix(theta_j);
        let b_j = State::rotation_matrix_derivative(theta_j);

        let n_hat_g = a_i * self.n_hat_local_i;
        let b_n = b_i * self.n_hat_local_i;

        let pt_i_g = state.body_point_global(&self.body_i_id_, &self.point_i_local, q);
        let pt_j_g = state.body_point_global(&self.body_j_id_, &self.point_j_local, q);
        let d = pt_j_g - pt_i_g;

        // Velocities (zero for ground bodies)
        let (theta_dot_i, r_dot_i) = if !state.is_ground(&self.body_i_id_) {
            let idx_i = state.get_index(&self.body_i_id_).unwrap();
            (
                q_dot[idx_i.theta_idx()],
                Vector2::new(q_dot[idx_i.x_idx()], q_dot[idx_i.y_idx()]),
            )
        } else {
            (0.0, Vector2::zeros())
        };

        let (theta_dot_j, r_dot_j) = if !state.is_ground(&self.body_j_id_) {
            let idx_j = state.get_index(&self.body_j_id_).unwrap();
            (
                q_dot[idx_j.theta_idx()],
                Vector2::new(q_dot[idx_j.x_idx()], q_dot[idx_j.y_idx()]),
            )
        } else {
            (0.0, Vector2::zeros())
        };

        // d_dot = velocity of d vector
        let d_dot = (r_dot_j + (b_j * self.point_j_local) * theta_dot_j)
            - (r_dot_i + (b_i * self.point_i_local) * theta_dot_i);

        // γ[0]: velocity-quadratic terms from Φ̈[0]
        let gamma_0 = n_hat_g.dot(&d) * theta_dot_i.powi(2)
            - 2.0 * theta_dot_i * b_n.dot(&d_dot)
            + n_hat_g.dot(&(a_j * self.point_j_local)) * theta_dot_j.powi(2)
            - n_hat_g.dot(&(a_i * self.point_i_local)) * theta_dot_i.powi(2);

        // γ[1] = 0 (rotation constraint is linear in θ)
        DVector::from_column_slice(&[gamma_0, 0.0])
    }
}

/// Create a prismatic joint that allows sliding along one axis.
pub fn make_prismatic_joint(
    joint_id: &str,
    body_i_id: &str,
    point_i_local: Vector2<f64>,
    body_j_id: &str,
    point_j_local: Vector2<f64>,
    axis_local_i: Vector2<f64>,
    delta_theta_0: f64,
) -> Result<PrismaticJoint, &'static str> {
    let norm = axis_local_i.norm();
    if norm < 1e-12 {
        return Err("axis_local_i must be non-zero");
    }
    let axis = axis_local_i / norm;
    // Perpendicular: rotate axis by 90° CCW → n̂ = (−e_y, e_x)
    let n_hat = Vector2::new(-axis.y, axis.x);

    Ok(PrismaticJoint {
        id_: joint_id.to_string(),
        body_i_id_: body_i_id.to_string(),
        body_j_id_: body_j_id.to_string(),
        point_i_local,
        point_j_local,
        axis_local_i: axis,
        n_hat_local_i: n_hat,
        delta_theta_0,
    })
}

// ── Cam-follower joint ────────────────────────────────────────────────────────

/// Cam profile definition — displacement s as a function of cam angle theta.
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

    /// Evaluate d²s/dtheta².
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
        // Full gamma for Φ = diff · u − s(θ_i)
        // where diff = pt_j − pt_i, u = A_i · dir.
        //
        // gamma = −d²Φ/dt² |_{q̈=0}
        //       = −[d̈iff · u + 2 ḋiff · u̇ + diff · ü] + s″(θ_i) θ̇_i²
        //
        // At q̈ = 0:
        //   ḋiff = ṙ_j + B_j s_j θ̇_j − ṙ_i − B_i s_i θ̇_i
        //   d̈iff = −A_j s_j θ̇_j² + A_i s_i θ̇_i²   (centripetal only)
        //   u̇   = B_i · dir · θ̇_i
        //   ü   = −A_i · dir · θ̇_i²                  (centripetal only)

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

        // ḋiff = d(pt_j − pt_i)/dt
        let d_dot = (r_dot_j + b_j * self.point_j_local * theta_dot_j)
            - (r_dot_i + b_i * self.point_i_local * theta_dot_i);

        // d̈iff at q̈=0 (centripetal terms only: −A·s·θ̇²)
        let d_ddot = -(a_j * self.point_j_local) * theta_dot_j.powi(2)
            + (a_i * self.point_i_local) * theta_dot_i.powi(2);

        // u̇ = B_i · dir · θ̇_i
        let u_dot = b_i * self.follower_dir * theta_dot_i;

        // ü at q̈=0: −A_i · dir · θ̇_i²
        let u_ddot = -(a_i * self.follower_dir) * theta_dot_i.powi(2);

        // Profile curvature: s″(θ_i) · θ̇_i²
        let d2s = self.profile.second_derivative(theta_i);

        // gamma = −[d̈iff · u + 2 ḋiff · u̇ + diff · ü] + s″θ̇²
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

/// Enum wrapper for dynamic dispatch of constraint types.
#[derive(Debug, Clone)]
pub enum JointConstraint {
    Revolute(RevoluteJoint),
    Fixed(FixedJoint),
    Prismatic(PrismaticJoint),
    CamFollower(CamFollowerJoint),
}

impl JointConstraint {
    pub fn point_i_local(&self) -> Vector2<f64> {
        match self {
            Self::Revolute(j) => j.point_i_local,
            Self::Fixed(j) => j.point_i_local,
            Self::Prismatic(j) => j.point_i_local,
            Self::CamFollower(j) => j.point_i_local,
        }
    }

    pub fn point_j_local(&self) -> Vector2<f64> {
        match self {
            Self::Revolute(j) => j.point_j_local,
            Self::Fixed(j) => j.point_j_local,
            Self::Prismatic(j) => j.point_j_local,
            Self::CamFollower(j) => j.point_j_local,
        }
    }

    pub fn is_revolute(&self) -> bool {
        matches!(self, Self::Revolute(_))
    }

    pub fn is_prismatic(&self) -> bool {
        matches!(self, Self::Prismatic(_))
    }

    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }

    pub fn is_cam_follower(&self) -> bool {
        matches!(self, Self::CamFollower(_))
    }
}

impl Constraint for JointConstraint {
    fn id(&self) -> &str {
        match self {
            Self::Revolute(j) => j.id(),
            Self::Fixed(j) => j.id(),
            Self::Prismatic(j) => j.id(),
            Self::CamFollower(j) => j.id(),
        }
    }
    fn n_equations(&self) -> usize {
        match self {
            Self::Revolute(j) => j.n_equations(),
            Self::Fixed(j) => j.n_equations(),
            Self::Prismatic(j) => j.n_equations(),
            Self::CamFollower(j) => j.n_equations(),
        }
    }
    fn dof_removed(&self) -> usize {
        match self {
            Self::Revolute(j) => j.dof_removed(),
            Self::Fixed(j) => j.dof_removed(),
            Self::Prismatic(j) => j.dof_removed(),
            Self::CamFollower(j) => j.dof_removed(),
        }
    }
    fn body_i_id(&self) -> &str {
        match self {
            Self::Revolute(j) => j.body_i_id(),
            Self::Fixed(j) => j.body_i_id(),
            Self::Prismatic(j) => j.body_i_id(),
            Self::CamFollower(j) => j.body_i_id(),
        }
    }
    fn body_j_id(&self) -> &str {
        match self {
            Self::Revolute(j) => j.body_j_id(),
            Self::Fixed(j) => j.body_j_id(),
            Self::Prismatic(j) => j.body_j_id(),
            Self::CamFollower(j) => j.body_j_id(),
        }
    }
    fn constraint(&self, state: &State, q: &DVector<f64>, t: f64) -> DVector<f64> {
        match self {
            Self::Revolute(j) => j.constraint(state, q, t),
            Self::Fixed(j) => j.constraint(state, q, t),
            Self::Prismatic(j) => j.constraint(state, q, t),
            Self::CamFollower(j) => j.constraint(state, q, t),
        }
    }
    fn phi_t(&self, state: &State, q: &DVector<f64>, t: f64) -> DVector<f64> {
        match self {
            Self::Revolute(j) => j.phi_t(state, q, t),
            Self::Fixed(j) => j.phi_t(state, q, t),
            Self::Prismatic(j) => j.phi_t(state, q, t),
            Self::CamFollower(j) => j.phi_t(state, q, t),
        }
    }
    fn jacobian(&self, state: &State, q: &DVector<f64>, t: f64) -> DMatrix<f64> {
        match self {
            Self::Revolute(j) => j.jacobian(state, q, t),
            Self::Fixed(j) => j.jacobian(state, q, t),
            Self::Prismatic(j) => j.jacobian(state, q, t),
            Self::CamFollower(j) => j.jacobian(state, q, t),
        }
    }
    fn gamma(
        &self,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        t: f64,
    ) -> DVector<f64> {
        match self {
            Self::Revolute(j) => j.gamma(state, q, q_dot, t),
            Self::Fixed(j) => j.gamma(state, q, q_dot, t),
            Self::Prismatic(j) => j.gamma(state, q, q_dot, t),
            Self::CamFollower(j) => j.gamma(state, q, q_dot, t),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    /// Build a simple 2-body state: crank (body 0) + coupler (body 1).
    fn two_body_state() -> State {
        let mut state = State::new();
        state.register_body("crank").unwrap();
        state.register_body("coupler").unwrap();
        state
    }

    #[test]
    fn revolute_constraint_zero_at_coincident_points() {
        let state = two_body_state();
        let mut q = state.make_q();
        // Place crank at origin, angle 0 → point (0.1, 0) in global
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.0);
        // Place coupler so its local (0,0) is at (0.1, 0) in global
        state.set_pose("coupler", &mut q, 0.1, 0.0, 0.0);

        let joint = make_revolute_joint(
            "J1",
            "crank",
            Vector2::new(0.1, 0.0),
            "coupler",
            Vector2::new(0.0, 0.0),
        );

        let phi = joint.constraint(&state, &q, 0.0);
        assert_abs_diff_eq!(phi[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(phi[1], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn revolute_constraint_nonzero_when_separated() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("coupler", &mut q, 0.2, 0.0, 0.0); // too far

        let joint = make_revolute_joint(
            "J1",
            "crank",
            Vector2::new(0.1, 0.0),
            "coupler",
            Vector2::new(0.0, 0.0),
        );

        let phi = joint.constraint(&state, &q, 0.0);
        assert_abs_diff_eq!(phi[0], -0.1, epsilon = 1e-14);
    }

    #[test]
    fn revolute_jacobian_finite_difference() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.5);
        state.set_pose("coupler", &mut q, 0.1, 0.05, 1.0);

        let joint = make_revolute_joint(
            "J1",
            "crank",
            Vector2::new(0.1, 0.0),
            "coupler",
            Vector2::new(0.0, 0.0),
        );

        let jac_analytical = joint.jacobian(&state, &q, 0.0);

        // Finite-difference Jacobian
        let eps = 1e-7;
        let n = state.n_coords();
        let mut jac_fd = DMatrix::zeros(2, n);
        for col in 0..n {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[col] += eps;
            q_minus[col] -= eps;
            let phi_plus = joint.constraint(&state, &q_plus, 0.0);
            let phi_minus = joint.constraint(&state, &q_minus, 0.0);
            for row in 0..2 {
                jac_fd[(row, col)] = (phi_plus[row] - phi_minus[row]) / (2.0 * eps);
            }
        }

        for row in 0..2 {
            for col in 0..n {
                assert_abs_diff_eq!(
                    jac_analytical[(row, col)],
                    jac_fd[(row, col)],
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn revolute_ground_to_body() {
        let mut state = State::new();
        state.register_body("crank").unwrap();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, PI / 4.0);

        let joint = make_revolute_joint(
            "J_ground",
            "ground",
            Vector2::new(0.0, 0.0),
            "crank",
            Vector2::new(0.0, 0.0),
        );

        let phi = joint.constraint(&state, &q, 0.0);
        assert_abs_diff_eq!(phi[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(phi[1], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn fixed_joint_constraint_zero_at_locked_config() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("coupler", &mut q, 0.1, 0.0, 0.0);

        let joint = make_fixed_joint(
            "F1",
            "crank",
            Vector2::new(0.1, 0.0),
            "coupler",
            Vector2::new(0.0, 0.0),
            0.0,
        );

        let phi = joint.constraint(&state, &q, 0.0);
        assert_abs_diff_eq!(phi[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(phi[1], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(phi[2], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn fixed_joint_jacobian_finite_difference() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.5);
        state.set_pose("coupler", &mut q, 0.1, 0.05, 1.0);

        let joint = make_fixed_joint(
            "F1",
            "crank",
            Vector2::new(0.1, 0.0),
            "coupler",
            Vector2::new(0.0, 0.0),
            0.5,
        );

        let jac_analytical = joint.jacobian(&state, &q, 0.0);

        let eps = 1e-7;
        let n = state.n_coords();
        let mut jac_fd = DMatrix::zeros(3, n);
        for col in 0..n {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[col] += eps;
            q_minus[col] -= eps;
            let phi_plus = joint.constraint(&state, &q_plus, 0.0);
            let phi_minus = joint.constraint(&state, &q_minus, 0.0);
            for row in 0..3 {
                jac_fd[(row, col)] = (phi_plus[row] - phi_minus[row]) / (2.0 * eps);
            }
        }

        for row in 0..3 {
            for col in 0..n {
                assert_abs_diff_eq!(
                    jac_analytical[(row, col)],
                    jac_fd[(row, col)],
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn prismatic_constraint_zero_on_axis() {
        let mut state = State::new();
        state.register_body("slider").unwrap();
        let mut q = state.make_q();
        // Slider at (0.5, 0) — on the x-axis (slide direction from ground)
        state.set_pose("slider", &mut q, 0.5, 0.0, 0.0);

        let joint = make_prismatic_joint(
            "P1",
            "ground",
            Vector2::new(0.0, 0.0),
            "slider",
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        let phi = joint.constraint(&state, &q, 0.0);
        assert_abs_diff_eq!(phi[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(phi[1], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn prismatic_jacobian_finite_difference() {
        let mut state = State::new();
        state.register_body("slider").unwrap();
        let mut q = state.make_q();
        state.set_pose("slider", &mut q, 0.5, 0.01, 0.0);

        let joint = make_prismatic_joint(
            "P1",
            "ground",
            Vector2::new(0.0, 0.0),
            "slider",
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        let jac_analytical = joint.jacobian(&state, &q, 0.0);

        let eps = 1e-7;
        let n = state.n_coords();
        let mut jac_fd = DMatrix::zeros(2, n);
        for col in 0..n {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[col] += eps;
            q_minus[col] -= eps;
            let phi_plus = joint.constraint(&state, &q_plus, 0.0);
            let phi_minus = joint.constraint(&state, &q_minus, 0.0);
            for row in 0..2 {
                jac_fd[(row, col)] = (phi_plus[row] - phi_minus[row]) / (2.0 * eps);
            }
        }

        for row in 0..2 {
            for col in 0..n {
                assert_abs_diff_eq!(
                    jac_analytical[(row, col)],
                    jac_fd[(row, col)],
                    epsilon = 1e-6
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // Gamma finite-difference helper
    // -----------------------------------------------------------------

    /// Compute gamma via finite differences on phi_dot = Phi_q * q_dot + Phi_t.
    ///
    /// At time t with state (q, q_dot), advance to q_plus = q + q_dot * dt,
    /// then gamma_fd = -(phi_dot_plus - phi_dot) / dt.
    fn gamma_fd<C: Constraint>(
        joint: &C,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        t: f64,
    ) -> DVector<f64> {
        let dt = 1e-7;

        let jac = joint.jacobian(state, q, t);
        let phi_t = joint.phi_t(state, q, t);
        let phi_dot = &jac * q_dot + &phi_t;

        let q_plus = q + q_dot * dt;
        let t_plus = t + dt;
        let jac_plus = joint.jacobian(state, &q_plus, t_plus);
        let phi_t_plus = joint.phi_t(state, &q_plus, t_plus);
        let phi_dot_plus = &jac_plus * q_dot + &phi_t_plus;

        -(&phi_dot_plus - &phi_dot) / dt
    }

    fn assert_gamma_matches_fd<C: Constraint>(
        joint: &C,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        t: f64,
        tol: f64,
    ) {
        let gamma_analytical = joint.gamma(state, q, q_dot, t);
        let gamma_numerical = gamma_fd(joint, state, q, q_dot, t);
        let n_eq = joint.n_equations();
        for i in 0..n_eq {
            assert_abs_diff_eq!(
                gamma_analytical[i],
                gamma_numerical[i],
                epsilon = tol
            );
        }
    }

    // -----------------------------------------------------------------
    // Revolute gamma FD tests
    // -----------------------------------------------------------------

    #[test]
    fn revolute_gamma_fd_two_bodies() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.1, 0.2, 0.7);
        state.set_pose("coupler", &mut q, 0.3, -0.1, 1.2);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.1; // crank x_dot
        q_dot[1] = -0.2; // crank y_dot
        q_dot[2] = 3.0; // crank theta_dot
        q_dot[3] = -0.05; // coupler x_dot
        q_dot[4] = 0.15; // coupler y_dot
        q_dot[5] = -2.0; // coupler theta_dot

        let joint = make_revolute_joint(
            "J1",
            "crank",
            Vector2::new(0.1, 0.05),
            "coupler",
            Vector2::new(-0.03, 0.02),
        );

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn revolute_gamma_fd_ground_to_body() {
        let mut state = State::new();
        state.register_body("crank").unwrap();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 1.0);

        let mut q_dot = state.make_q();
        q_dot[2] = 5.0; // theta_dot

        let joint = make_revolute_joint(
            "J_gnd",
            "ground",
            Vector2::new(0.0, 0.0),
            "crank",
            Vector2::new(0.0, 0.0),
        );

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    // -----------------------------------------------------------------
    // Fixed gamma FD tests
    // -----------------------------------------------------------------

    #[test]
    fn fixed_gamma_fd_two_bodies() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.1, 0.2, 0.7);
        state.set_pose("coupler", &mut q, 0.3, -0.1, 1.2);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.1;
        q_dot[1] = -0.2;
        q_dot[2] = 3.0;
        q_dot[3] = -0.05;
        q_dot[4] = 0.15;
        q_dot[5] = -2.0;

        let joint = make_fixed_joint(
            "F1",
            "crank",
            Vector2::new(0.1, 0.05),
            "coupler",
            Vector2::new(-0.03, 0.02),
            0.5,
        );

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    // -----------------------------------------------------------------
    // Prismatic gamma FD tests
    // -----------------------------------------------------------------

    #[test]
    fn prismatic_gamma_fd_x_axis_ground_to_slider() {
        let mut state = State::new();
        state.register_body("slider").unwrap();
        let mut q = state.make_q();
        state.set_pose("slider", &mut q, 0.5, 0.01, 0.0);

        let mut q_dot = state.make_q();
        q_dot[0] = 1.0; // slider x_dot
        q_dot[1] = 0.05; // slider y_dot
        q_dot[2] = 0.0; // slider theta_dot (locked by constraint)

        let joint = make_prismatic_joint(
            "P_x",
            "ground",
            Vector2::new(0.0, 0.0),
            "slider",
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn prismatic_gamma_fd_y_axis_ground_to_slider() {
        let mut state = State::new();
        state.register_body("slider").unwrap();
        let mut q = state.make_q();
        state.set_pose("slider", &mut q, 0.02, 0.8, 0.0);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.03;
        q_dot[1] = 2.0;
        q_dot[2] = 0.0;

        let joint = make_prismatic_joint(
            "P_y",
            "ground",
            Vector2::new(0.0, 0.0),
            "slider",
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 1.0),
            0.0,
        )
        .unwrap();

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn prismatic_gamma_fd_rotated_rail() {
        // Body i (rail) is at a nonzero angle, so the axis is rotated in world space.
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, PI / 6.0); // rail at 30 deg
        state.set_pose("coupler", &mut q, 0.5, 0.3, PI / 6.0); // slider

        let mut q_dot = state.make_q();
        q_dot[0] = 0.0; // crank x_dot
        q_dot[1] = 0.0; // crank y_dot
        q_dot[2] = 0.0; // crank theta_dot (rail stationary)
        q_dot[3] = 0.5; // coupler x_dot
        q_dot[4] = 0.3; // coupler y_dot
        q_dot[5] = 0.0; // coupler theta_dot (locked)

        let joint = make_prismatic_joint(
            "P_rot",
            "crank",
            Vector2::new(0.0, 0.0),
            "coupler",
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn prismatic_gamma_fd_moving_parent_body() {
        // Body i (rail) is not ground and has translational velocity.
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.1, 0.2, 0.3);
        state.set_pose("coupler", &mut q, 0.6, 0.25, 0.3);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.5; // crank x_dot
        q_dot[1] = -0.3; // crank y_dot
        q_dot[2] = 0.0; // crank theta_dot
        q_dot[3] = 1.0; // coupler x_dot
        q_dot[4] = -0.1; // coupler y_dot
        q_dot[5] = 0.0; // coupler theta_dot

        let joint = make_prismatic_joint(
            "P_move",
            "crank",
            Vector2::new(0.05, 0.02),
            "coupler",
            Vector2::new(-0.01, 0.0),
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn prismatic_gamma_fd_nonzero_rail_angular_velocity() {
        // The rail-carrying body has nonzero angular velocity — most complex case.
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.1, 0.2, 0.8);
        state.set_pose("coupler", &mut q, 0.5, 0.4, 0.8);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.3; // crank x_dot
        q_dot[1] = -0.1; // crank y_dot
        q_dot[2] = 2.5; // crank theta_dot (nonzero angular velocity!)
        q_dot[3] = 0.8; // coupler x_dot
        q_dot[4] = 0.2; // coupler y_dot
        q_dot[5] = 2.5; // coupler theta_dot (locked, same as crank)

        let joint = make_prismatic_joint(
            "P_omega",
            "crank",
            Vector2::new(0.05, 0.03),
            "coupler",
            Vector2::new(-0.02, 0.01),
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    // -----------------------------------------------------------------
    // Cam-follower gamma FD tests
    // -----------------------------------------------------------------

    #[test]
    fn cam_follower_gamma_fd_two_bodies() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.1, 0.2, 0.7);
        state.set_pose("coupler", &mut q, 0.4, 0.3, 1.1);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.15; // crank x_dot
        q_dot[1] = -0.1; // crank y_dot
        q_dot[2] = 3.5; // crank theta_dot
        q_dot[3] = -0.2; // coupler x_dot
        q_dot[4] = 0.25; // coupler y_dot
        q_dot[5] = -1.5; // coupler theta_dot

        // Use a harmonic profile so s'' is nonzero everywhere
        let profile = CamProfile::Harmonic {
            amplitude: 0.05,
            frequency: 2.0,
            phase: 0.3,
            offset: 0.1,
        };

        let joint = make_cam_follower(
            "CF1",
            "crank",
            "coupler",
            Vector2::new(0.05, 0.02),
            Vector2::new(-0.03, 0.01),
            Vector2::new(1.0, 0.3),
            profile,
        );

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn cam_follower_gamma_fd_ground_to_body() {
        let mut state = State::new();
        state.register_body("follower").unwrap();
        let mut q = state.make_q();
        state.set_pose("follower", &mut q, 0.3, 0.1, 0.5);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.4; // follower x_dot
        q_dot[1] = -0.3; // follower y_dot
        q_dot[2] = 2.0; // follower theta_dot

        // Cam body is ground (theta_i = 0, constant), follower slides
        let profile = CamProfile::Polynomial {
            coefficients: vec![0.0, 0.1, -0.05],
        };

        let joint = make_cam_follower(
            "CF_gnd",
            "ground",
            "follower",
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 1.0),
            profile,
        );

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn cam_follower_gamma_fd_cam_rotating() {
        // Cam body rotates with nonzero angular velocity — exercises all terms
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.4);
        state.set_pose("coupler", &mut q, 0.2, 0.15, 0.0);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.1; // crank x_dot
        q_dot[1] = 0.05; // crank y_dot
        q_dot[2] = 4.0; // crank theta_dot (large — stresses centripetal terms)
        q_dot[3] = 0.3; // coupler x_dot
        q_dot[4] = -0.15; // coupler y_dot
        q_dot[5] = 1.0; // coupler theta_dot

        let profile = CamProfile::Harmonic {
            amplitude: 0.08,
            frequency: 3.0,
            phase: 0.0,
            offset: 0.0,
        };

        let joint = make_cam_follower(
            "CF_rot",
            "crank",
            "coupler",
            Vector2::new(0.1, 0.0),
            Vector2::new(-0.05, 0.02),
            Vector2::new(0.6, 0.8),
            profile,
        );

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }
}
