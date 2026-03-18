//! Generalized coordinate vector q and body-to-index mapping.
//!
//! Each moving body contributes 3 coordinates to q: (x, y, θ).
//! Ground is excluded from q — its pose is fixed at (0, 0, 0).

use std::collections::HashMap;

use nalgebra::{DVector, Matrix2, Vector2};
use thiserror::Error;

pub const GROUND_ID: &str = "ground";

/// Index mapping for one body's coordinates in the state vector q.
#[derive(Debug, Clone)]
pub struct BodyIndex {
    pub body_id: String,
    pub q_start: usize,
}

impl BodyIndex {
    pub fn x_idx(&self) -> usize {
        self.q_start
    }
    pub fn y_idx(&self) -> usize {
        self.q_start + 1
    }
    pub fn theta_idx(&self) -> usize {
        self.q_start + 2
    }
}

/// Generalized coordinate state for a mechanism.
///
/// Manages the mapping between body IDs and their positions in q.
/// Provides accessors so the rest of the codebase never does raw
/// index arithmetic on q.
#[derive(Debug, Clone, Default)]
pub struct State {
    body_indices: HashMap<String, BodyIndex>,
    n_moving_bodies: usize,
}

impl State {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of generalized coordinates.
    pub fn n_coords(&self) -> usize {
        3 * self.n_moving_bodies
    }

    pub fn n_moving_bodies(&self) -> usize {
        self.n_moving_bodies
    }

    /// Ordered list of moving body IDs (same order as in q).
    pub fn body_ids(&self) -> Vec<String> {
        let mut entries: Vec<_> = self.body_indices.iter().collect();
        entries.sort_by_key(|(_, idx)| idx.q_start);
        entries.into_iter().map(|(id, _)| id.clone()).collect()
    }

    /// Register a moving body and assign it a coordinate block in q.
    pub fn register_body(&mut self, body_id: &str) -> Result<BodyIndex, StateError> {
        if body_id == GROUND_ID {
            return Err(StateError::GroundRegistration);
        }
        if self.body_indices.contains_key(body_id) {
            return Err(StateError::DuplicateBody(body_id.to_string()));
        }
        let idx = BodyIndex {
            body_id: body_id.to_string(),
            q_start: 3 * self.n_moving_bodies,
        };
        self.body_indices.insert(body_id.to_string(), idx.clone());
        self.n_moving_bodies += 1;
        Ok(idx)
    }

    /// Get the coordinate index mapping for a body.
    pub fn get_index(&self, body_id: &str) -> Result<&BodyIndex, StateError> {
        if body_id == GROUND_ID {
            return Err(StateError::GroundHasNoIndex);
        }
        self.body_indices
            .get(body_id)
            .ok_or_else(|| StateError::UnknownBody(body_id.to_string()))
    }

    pub fn is_ground(&self, body_id: &str) -> bool {
        body_id == GROUND_ID
    }

    /// Get (x, y, θ) for a body from the state vector. Ground returns (0, 0, 0).
    pub fn get_pose(&self, body_id: &str, q: &DVector<f64>) -> (f64, f64, f64) {
        if body_id == GROUND_ID {
            return (0.0, 0.0, 0.0);
        }
        let idx = self.get_index(body_id).expect("body not registered");
        (q[idx.x_idx()], q[idx.y_idx()], q[idx.theta_idx()])
    }

    /// Get [x, y] position vector for a body from q.
    pub fn get_position(&self, body_id: &str, q: &DVector<f64>) -> Vector2<f64> {
        if body_id == GROUND_ID {
            return Vector2::zeros();
        }
        let idx = self.get_index(body_id).expect("body not registered");
        Vector2::new(q[idx.x_idx()], q[idx.y_idx()])
    }

    /// Get θ for a body from q.
    pub fn get_angle(&self, body_id: &str, q: &DVector<f64>) -> f64 {
        if body_id == GROUND_ID {
            return 0.0;
        }
        let idx = self.get_index(body_id).expect("body not registered");
        q[idx.theta_idx()]
    }

    /// Set only θ for a body in the state vector.
    pub fn set_angle(&self, body_id: &str, q: &mut DVector<f64>, theta: f64) {
        assert!(
            body_id != GROUND_ID,
            "Cannot set angle for ground body."
        );
        let idx = self.get_index(body_id).expect("body not registered");
        q[idx.theta_idx()] = theta;
    }

    /// Create a zero-initialized state vector of the correct size.
    pub fn make_q(&self) -> DVector<f64> {
        DVector::zeros(self.n_coords())
    }

    /// Set (x, y, θ) for a body in the state vector.
    pub fn set_pose(
        &self,
        body_id: &str,
        q: &mut DVector<f64>,
        x: f64,
        y: f64,
        theta: f64,
    ) {
        assert!(
            body_id != GROUND_ID,
            "Cannot set pose for ground body."
        );
        let idx = self.get_index(body_id).expect("body not registered");
        q[idx.x_idx()] = x;
        q[idx.y_idx()] = y;
        q[idx.theta_idx()] = theta;
    }

    /// 2×2 rotation matrix A(θ) = [[cos θ, −sin θ], [sin θ, cos θ]].
    pub fn rotation_matrix(theta: f64) -> Matrix2<f64> {
        let (s, c) = theta.sin_cos();
        Matrix2::new(c, -s, s, c)
    }

    /// 2×2 derivative B(θ) = dA/dθ = [[−sin θ, −cos θ], [cos θ, −sin θ]].
    pub fn rotation_matrix_derivative(theta: f64) -> Matrix2<f64> {
        let (s, c) = theta.sin_cos();
        Matrix2::new(-s, -c, c, -s)
    }

    /// Transform a point from body-local to global coordinates.
    /// r_global = r_body + A(θ) · s_local
    pub fn body_point_global(
        &self,
        body_id: &str,
        local_point: &Vector2<f64>,
        q: &DVector<f64>,
    ) -> Vector2<f64> {
        if body_id == GROUND_ID {
            return *local_point;
        }
        let pos = self.get_position(body_id, q);
        let theta = self.get_angle(body_id, q);
        let a = Self::rotation_matrix(theta);
        pos + a * local_point
    }

    /// Partial derivative of global point position w.r.t. θ: B(θ) · s_local.
    pub fn body_point_global_derivative(
        &self,
        body_id: &str,
        local_point: &Vector2<f64>,
        q: &DVector<f64>,
    ) -> Vector2<f64> {
        if body_id == GROUND_ID {
            return Vector2::zeros();
        }
        let theta = self.get_angle(body_id, q);
        let b = Self::rotation_matrix_derivative(theta);
        b * local_point
    }

    /// Compute the global velocity of a point on a body.
    /// v_P = v_body + B(θ) · s_local · θ̇
    pub fn body_point_velocity(
        &self,
        body_id: &str,
        local_point: &Vector2<f64>,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
    ) -> Vector2<f64> {
        if body_id == GROUND_ID {
            return Vector2::zeros();
        }
        let idx = self.get_index(body_id).expect("body not registered");
        let v_body = Vector2::new(q_dot[idx.x_idx()], q_dot[idx.y_idx()]);
        let theta = self.get_angle(body_id, q);
        let theta_dot = q_dot[idx.theta_idx()];
        let b = Self::rotation_matrix_derivative(theta);
        v_body + (b * local_point) * theta_dot
    }
}

#[derive(Debug, Error)]
pub enum StateError {
    #[error("Ground body must not be registered in the state vector")]
    GroundRegistration,
    #[error("Body '{0}' is already registered")]
    DuplicateBody(String),
    #[error("Ground body has no state indices — its pose is fixed at (0, 0, 0)")]
    GroundHasNoIndex,
    #[error("Body '{0}' is not registered")]
    UnknownBody(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    fn register_and_retrieve() {
        let mut state = State::new();
        let idx = state.register_body("crank").unwrap();
        assert_eq!(idx.x_idx(), 0);
        assert_eq!(idx.y_idx(), 1);
        assert_eq!(idx.theta_idx(), 2);
        assert_eq!(state.n_coords(), 3);

        let idx2 = state.register_body("coupler").unwrap();
        assert_eq!(idx2.q_start, 3);
        assert_eq!(state.n_coords(), 6);
    }

    #[test]
    fn ground_registration_rejected() {
        let mut state = State::new();
        assert!(state.register_body("ground").is_err());
    }

    #[test]
    fn duplicate_body_rejected() {
        let mut state = State::new();
        state.register_body("crank").unwrap();
        assert!(state.register_body("crank").is_err());
    }

    #[test]
    fn ground_pose_is_origin() {
        let state = State::new();
        let q = DVector::zeros(0);
        assert_eq!(state.get_pose("ground", &q), (0.0, 0.0, 0.0));
        assert_eq!(state.get_position("ground", &q), Vector2::zeros());
        assert_eq!(state.get_angle("ground", &q), 0.0);
    }

    #[test]
    fn set_and_get_pose() {
        let mut state = State::new();
        state.register_body("crank").unwrap();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 1.0, 2.0, PI / 4.0);
        let (x, y, theta) = state.get_pose("crank", &q);
        assert_abs_diff_eq!(x, 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(y, 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(theta, PI / 4.0, epsilon = 1e-15);
    }

    #[test]
    fn rotation_matrix_identity_at_zero() {
        let a = State::rotation_matrix(0.0);
        assert_abs_diff_eq!(a[(0, 0)], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(a[(0, 1)], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(a[(1, 0)], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(a[(1, 1)], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn rotation_matrix_90_degrees() {
        let a = State::rotation_matrix(PI / 2.0);
        assert_abs_diff_eq!(a[(0, 0)], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(a[(0, 1)], -1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(a[(1, 0)], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(a[(1, 1)], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn body_point_global_ground_passthrough() {
        let state = State::new();
        let q = DVector::zeros(0);
        let pt = Vector2::new(1.0, 2.0);
        let global = state.body_point_global("ground", &pt, &q);
        assert_abs_diff_eq!(global.x, 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(global.y, 2.0, epsilon = 1e-15);
    }

    #[test]
    fn body_point_global_rotated() {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, PI / 2.0);

        let local = Vector2::new(1.0, 0.0);
        let global = state.body_point_global("bar", &local, &q);
        assert_abs_diff_eq!(global.x, 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(global.y, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn body_point_global_translated_and_rotated() {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 3.0, 4.0, PI / 2.0);

        let local = Vector2::new(1.0, 0.0);
        let global = state.body_point_global("bar", &local, &q);
        assert_abs_diff_eq!(global.x, 3.0, epsilon = 1e-14);
        assert_abs_diff_eq!(global.y, 5.0, epsilon = 1e-14);
    }

    #[test]
    fn body_ids_ordered() {
        let mut state = State::new();
        state.register_body("coupler").unwrap();
        state.register_body("crank").unwrap();
        state.register_body("rocker").unwrap();
        let ids = state.body_ids();
        assert_eq!(ids, vec!["coupler", "crank", "rocker"]);
    }
}
