//! The `Constraint` trait that all joint constraints must implement.

use nalgebra::{DMatrix, DVector};

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
