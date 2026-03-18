//! Assemble total generalized force Q from all force elements.

use nalgebra::DVector;

use crate::core::state::State;
use crate::forces::gravity::Gravity;

/// Assemble total generalized force Q from gravity (and future force elements).
///
/// Currently supports Gravity. Additional force element types
/// will be added as the port progresses (springs, dampers, etc.).
pub fn assemble_q(
    state: &State,
    gravity: Option<&Gravity>,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    t: f64,
) -> DVector<f64> {
    let mut total = DVector::zeros(state.n_coords());

    if let Some(g) = gravity {
        total += g.evaluate(state, q, q_dot, t);
    }

    total
}
