//! Uniform gravity force element.

use std::collections::HashMap;

use nalgebra::{DVector, Vector2};

use crate::core::body::Body;
use crate::core::state::{State, GROUND_ID};
use crate::forces::helpers::point_force_to_q;

/// Uniform gravitational field applied to all bodies with mass > 0.
pub struct Gravity {
    pub g_vector: Vector2<f64>,
    pub bodies: HashMap<String, Body>,
}

impl Gravity {
    pub fn new(g_vector: Vector2<f64>, bodies: &HashMap<String, Body>) -> Self {
        Self {
            g_vector,
            bodies: bodies.clone(),
        }
    }

    pub fn id(&self) -> &str {
        "gravity"
    }

    /// Compute gravity generalized force for all bodies.
    ///
    /// For each body with mass > 0 (skip ground):
    ///   F = mass * g_vector, applied at CG
    pub fn evaluate(
        &self,
        state: &State,
        q: &DVector<f64>,
        _q_dot: &DVector<f64>,
        _t: f64,
    ) -> DVector<f64> {
        let mut total_q = DVector::zeros(state.n_coords());

        for (body_id, body) in &self.bodies {
            if body_id == GROUND_ID || body.mass <= 0.0 {
                continue;
            }
            let force_global = self.g_vector * body.mass;
            total_q += point_force_to_q(state, body_id, &body.cg_local, &force_global, q);
        }

        total_q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use approx::assert_abs_diff_eq;

    #[test]
    fn gravity_produces_downward_forces() {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);

        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar = make_bar("bar", "A", "B", 1.0, 2.0, 0.0);
        let mut bodies = HashMap::new();
        bodies.insert("ground".to_string(), ground);
        bodies.insert("bar".to_string(), bar);

        let gravity = Gravity::new(Vector2::new(0.0, -9.81), &bodies);
        let q_dot = DVector::zeros(3);
        let result = gravity.evaluate(&state, &q, &q_dot, 0.0);

        // Fy = mass * g = 2.0 * (-9.81) = -19.62
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result[1], -19.62, epsilon = 1e-10);
        // Torque from gravity at CG=(0.5, 0) with θ=0:
        // B(0) · (0.5, 0) = (0, 0.5), dot (0, -19.62) = -9.81
        assert_abs_diff_eq!(result[2], -9.81, epsilon = 1e-10);
    }

    #[test]
    fn gravity_skips_massless_bodies() {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let q = state.make_q();

        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar = make_bar("bar", "A", "B", 1.0, 0.0, 0.0); // zero mass
        let mut bodies = HashMap::new();
        bodies.insert("ground".to_string(), ground);
        bodies.insert("bar".to_string(), bar);

        let gravity = Gravity::new(Vector2::new(0.0, -9.81), &bodies);
        let q_dot = DVector::zeros(3);
        let result = gravity.evaluate(&state, &q, &q_dot, 0.0);

        for i in 0..result.len() {
            assert_abs_diff_eq!(result[i], 0.0, epsilon = 1e-15);
        }
    }
}
