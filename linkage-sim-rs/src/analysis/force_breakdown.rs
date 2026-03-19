//! Force element contribution breakdown.
//!
//! Evaluates each force element individually at a given configuration to
//! determine which dominates: gravity, springs, dampers, motors, etc.
//! Ported from `linkage_sim.analysis.force_breakdown`.

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;

/// Contribution of a single force element to the generalized force vector.
///
/// Each entry represents one force element's individual contribution
/// at a specific mechanism configuration. The `q_norm` field gives
/// the overall magnitude of that element's generalized force vector,
/// useful for ranking which elements dominate.
#[derive(Debug, Clone)]
pub struct ForceContribution {
    /// Index of this force element in the mechanism's force list.
    pub element_index: usize,
    /// Human-readable type name (e.g., "Gravity", "Linear Spring").
    pub type_name: String,
    /// L2 norm of the generalized force vector from this element.
    pub q_norm: f64,
}

/// Evaluate each force element's individual contribution to Q.
///
/// Iterates over all force elements in the mechanism and evaluates
/// each one separately, producing a `ForceContribution` with the
/// norm of each element's generalized force vector.
///
/// # Arguments
/// * `mech` - A built Mechanism instance.
/// * `q` - Generalized coordinate vector.
/// * `q_dot` - Generalized velocity vector.
/// * `t` - Time.
///
/// # Returns
/// A `Vec<ForceContribution>`, one per force element, in the same
/// order as `mech.forces()`.
pub fn evaluate_contributions(
    mech: &Mechanism,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    t: f64,
) -> Vec<ForceContribution> {
    mech.forces()
        .iter()
        .enumerate()
        .map(|(i, force)| {
            let q_vec = force.evaluate(mech.state(), mech.bodies(), q, q_dot, t);
            ForceContribution {
                element_index: i,
                type_name: force.type_name().to_string(),
                q_norm: q_vec.norm(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use crate::core::mechanism::Mechanism;
    use crate::forces::elements::{
        ExternalForceElement, ForceElement, GravityElement, LinearSpringElement,
    };
    use approx::assert_abs_diff_eq;

    fn build_mechanism_with_forces() -> Mechanism {
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar = make_bar("bar", "A", "B", 1.0, 2.0, 0.1);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.add_revolute_driver("D1", "ground", "bar", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();

        // Add gravity and an external force
        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        mech.add_force(ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.5, 0.0],
            force: [100.0, 0.0],
        }));

        mech.build().unwrap();
        mech
    }

    #[test]
    fn contributions_returns_one_per_force_element() {
        let mech = build_mechanism_with_forces();
        let q = mech.state().make_q();
        let q_dot = DVector::zeros(mech.state().n_coords());

        let contributions = evaluate_contributions(&mech, &q, &q_dot, 0.0);

        assert_eq!(contributions.len(), 2);
        assert_eq!(contributions[0].element_index, 0);
        assert_eq!(contributions[0].type_name, "Gravity");
        assert_eq!(contributions[1].element_index, 1);
        assert_eq!(contributions[1].type_name, "External Force");
    }

    #[test]
    fn contributions_norms_are_nonnegative() {
        let mech = build_mechanism_with_forces();
        let mut q = mech.state().make_q();
        mech.state().set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(mech.state().n_coords());

        let contributions = evaluate_contributions(&mech, &q, &q_dot, 0.0);

        for c in &contributions {
            assert!(c.q_norm >= 0.0, "norm should be non-negative");
        }

        // Gravity on a 2 kg bar should produce a non-trivial force
        assert!(
            contributions[0].q_norm > 1.0,
            "gravity on 2 kg bar should have notable norm"
        );
        // External force of 100 N should dominate
        assert!(
            contributions[1].q_norm > 50.0,
            "100 N external force should have notable norm"
        );
    }

    #[test]
    fn contributions_empty_when_no_forces() {
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar = make_bar("bar", "A", "B", 1.0, 2.0, 0.1);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.add_revolute_driver("D1", "ground", "bar", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();
        mech.build().unwrap();

        let q = mech.state().make_q();
        let q_dot = DVector::zeros(mech.state().n_coords());

        let contributions = evaluate_contributions(&mech, &q, &q_dot, 0.0);
        assert!(contributions.is_empty());
    }

    #[test]
    fn contributions_spring_zero_at_free_length() {
        let ground = make_ground(&[("O", 0.0, 0.0), ("P", 1.0, 0.0)]);
        let bar = make_bar("bar", "A", "B", 1.0, 1.0, 0.05);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.add_revolute_driver("D1", "ground", "bar", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();

        // Spring from bar tip to ground point, free length = distance between them
        mech.add_force(ForceElement::LinearSpring(LinearSpringElement {
            body_a: "bar".into(),
            point_a: [0.5, 0.0], // half-length point
            body_b: "ground".into(),
            point_b: [1.0, 0.0], // ground point P
            stiffness: 500.0,
            free_length: 0.5, // exactly the initial distance
        }));

        mech.build().unwrap();

        let mut q = mech.state().make_q();
        // Place bar horizontally at origin so point_a is at (0.5, 0)
        // and point_b is ground P at (1.0, 0) → distance = 0.5 = free_length
        mech.state().set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(mech.state().n_coords());

        let contributions = evaluate_contributions(&mech, &q, &q_dot, 0.0);
        assert_eq!(contributions.len(), 1);
        assert_abs_diff_eq!(contributions[0].q_norm, 0.0, epsilon = 1e-10);
    }
}
