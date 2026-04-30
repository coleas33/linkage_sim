//! `ControlTarget` enum: scalar output observables that can be controlled.
//!
//! Each variant implements `g(q)`, `∇_q g(q)`, and `∇²_q g(q)`. These feed the
//! inverse-Newton outer loop in `solver.rs` and the closed-form / FD helpers
//! in `derivatives.rs`.
//!
//! See: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8
//!      docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §5.1

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::core::mechanism::Mechanism;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ControlTarget {
    /// Body angle θ_body. Linear in q (Hessian = 0).
    Angle { body_id: String },
    /// World x-coord of body-local point.
    WorldX { body_id: String, local_pt: [f64; 2] },
    /// World y-coord of body-local point.
    WorldY { body_id: String, local_pt: [f64; 2] },
    /// Projection of body-local point onto fixed line through `axis_origin` along `axis_dir`.
    /// `axis_dir` is normalized internally by `evaluate`/`gradient`/`hessian`.
    Projection {
        body_id: String,
        local_pt: [f64; 2],
        axis_origin: [f64; 2],
        axis_dir: [f64; 2],
    },
}

impl ControlTarget {
    /// Evaluate `g(q)` — scalar value of the observable at configuration `q`.
    pub fn evaluate(&self, mech: &Mechanism, q: &DVector<f64>) -> f64 {
        match self {
            ControlTarget::Angle { body_id } => mech.state().get_angle(body_id, q),
            ControlTarget::WorldX { body_id, local_pt } => {
                let p = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                mech.state().body_point_global(body_id, &p, q).x
            }
            ControlTarget::WorldY { body_id, local_pt } => {
                let p = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                mech.state().body_point_global(body_id, &p, q).y
            }
            ControlTarget::Projection {
                body_id,
                local_pt,
                axis_origin,
                axis_dir,
            } => {
                let p = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                let p_world = mech.state().body_point_global(body_id, &p, q);
                let origin = nalgebra::Vector2::new(axis_origin[0], axis_origin[1]);
                let dir = nalgebra::Vector2::new(axis_dir[0], axis_dir[1]);
                let dir_norm = dir.norm();
                assert!(dir_norm > 1e-12, "Projection axis_dir is zero");
                let unit = dir / dir_norm;
                (p_world - origin).dot(&unit)
            }
        }
    }

    /// Gradient `∇_q g(q)` — length n_coords.
    pub fn gradient(&self, mech: &Mechanism, q: &DVector<f64>) -> DVector<f64> {
        let state = mech.state();
        let n = state.n_coords();
        let mut grad = DVector::zeros(n);

        match self {
            ControlTarget::Angle { body_id } => {
                if !state.is_ground(body_id) {
                    let idx = state.get_index(body_id).expect("body not registered");
                    grad[idx.theta_idx()] = 1.0;
                }
            }
            ControlTarget::WorldX { body_id, local_pt } => {
                if !state.is_ground(body_id) {
                    let idx = state.get_index(body_id).expect("body not registered");
                    let p = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                    // P_x = r_x + (A(θ)·s)_x ;  ∂P_x/∂x = 1, ∂P_x/∂θ = (B(θ)·s)_x
                    grad[idx.x_idx()] = 1.0;
                    let bs = state.body_point_global_derivative(body_id, &p, q);
                    grad[idx.theta_idx()] = bs.x;
                }
            }
            ControlTarget::WorldY { body_id, local_pt } => {
                if !state.is_ground(body_id) {
                    let idx = state.get_index(body_id).expect("body not registered");
                    let p = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                    grad[idx.y_idx()] = 1.0;
                    let bs = state.body_point_global_derivative(body_id, &p, q);
                    grad[idx.theta_idx()] = bs.y;
                }
            }
            ControlTarget::Projection {
                body_id,
                local_pt,
                axis_origin: _,
                axis_dir,
            } => {
                if !state.is_ground(body_id) {
                    let idx = state.get_index(body_id).expect("body not registered");
                    let p = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                    let dir = nalgebra::Vector2::new(axis_dir[0], axis_dir[1]);
                    let dir_norm = dir.norm();
                    assert!(dir_norm > 1e-12, "Projection axis_dir is zero");
                    let unit = dir / dir_norm;
                    // ∂g/∂r = unit ;  ∂g/∂θ = unit · B(θ)·s
                    grad[idx.x_idx()] = unit.x;
                    grad[idx.y_idx()] = unit.y;
                    let bs = state.body_point_global_derivative(body_id, &p, q);
                    grad[idx.theta_idx()] = unit.dot(&bs);
                }
            }
        }

        grad
    }

    /// Hessian `∇²_q g(q)` — n_coords × n_coords. Zero for variants linear in q.
    pub fn hessian(&self, mech: &Mechanism, q: &DVector<f64>) -> DMatrix<f64> {
        let n = mech.state().n_coords();
        let _ = q;
        match self {
            ControlTarget::Angle { .. }
            | ControlTarget::WorldX { .. }
            | ControlTarget::WorldY { .. }
            | ControlTarget::Projection { .. } => DMatrix::zeros(n, n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::inverse_kinematics::test_helpers::{build_fourbar, solve_at};
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    fn angle_target_evaluates_correctly() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.25); // crank at 0.25 rev = π/2
        let target = ControlTarget::Angle { body_id: "crank".into() };
        let g = target.evaluate(&mech, &q);
        assert_abs_diff_eq!(g, PI / 2.0, epsilon = 1e-8);
    }

    #[test]
    fn angle_target_gradient_is_unit_at_theta() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.0);
        let target = ControlTarget::Angle { body_id: "crank".into() };
        let grad = target.gradient(&mech, &q);
        let crank_idx = mech.state().get_index("crank").unwrap();
        assert_abs_diff_eq!(grad[crank_idx.theta_idx()], 1.0, epsilon = 1e-15);
        // All other entries zero
        for i in 0..grad.len() {
            if i != crank_idx.theta_idx() {
                assert_abs_diff_eq!(grad[i], 0.0, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn angle_target_hessian_is_zero() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.0);
        let target = ControlTarget::Angle { body_id: "crank".into() };
        let h = target.hessian(&mech, &q);
        let n = mech.state().n_coords();
        assert_eq!(h.nrows(), n);
        assert_eq!(h.ncols(), n);
        for i in 0..n {
            for j in 0..n {
                assert_abs_diff_eq!(h[(i, j)], 0.0, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn world_x_target_evaluates_at_body_local_point() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.0); // crank along +x
        // Body-local point at (0.005, 0.0) on crank == midpoint of crank bar
        let target = ControlTarget::WorldX {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
        };
        let g = target.evaluate(&mech, &q);
        // At t=0, crank centroid is at (0.005, 0.0) world; midpoint same x.
        assert_abs_diff_eq!(g, 0.005, epsilon = 1e-8);
    }

    #[test]
    fn world_y_target_at_rotated_pose() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.25); // crank at π/2
        // Crank tip at body-local (0.01, 0) — bar length 0.01.
        let target = ControlTarget::WorldY {
            body_id: "crank".into(),
            local_pt: [0.01, 0.0],
        };
        let g = target.evaluate(&mech, &q);
        // Crank now points up; tip-y ≈ 0.01.
        assert_abs_diff_eq!(g, 0.01, epsilon = 1e-7);
    }

    #[test]
    fn world_x_gradient_finite_difference_check() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.1);
        let target = ControlTarget::WorldX {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
        };
        let grad = target.gradient(&mech, &q);
        // FD check on crank's x, y, theta
        let crank = mech.state().get_index("crank").unwrap();
        let h = 1e-7;

        for &(idx, label) in &[
            (crank.x_idx(), "x"),
            (crank.y_idx(), "y"),
            (crank.theta_idx(), "theta"),
        ] {
            let mut q_plus = q.clone();
            q_plus[idx] += h;
            let mut q_minus = q.clone();
            q_minus[idx] -= h;
            let fd = (target.evaluate(&mech, &q_plus) - target.evaluate(&mech, &q_minus)) / (2.0 * h);
            assert!(
                (grad[idx] - fd).abs() < 1e-5,
                "FD gradient mismatch at coord '{}' (idx {}): analytic={}, fd={}",
                label, idx, grad[idx], fd,
            );
        }
    }

    #[test]
    fn projection_target_along_x_equals_world_x_minus_origin() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.0);
        // Projection of crank tip onto world-x axis through origin
        let target = ControlTarget::Projection {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
            axis_origin: [0.0, 0.0],
            axis_dir: [1.0, 0.0],
        };
        let g = target.evaluate(&mech, &q);
        // Same as WorldX with same point at origin
        let world_x = ControlTarget::WorldX {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
        };
        let g_x = world_x.evaluate(&mech, &q);
        assert_abs_diff_eq!(g, g_x, epsilon = 1e-12);
    }

    #[test]
    fn projection_normalizes_axis_dir() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.0);
        // axis_dir not normalized; evaluate should yield same result as if normalized
        let unnorm = ControlTarget::Projection {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
            axis_origin: [0.0, 0.0],
            axis_dir: [3.0, 0.0], // length 3
        };
        let norm = ControlTarget::Projection {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
            axis_origin: [0.0, 0.0],
            axis_dir: [1.0, 0.0],
        };
        // evaluate normalizes internally
        assert_abs_diff_eq!(
            unnorm.evaluate(&mech, &q),
            norm.evaluate(&mech, &q),
            epsilon = 1e-12
        );
    }

    #[test]
    fn projection_gradient_fd_check() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.1);
        let target = ControlTarget::Projection {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
            axis_origin: [0.01, 0.0],
            axis_dir: [0.6, 0.8], // 3-4-5 triangle, length 1
        };
        let grad = target.gradient(&mech, &q);
        let crank = mech.state().get_index("crank").unwrap();
        let h = 1e-7;
        for &(idx, label) in &[
            (crank.x_idx(), "x"),
            (crank.y_idx(), "y"),
            (crank.theta_idx(), "theta"),
        ] {
            let mut qp = q.clone();
            qp[idx] += h;
            let mut qm = q.clone();
            qm[idx] -= h;
            let fd = (target.evaluate(&mech, &qp) - target.evaluate(&mech, &qm)) / (2.0 * h);
            assert!(
                (grad[idx] - fd).abs() < 1e-5,
                "Projection gradient FD mismatch at '{}' (idx {}): analytic={}, fd={}",
                label, idx, grad[idx], fd,
            );
        }
    }
}
