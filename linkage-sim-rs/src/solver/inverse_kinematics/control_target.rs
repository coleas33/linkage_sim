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
use crate::core::state::State;

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
    /// Euclidean distance from a body-local point to a fixed reference point.
    Distance {
        body_id: String,
        local_pt: [f64; 2],
        ref_pt: [f64; 2],
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
            ControlTarget::Distance {
                body_id,
                local_pt,
                ref_pt,
            } => {
                let p = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                let p_world = mech.state().body_point_global(body_id, &p, q);
                let r = nalgebra::Vector2::new(ref_pt[0], ref_pt[1]);
                (p_world - r).norm()
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
            ControlTarget::Distance {
                body_id,
                local_pt,
                ref_pt,
            } => {
                if !state.is_ground(body_id) {
                    let idx = state.get_index(body_id).expect("body not registered");
                    let p = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                    let p_world = state.body_point_global(body_id, &p, q);
                    let r = nalgebra::Vector2::new(ref_pt[0], ref_pt[1]);
                    let d = p_world - r;
                    let length = d.norm();
                    if length < 1e-12 {
                        // Gradient undefined at distance 0 — leave zero
                        return grad;
                    }
                    let unit = d / length;
                    grad[idx.x_idx()] = unit.x;
                    grad[idx.y_idx()] = unit.y;
                    let bs = state.body_point_global_derivative(body_id, &p, q);
                    grad[idx.theta_idx()] = unit.dot(&bs);
                }
            }
        }

        grad
    }

    /// Hessian `∇²_q g(q)` — n_coords × n_coords. Sparse: only entries
    /// involving the targeted body's (x, y, θ) coordinates can be non-zero.
    ///
    /// See: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8.3
    pub fn hessian(&self, mech: &Mechanism, q: &DVector<f64>) -> DMatrix<f64> {
        let state = mech.state();
        let n = state.n_coords();

        match self {
            // g = θ_i is linear in q ⇒ Hessian = 0.
            ControlTarget::Angle { .. } => DMatrix::zeros(n, n),

            // g = r_x + (A(θ)·s)_x ;  ∂²g/∂θ² = -(A(θ)·s)_x ; all other entries 0.
            ControlTarget::WorldX { body_id, local_pt } => {
                if state.is_ground(body_id) {
                    return DMatrix::zeros(n, n);
                }
                let idx = state.get_index(body_id).expect("body not registered");
                let p = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                let theta = state.get_angle(body_id, q);
                let a_s = State::rotation_matrix(theta) * p;
                let mut h = DMatrix::zeros(n, n);
                h[(idx.theta_idx(), idx.theta_idx())] = -a_s.x;
                h
            }

            // g = r_y + (A(θ)·s)_y ;  ∂²g/∂θ² = -(A(θ)·s)_y ; all other entries 0.
            ControlTarget::WorldY { body_id, local_pt } => {
                if state.is_ground(body_id) {
                    return DMatrix::zeros(n, n);
                }
                let idx = state.get_index(body_id).expect("body not registered");
                let p = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                let theta = state.get_angle(body_id, q);
                let a_s = State::rotation_matrix(theta) * p;
                let mut h = DMatrix::zeros(n, n);
                h[(idx.theta_idx(), idx.theta_idx())] = -a_s.y;
                h
            }

            // g = unit · (P − origin). Only ∂²g/∂θ² = -unit · A(θ)·s is non-zero.
            ControlTarget::Projection {
                body_id,
                local_pt,
                axis_origin: _,
                axis_dir,
            } => {
                if state.is_ground(body_id) {
                    return DMatrix::zeros(n, n);
                }
                let idx = state.get_index(body_id).expect("body not registered");
                let p = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                let theta = state.get_angle(body_id, q);
                let a_s = State::rotation_matrix(theta) * p;
                let dir = nalgebra::Vector2::new(axis_dir[0], axis_dir[1]);
                let dir_norm = dir.norm();
                assert!(dir_norm > 1e-12, "Projection axis_dir is zero");
                let unit = dir / dir_norm;
                let mut h = DMatrix::zeros(n, n);
                h[(idx.theta_idx(), idx.theta_idx())] = -unit.dot(&a_s);
                h
            }

            // g = ‖P − ref‖. Six independent entries (3×3 block, symmetric).
            // Let d = P − ref, ℓ = ‖d‖, û = d/ℓ, A·s = A(θ)·s, B·s = B(θ)·s.
            //   ∂²g/∂x²    = (1 − û_x²)/ℓ
            //   ∂²g/∂y²    = (1 − û_y²)/ℓ
            //   ∂²g/∂x∂y   = −û_x·û_y/ℓ
            //   ∂²g/∂x∂θ   = ((B·s)_x − û_x·(û · B·s))/ℓ
            //   ∂²g/∂y∂θ   = ((B·s)_y − û_y·(û · B·s))/ℓ
            //   ∂²g/∂θ²    = (‖B·s‖² − (û · B·s)²)/ℓ − û · (A·s)
            // Edge case: ℓ < 1e-12 ⇒ Hessian undefined; return zeros (matches gradient).
            ControlTarget::Distance {
                body_id,
                local_pt,
                ref_pt,
            } => {
                if state.is_ground(body_id) {
                    return DMatrix::zeros(n, n);
                }
                let idx = state.get_index(body_id).expect("body not registered");
                let p = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                let p_world = state.body_point_global(body_id, &p, q);
                let r = nalgebra::Vector2::new(ref_pt[0], ref_pt[1]);
                let d = p_world - r;
                let ell = d.norm();
                if ell < 1e-12 {
                    return DMatrix::zeros(n, n);
                }
                let unit = d / ell;
                let theta = state.get_angle(body_id, q);
                let a_s = State::rotation_matrix(theta) * p;
                let bs = state.body_point_global_derivative(body_id, &p, q);
                let u_dot_bs = unit.dot(&bs);
                let bs_norm_sq = bs.norm_squared();

                let mut h = DMatrix::zeros(n, n);
                let xi = idx.x_idx();
                let yi = idx.y_idx();
                let ti = idx.theta_idx();
                h[(xi, xi)] = (1.0 - unit.x * unit.x) / ell;
                h[(yi, yi)] = (1.0 - unit.y * unit.y) / ell;
                let off_xy = -unit.x * unit.y / ell;
                h[(xi, yi)] = off_xy;
                h[(yi, xi)] = off_xy;
                let off_xt = (bs.x - unit.x * u_dot_bs) / ell;
                h[(xi, ti)] = off_xt;
                h[(ti, xi)] = off_xt;
                let off_yt = (bs.y - unit.y * u_dot_bs) / ell;
                h[(yi, ti)] = off_yt;
                h[(ti, yi)] = off_yt;
                h[(ti, ti)] = (bs_norm_sq - u_dot_bs * u_dot_bs) / ell - unit.dot(&a_s);
                h
            }
        }
    }

    /// Construct an `Angle` variant. Panics if `body_id` is `"ground"` (ground
    /// has fixed θ = 0; controlling its angle would be a degenerate target).
    pub fn angle(body_id: impl Into<String>) -> Self {
        let body_id = body_id.into();
        assert!(body_id != "ground", "ControlTarget::angle cannot target ground (θ is fixed)");
        ControlTarget::Angle { body_id }
    }

    /// Construct a `WorldX` variant. Panics if `body_id` is `"ground"` (ground
    /// is at fixed origin; observable would be the constant `local_pt[0]`).
    pub fn world_x(body_id: impl Into<String>, local_pt: [f64; 2]) -> Self {
        let body_id = body_id.into();
        assert!(body_id != "ground", "ControlTarget::world_x cannot target ground (pose is fixed)");
        ControlTarget::WorldX {
            body_id,
            local_pt,
        }
    }

    /// Construct a `WorldY` variant. Panics if `body_id` is `"ground"`.
    pub fn world_y(body_id: impl Into<String>, local_pt: [f64; 2]) -> Self {
        let body_id = body_id.into();
        assert!(body_id != "ground", "ControlTarget::world_y cannot target ground (pose is fixed)");
        ControlTarget::WorldY {
            body_id,
            local_pt,
        }
    }

    /// Construct a `Projection` variant. `axis_dir` is normalized in place;
    /// panics if `axis_dir` has zero length or `body_id` is `"ground"`.
    pub fn projection(
        body_id: impl Into<String>,
        local_pt: [f64; 2],
        axis_origin: [f64; 2],
        axis_dir: [f64; 2],
    ) -> Self {
        let body_id = body_id.into();
        assert!(body_id != "ground", "ControlTarget::projection cannot target ground (pose is fixed)");
        let dir = nalgebra::Vector2::new(axis_dir[0], axis_dir[1]);
        let n = dir.norm();
        assert!(n > 1e-12, "ControlTarget::projection axis_dir must be non-zero");
        let unit = dir / n;
        ControlTarget::Projection {
            body_id,
            local_pt,
            axis_origin,
            axis_dir: [unit.x, unit.y],
        }
    }

    /// Construct a `Distance` variant. Panics if `body_id` is `"ground"`.
    pub fn distance(
        body_id: impl Into<String>,
        local_pt: [f64; 2],
        ref_pt: [f64; 2],
    ) -> Self {
        let body_id = body_id.into();
        assert!(body_id != "ground", "ControlTarget::distance cannot target ground (pose is fixed)");
        ControlTarget::Distance {
            body_id,
            local_pt,
            ref_pt,
        }
    }

    /// Human-readable unit label for the observable's value.
    pub fn unit_label(&self) -> &'static str {
        match self {
            ControlTarget::Angle { .. } => "rad",
            ControlTarget::WorldX { .. }
            | ControlTarget::WorldY { .. }
            | ControlTarget::Projection { .. }
            | ControlTarget::Distance { .. } => "m",
        }
    }

    /// Body ID this target reads from. For UI / diagnostics.
    pub fn body_id(&self) -> &str {
        match self {
            ControlTarget::Angle { body_id }
            | ControlTarget::WorldX { body_id, .. }
            | ControlTarget::WorldY { body_id, .. }
            | ControlTarget::Projection { body_id, .. }
            | ControlTarget::Distance { body_id, .. } => body_id,
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

    #[test]
    fn distance_target_evaluates_to_norm() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.0);
        // Distance from crank midpoint to origin
        let target = ControlTarget::Distance {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
            ref_pt: [0.0, 0.0],
        };
        let g = target.evaluate(&mech, &q);
        // At t=0, body-local (0.005, 0.0) maps to world (0.005, 0.0)
        // (same as WorldX test). Distance to (0,0) = 0.005.
        assert_abs_diff_eq!(g, 0.005, epsilon = 1e-7);
    }

    #[test]
    fn distance_gradient_fd_check() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.2);
        let target = ControlTarget::Distance {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
            ref_pt: [0.0, 0.005],
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
                "Distance gradient FD mismatch at '{}' (idx {}): analytic={}, fd={}",
                label, idx, grad[idx], fd,
            );
        }
    }

    #[test]
    fn projection_constructor_normalizes_axis() {
        let target = ControlTarget::projection(
            "crank", [0.005, 0.0], [0.0, 0.0], [3.0, 4.0],
        );
        if let ControlTarget::Projection { axis_dir, .. } = target {
            assert_abs_diff_eq!(axis_dir[0], 0.6, epsilon = 1e-15);
            assert_abs_diff_eq!(axis_dir[1], 0.8, epsilon = 1e-15);
        } else {
            panic!("not Projection variant");
        }
    }

    #[test]
    #[should_panic(expected = "axis_dir")]
    fn projection_constructor_rejects_zero_axis() {
        let _ = ControlTarget::projection("crank", [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "ground")]
    fn angle_constructor_rejects_ground() {
        let _ = ControlTarget::angle("ground");
    }

    #[test]
    #[should_panic(expected = "ground")]
    fn world_x_constructor_rejects_ground() {
        let _ = ControlTarget::world_x("ground", [0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "ground")]
    fn distance_constructor_rejects_ground() {
        let _ = ControlTarget::distance("ground", [0.0, 0.0], [0.0, 0.0]);
    }

    #[test]
    fn unit_labels() {
        assert_eq!(
            ControlTarget::Angle { body_id: "x".into() }.unit_label(),
            "rad"
        );
        assert_eq!(
            ControlTarget::WorldX { body_id: "x".into(), local_pt: [0.0, 0.0] }.unit_label(),
            "m"
        );
        assert_eq!(
            ControlTarget::Distance {
                body_id: "x".into(),
                local_pt: [0.0, 0.0],
                ref_pt: [0.0, 0.0],
            }.unit_label(),
            "m"
        );
    }

    /// Helper: FD-check that `target.hessian(q)` agrees with central differences
    /// of `target.gradient(q)` at the given coordinate indices.
    fn assert_hessian_fd_matches(
        target: &ControlTarget,
        mech: &Mechanism,
        q: &DVector<f64>,
        indices: &[usize],
    ) {
        let h_analytic = target.hessian(mech, q);
        let h = 1e-7;
        for &i in indices {
            for &j in indices {
                let mut q_plus = q.clone();
                q_plus[j] += h;
                let mut q_minus = q.clone();
                q_minus[j] -= h;
                let grad_plus = target.gradient(mech, &q_plus);
                let grad_minus = target.gradient(mech, &q_minus);
                let fd = (grad_plus[i] - grad_minus[i]) / (2.0 * h);
                assert!(
                    (h_analytic[(i, j)] - fd).abs() < 1e-5,
                    "Hessian mismatch at ({}, {}): analytic={}, fd={}",
                    i, j, h_analytic[(i, j)], fd,
                );
            }
        }
    }

    #[test]
    fn world_x_hessian_fd_check() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.2);
        let target = ControlTarget::world_x("crank", [0.005, 0.0]);
        let crank = mech.state().get_index("crank").unwrap();
        assert_hessian_fd_matches(
            &target,
            &mech,
            &q,
            &[crank.x_idx(), crank.y_idx(), crank.theta_idx()],
        );
    }

    #[test]
    fn world_y_hessian_fd_check() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.2);
        let target = ControlTarget::world_y("crank", [0.005, 0.0]);
        let crank = mech.state().get_index("crank").unwrap();
        assert_hessian_fd_matches(
            &target,
            &mech,
            &q,
            &[crank.x_idx(), crank.y_idx(), crank.theta_idx()],
        );
    }

    #[test]
    fn projection_hessian_fd_check() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.2);
        let target = ControlTarget::Projection {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
            axis_origin: [0.01, 0.0],
            axis_dir: [0.6, 0.8], // unit length
        };
        let crank = mech.state().get_index("crank").unwrap();
        assert_hessian_fd_matches(
            &target,
            &mech,
            &q,
            &[crank.x_idx(), crank.y_idx(), crank.theta_idx()],
        );
    }

    #[test]
    fn distance_hessian_fd_check() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.2);
        let target = ControlTarget::Distance {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
            ref_pt: [0.0, 0.005],
        };
        let crank = mech.state().get_index("crank").unwrap();
        assert_hessian_fd_matches(
            &target,
            &mech,
            &q,
            &[crank.x_idx(), crank.y_idx(), crank.theta_idx()],
        );
    }

    #[test]
    fn distance_hessian_returns_zero_at_singularity() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.0);
        let body_origin_world = mech.state().body_point_global(
            "crank",
            &nalgebra::Vector2::new(0.0, 0.0),
            &q,
        );
        let target = ControlTarget::Distance {
            body_id: "crank".into(),
            local_pt: [0.0, 0.0],
            ref_pt: [body_origin_world.x, body_origin_world.y],
        };
        let h = target.hessian(&mech, &q);
        for i in 0..h.nrows() {
            for j in 0..h.ncols() {
                assert_abs_diff_eq!(h[(i, j)], 0.0, epsilon = 1e-15);
            }
        }
    }

    /// Fold-in from Task 1.7 review: exercise the Distance::gradient singularity branch.
    #[test]
    fn distance_gradient_returns_zero_at_singularity() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.0);
        // Body-local origin maps to body's world origin; setting ref_pt equal to that
        // forces length = 0 and the early-return-zero branch.
        let body_origin_world = mech.state().body_point_global(
            "crank",
            &nalgebra::Vector2::new(0.0, 0.0),
            &q,
        );
        let target = ControlTarget::Distance {
            body_id: "crank".into(),
            local_pt: [0.0, 0.0],
            ref_pt: [body_origin_world.x, body_origin_world.y],
        };
        let grad = target.gradient(&mech, &q);
        // All components should be zero (singularity early-return)
        for i in 0..grad.len() {
            assert_abs_diff_eq!(grad[i], 0.0, epsilon = 1e-15);
        }
    }
}
