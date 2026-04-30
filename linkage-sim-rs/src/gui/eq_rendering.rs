//! Shared symbolic-form rendering for loop equations Φ_J* across the GUI.
//!
//! Used by:
//! - The "Equations" tab in the property panel (E1)
//! - The canvas equation overlay (E2)
//! - The HTML report's "Loop equations" section (E4)
//!
//! Constraints in a Mechanism are iterated via `mech.all_constraints()` which
//! yields `&dyn Constraint`. The trait object hides the concrete type, so we
//! reconstruct the joint kind by index: joints come first (in `mech.joints()`
//! order), then revolute drivers (`mech.drivers()`), then linear drivers
//! (`mech.linear_drivers()`). This matches the row-block layout that
//! `assemble_constraints` and the constraint_ranges machinery rely on, so the
//! kinds we report here line up with the rows of Φ.
//!
//! The functions in this module are pure — given a `Mechanism` and an index
//! they return strings without mutating anything — so they are safe to call
//! from rendering passes that already hold `&AppState`.
//!
//! No tests for the panel/overlay sites themselves (UI-only); the small
//! string helpers below have unit tests at the bottom of this file.

use crate::core::constraint::{Constraint, JointConstraint};
use crate::core::driver::DriverMeta;
use crate::core::mechanism::Mechanism;

/// Constraint kind used purely for symbolic-form rendering and color coding.
///
/// `Driver*` variants use the constraint id and any `DriverMeta` to render a
/// closed-form `f(t) = …` expression where possible; otherwise we fall back
/// to the generic `f(t)` notation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EqKind {
    Revolute,
    Prismatic,
    Fixed,
    CamFollower,
    RevoluteDriver,
    LinearDriver,
}

impl EqKind {
    /// Short label used as a constraint-type prefix on the canvas overlay.
    pub fn short_label(self) -> &'static str {
        match self {
            EqKind::Revolute => "Φ_rev",
            EqKind::Prismatic => "Φ_pri",
            EqKind::Fixed => "Φ_fix",
            EqKind::CamFollower => "Φ_cam",
            EqKind::RevoluteDriver => "Φ_rd",
            EqKind::LinearDriver => "Φ_ld",
        }
    }

    /// True if this is a driver constraint (1 equation, λ is the input
    /// torque/force rather than a reaction force).
    pub fn is_driver(self) -> bool {
        matches!(self, EqKind::RevoluteDriver | EqKind::LinearDriver)
    }

    /// Units string for the Lagrange multiplier λ associated with this kind.
    /// Drivers carry torque (rotational) or force (linear); joints carry
    /// constraint-row reactions and are rendered as "N" / "N·m" by row index
    /// (see `lambda_units_at_row`).
    pub fn driver_lambda_units(self) -> &'static str {
        match self {
            EqKind::RevoluteDriver => "N\u{00b7}m",
            EqKind::LinearDriver => "N",
            _ => "",
        }
    }
}

/// Identify the kind of the constraint at logical index `idx` in
/// `mech.all_constraints()`.
///
/// The ordering is: all joints (in `joints()` order), then all revolute
/// drivers, then all linear drivers. Returns `None` if the index is out of
/// range.
pub fn kind_at(mech: &Mechanism, idx: usize) -> Option<EqKind> {
    let n_joints = mech.joints().len();
    let n_rev_drivers = mech.drivers().len();
    let n_lin_drivers = mech.linear_drivers().len();

    if idx < n_joints {
        Some(match &mech.joints()[idx] {
            JointConstraint::Revolute(_) => EqKind::Revolute,
            JointConstraint::Prismatic(_) => EqKind::Prismatic,
            JointConstraint::Fixed(_) => EqKind::Fixed,
            JointConstraint::CamFollower(_) => EqKind::CamFollower,
        })
    } else if idx < n_joints + n_rev_drivers {
        Some(EqKind::RevoluteDriver)
    } else if idx < n_joints + n_rev_drivers + n_lin_drivers {
        Some(EqKind::LinearDriver)
    } else {
        None
    }
}

/// Total number of constraints (joints + drivers).
pub fn n_constraints(mech: &Mechanism) -> usize {
    mech.joints().len() + mech.drivers().len() + mech.linear_drivers().len()
}

/// Map a row index in Φ to the constraint index (the index used by `kind_at`).
///
/// Each constraint contributes `n_equations` rows; this scans those blocks.
pub fn constraint_index_for_row(mech: &Mechanism, row: usize) -> Option<usize> {
    let mut accum = 0usize;
    for (i, c) in mech.all_constraints().enumerate() {
        let next = accum + c.n_equations();
        if row < next {
            return Some(i);
        }
        accum = next;
    }
    None
}

/// Symbolic form `Φ = ...` for a constraint, rendered as a multi-line string
/// when the constraint has more than one equation (one line per row).
///
/// For drivers, the parameterization is rendered when the `DriverMeta` is
/// available (constant speed, linear length, cosine stroke, expression);
/// otherwise the fallback `f(t)` notation is used.
pub fn symbolic_form(mech: &Mechanism, idx: usize) -> String {
    let Some(kind) = kind_at(mech, idx) else {
        return String::new();
    };
    let n_joints = mech.joints().len();
    match kind {
        EqKind::Revolute => {
            "Φ_rev: r_i + A(θ_i)·s_i − r_j − A(θ_j)·s_j = 0  (2 eqs)".to_string()
        }
        EqKind::Prismatic => {
            "Φ_pri: n̂(θ_i)·(r_j − r_i + A·s_j − A·s_i) = 0,  θ_j − θ_i − Δ₀ = 0  (2 eqs)"
                .to_string()
        }
        EqKind::Fixed => {
            "Φ_fix: r_j + A(θ_j)·s_j − r_i − A(θ_i)·s_i = 0,  θ_j − θ_i − Δ₀ = 0  (3 eqs)"
                .to_string()
        }
        EqKind::CamFollower => {
            "Φ_cam: n̂·(P_j − P_i) − s(t) = 0  (1 eq)".to_string()
        }
        EqKind::RevoluteDriver => {
            let drv_idx = idx - n_joints;
            let drv = &mech.drivers()[drv_idx];
            let f_t = render_driver_f_t(drv.meta());
            format!("Φ_rd: θ_j − θ_i − {} = 0  (1 eq)", f_t)
        }
        EqKind::LinearDriver => {
            let drv_idx = idx - n_joints - mech.drivers().len();
            let drv = &mech.linear_drivers()[drv_idx];
            let d_t = render_driver_f_t(drv.meta());
            format!("Φ_ld: ‖P_b − P_a‖ − {} = 0  (1 eq)", d_t)
        }
    }
}

/// Return the right-hand side `f(t) = …` (or `d(t) = …`) text for a driver,
/// using `DriverMeta` when present.
fn render_driver_f_t(meta: Option<&DriverMeta>) -> String {
    match meta {
        Some(DriverMeta::ConstantSpeed { omega, theta_0 }) => {
            format!("(θ₀ + ω·t)  [θ₀={:.3} rad, ω={:.3} rad/s]", theta_0, omega)
        }
        Some(DriverMeta::Expression { expr, .. }) => format!("({})", expr),
        Some(DriverMeta::LinearLength { velocity, length_0 }) => {
            format!("(L₀ + v·t)  [L₀={:.4} m, v={:.4} m/s]", length_0, velocity)
        }
        Some(DriverMeta::CosineStroke {
            stroke_min,
            stroke_max,
            initial_length,
        }) => format!(
            "(mid + amp·cos(2π·t + φ))  [stroke=[{:.4}, {:.4}] m, L₀={:.4} m]",
            stroke_min, stroke_max, initial_length
        ),
        None => "f(t)".to_string(),
    }
}

/// Render the residual norm for a constraint as a short string (e.g.
/// "‖Φ‖ = 1.23e-10"). For multi-row constraints this is the L2 norm of the
/// rows; for 1-row constraints it is the absolute value.
pub fn residual_norm_string(mech: &Mechanism, idx: usize, phi: &nalgebra::DVector<f64>) -> String {
    let Some(range) = mech.constraint_ranges().get(idx) else {
        return String::new();
    };
    let mut sumsq = 0.0;
    for r in 0..range.n_equations {
        let v = phi[range.row_start + r];
        sumsq += v * v;
    }
    let norm = sumsq.sqrt();
    if range.n_equations == 1 {
        format!("Φ = {:+.3e}", phi[range.row_start])
    } else {
        format!("‖Φ‖ = {:.3e}", norm)
    }
}

/// Render the Lagrange multiplier(s) for a constraint as a short labelled
/// string. Returns "λ = …" with units determined by row layout:
/// - 1-row driver: torque (rev driver) or force (lin driver)
/// - 1-row joint (cam follower): contact force, units N
/// - 2-row joint (revolute / prismatic): magnitude of `[λ_x, λ_y]` in N
/// - 3-row joint (fixed): force magnitude in N + moment in N·m
///
/// Returns an empty string if `lambdas` is `None`.
pub fn lambda_string(
    mech: &Mechanism,
    idx: usize,
    lambdas: Option<&nalgebra::DVector<f64>>,
) -> String {
    let Some(lams) = lambdas else { return String::new() };
    let Some(range) = mech.constraint_ranges().get(idx) else {
        return String::new();
    };
    let Some(kind) = kind_at(mech, idx) else { return String::new() };

    match (kind, range.n_equations) {
        (EqKind::RevoluteDriver, 1) => {
            format!("λ = {:+.3} N\u{00b7}m", lams[range.row_start])
        }
        (EqKind::LinearDriver, 1) => format!("λ = {:+.3} N", lams[range.row_start]),
        (EqKind::CamFollower, 1) => format!("λ = {:+.3} N", lams[range.row_start]),
        (_, 2) => {
            let fx = lams[range.row_start];
            let fy = lams[range.row_start + 1];
            let mag = (fx * fx + fy * fy).sqrt();
            format!("‖λ‖ = {:.3} N  ({:+.3}, {:+.3})", mag, fx, fy)
        }
        (_, 3) => {
            let fx = lams[range.row_start];
            let fy = lams[range.row_start + 1];
            let mz = lams[range.row_start + 2];
            let mag = (fx * fx + fy * fy).sqrt();
            format!(
                "‖F‖ = {:.3} N, M_z = {:+.3} N\u{00b7}m  ({:+.3}, {:+.3}, {:+.3})",
                mag, mz, fx, fy, mz
            )
        }
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::samples::{build_sample, SampleMechanism};

    #[test]
    fn kind_at_classifies_fourbar_constraints() {
        let (mech, _q0) = build_sample(SampleMechanism::FourBar);
        // FourBar sample: 4 revolute joints + 1 revolute driver = 5 constraints
        assert_eq!(n_constraints(&mech), 5);
        for i in 0..4 {
            assert_eq!(kind_at(&mech, i), Some(EqKind::Revolute));
        }
        assert_eq!(kind_at(&mech, 4), Some(EqKind::RevoluteDriver));
        assert_eq!(kind_at(&mech, 5), None);
    }

    #[test]
    fn symbolic_form_revolute_starts_with_phi() {
        let (mech, _q0) = build_sample(SampleMechanism::FourBar);
        let s = symbolic_form(&mech, 0);
        assert!(s.starts_with("Φ_rev:"), "got: {}", s);
    }

    #[test]
    fn symbolic_form_revolute_driver_includes_omega_and_theta0() {
        let (mech, _q0) = build_sample(SampleMechanism::FourBar);
        let s = symbolic_form(&mech, 4);
        // Driver row should include θ_j − θ_i and an f(t) form.
        assert!(s.starts_with("Φ_rd:"), "got: {}", s);
        assert!(s.contains("θ_j − θ_i"), "got: {}", s);
    }

    #[test]
    fn constraint_index_for_row_maps_blocks() {
        let (mech, _q0) = build_sample(SampleMechanism::FourBar);
        // 4 revolute joints (2 eq each = rows 0..8) + 1 driver (1 eq = row 8).
        assert_eq!(constraint_index_for_row(&mech, 0), Some(0));
        assert_eq!(constraint_index_for_row(&mech, 1), Some(0));
        assert_eq!(constraint_index_for_row(&mech, 2), Some(1));
        assert_eq!(constraint_index_for_row(&mech, 7), Some(3));
        assert_eq!(constraint_index_for_row(&mech, 8), Some(4));
        assert_eq!(constraint_index_for_row(&mech, 9), None);
    }

    #[test]
    fn lambda_string_revolute_two_eqs_includes_norm() {
        let (mech, _q0) = build_sample(SampleMechanism::FourBar);
        // Build a fake λ vector of zeros (not solved, just shape-checking).
        let n = mech.n_constraints();
        let lams = nalgebra::DVector::from_element(n, 0.0);
        let s = lambda_string(&mech, 0, Some(&lams));
        assert!(s.starts_with("‖λ‖"), "got: {}", s);
    }

    #[test]
    fn lambda_string_driver_one_eq_includes_units() {
        let (mech, _q0) = build_sample(SampleMechanism::FourBar);
        let n = mech.n_constraints();
        let lams = nalgebra::DVector::from_element(n, 1.5);
        // Driver index = 4 in FourBar.
        let s = lambda_string(&mech, 4, Some(&lams));
        assert!(s.starts_with("λ = "), "got: {}", s);
        assert!(s.contains("N\u{00b7}m"), "expected torque units, got: {}", s);
    }

    #[test]
    fn short_labels_are_distinct() {
        let labels = [
            EqKind::Revolute.short_label(),
            EqKind::Prismatic.short_label(),
            EqKind::Fixed.short_label(),
            EqKind::CamFollower.short_label(),
            EqKind::RevoluteDriver.short_label(),
            EqKind::LinearDriver.short_label(),
        ];
        for (i, a) in labels.iter().enumerate() {
            for (j, b) in labels.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "labels {} and {} collide", i, j);
                }
            }
        }
    }
}
