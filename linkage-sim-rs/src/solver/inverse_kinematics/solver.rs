//! Inverse Newton outer loop, bisection fallback, and workspace probe.
//!
//! See: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8.1
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §6

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;
use crate::error::LinkageError;
use crate::solver::assembly::assemble_jacobian;
use crate::solver::kinematics::solve_position;

use super::control_target::ControlTarget;
use super::severity::{InverseSolveStatus, Severity};

/// Result of a workspace probe — pairs of (input parameter `u_k`, observable `g(q(u_k))`)
/// across the input range.
#[derive(Debug, Clone)]
pub struct WorkspaceProbe {
    /// Sampled input values, monotonically increasing.
    pub u_samples: Vec<f64>,
    /// Observable value at each sample.
    pub g_samples: Vec<f64>,
    /// Min / max of `g` across the probe.
    pub g_min: f64,
    pub g_max: f64,
}

/// Probe the workspace by sweeping the driver parameter over `[u_min, u_max]` in `n` samples.
/// Each sample requires one forward `solve_position` call.
///
/// `u_0` is the initial driver value (= θ_0 for revolute, = L_0 for linear). The mapping
/// from `u` to the mechanism's `t`-frame is `t = (u − u_0) / nominal_rate`.
///
/// Returns `Err(LinkageError)` if any forward solve fails with a hard error
/// (e.g., `SvdSolveFailed`, `MechanismNotBuilt`). Samples that merely fail to
/// converge (Newton hit `max_iter` without reaching tolerance) are silently
/// skipped — they appear as gaps in `u_samples`/`g_samples`. If *every* sample
/// fails to converge, `g_samples` is empty and `g_min`/`g_max` will be
/// `+∞`/`-∞` sentinels; callers should detect this case explicitly.
pub fn workspace_probe(
    mech: &Mechanism,
    q_seed: &DVector<f64>,
    u_min: f64,
    u_max: f64,
    u_0: f64,
    nominal_rate: f64,
    target: &ControlTarget,
    n_samples: usize,
) -> Result<WorkspaceProbe, LinkageError> {
    assert!(n_samples >= 2, "workspace_probe requires at least 2 samples");
    assert!(nominal_rate.abs() > 1e-12, "nominal_rate must be non-zero");
    assert!(u_max > u_min, "u_max must be > u_min");

    let mut u_samples = Vec::with_capacity(n_samples);
    let mut g_samples = Vec::with_capacity(n_samples);
    let mut q_prev = q_seed.clone();

    for i in 0..n_samples {
        let frac = (i as f64) / ((n_samples - 1) as f64);
        let u = u_min + frac * (u_max - u_min);
        let t_mech = (u - u_0) / nominal_rate;

        let res = solve_position(mech, &q_prev, t_mech, 1e-10, 50)?;
        if !res.converged {
            // Skip this sample but continue probing; it'll show as a gap in g.
            continue;
        }
        u_samples.push(u);
        g_samples.push(target.evaluate(mech, &res.q));
        q_prev = res.q;
    }

    let g_min = g_samples.iter().copied().fold(f64::INFINITY, f64::min);
    let g_max = g_samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    Ok(WorkspaceProbe {
        u_samples,
        g_samples,
        g_min,
        g_max,
    })
}

/// Result of one inverse-target solve at a single sample point.
#[derive(Debug, Clone)]
pub struct InverseSolveResult {
    /// Converged input parameter.
    pub u: f64,
    /// Converged configuration.
    pub q: DVector<f64>,
    /// `g(q)` at convergence.
    pub achieved: f64,
    /// `|g(q) − target|` at convergence.
    pub residual: f64,
    /// Number of Newton outer iterations.
    pub iterations: usize,
    /// Convergence / failure status.
    pub status: InverseSolveStatus,
}

/// Solve for the input parameter `u` such that `target.evaluate(q(u)) = h`.
///
/// Outer Newton on `r(u) = g(q(u)) − h = 0`. Per-iteration:
///   1. Forward solve `q_k = solve_position(mech, q_{k-1}, t_mech)` where
///      `t_mech = (u_k − u_0) / nominal_rate`.
///   2. Compute `r_k = target.evaluate(q_k) − h`.
///   3. If `|r_k| < tol`, converged.
///   4. Compute `dq/du = −Φ_q⁻¹ Φ_u`. Φ_u has a single −1 entry on the driver row
///      (since Φ depends on u only through `−u` after re-parameterization).
///   5. `r'(u) = ∇_q g · dq/du`. Update `u_{k+1} = u_k − r_k / r'(u)`.
///
/// Each call internally runs a fresh `workspace_probe` (~64 forward solves).
/// **In hot paths (e.g. computing a trajectory of N samples against the same
/// target/range), callers should cache the probe externally** — see spec §6.6.
/// The cache lives on `AppState` and is not part of this solver layer.
///
/// Failure detection added in Task 1.11.
///
/// Math reference: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8.1.
#[allow(clippy::too_many_arguments)]
pub fn solve_for_target(
    mech: &Mechanism,
    q_seed: &DVector<f64>,
    target: &ControlTarget,
    h: f64,
    severity: Severity,
    u_range: (f64, f64),
    u_0: f64,
    nominal_rate: f64,
    tol: f64,
    max_iter: usize,
    n_probe: usize,
) -> Result<InverseSolveResult, LinkageError> {
    assert!(nominal_rate.abs() > 1e-12, "nominal_rate must be non-zero");

    // 1. Workspace probe.
    let probe = workspace_probe(
        mech, q_seed, u_range.0, u_range.1, u_0, nominal_rate, target, n_probe,
    )?;

    if probe.g_samples.is_empty() {
        return Err(LinkageError::SvdSolveFailed);
    }

    // 2a. Reachability check. Slack accommodates probe granularity: between
    // adjacent samples `g` can vary by up to ~|g_max - g_min| × (1 / n_probe),
    // so a fraction of the workspace span is the appropriate scale (much
    // larger than mere FP noise).
    let workspace_span = (probe.g_max - probe.g_min).abs();
    let reachability_slack = (1e-3 * workspace_span).max(1e-9);
    if h < probe.g_min - reachability_slack || h > probe.g_max + reachability_slack {
        let achieved_clamp = h.clamp(probe.g_min, probe.g_max);
        let status = InverseSolveStatus::Reachability {
            target: h,
            achieved_clamp,
            workspace_min: Some(probe.g_min),
            workspace_max: Some(probe.g_max),
        };
        return classify_or_fail(severity, status, q_seed.clone(), 0.0);
    }

    // 2b. Bracket: find u_seed in the probe whose g is closest to h.
    let mut closest_idx = 0usize;
    let mut closest_diff = f64::INFINITY;
    for (i, g_i) in probe.g_samples.iter().enumerate() {
        let d = (g_i - h).abs();
        if d < closest_diff {
            closest_diff = d;
            closest_idx = i;
        }
    }
    let u_seed = probe.u_samples[closest_idx];

    // 3. Forward solve at u_seed.
    let mut u_k = u_seed;
    let t_mech_seed = (u_seed - u_0) / nominal_rate;
    let mut q_k = solve_position(mech, q_seed, t_mech_seed, 1e-10, 50)?.q;

    let driver_row = mech.driver_row();

    // 4. Outer Newton.
    let mut iterations = 0usize;
    let mut achieved = target.evaluate(mech, &q_k);
    let mut residual = (achieved - h).abs();
    let mut converged = residual < tol;
    let branch_threshold = 0.5 * max_body_length(mech);

    for k in 1..=max_iter {
        iterations = k;
        if converged {
            break;
        }

        let t_mech_k = (u_k - u_0) / nominal_rate;
        let phi_q = assemble_jacobian(mech, &q_k, t_mech_k);
        let mut phi_u = DVector::zeros(mech.n_constraints());
        phi_u[driver_row] = -1.0;
        let svd = phi_q.svd(true, true);
        let dq_du = svd.solve(&-phi_u, 1e-14)
            .map_err(|_| LinkageError::SvdSolveFailed)?;
        let grad = target.gradient(mech, &q_k);
        let r_prime = grad.dot(&dq_du);

        let grad_norm = grad.norm();
        let dqdu_norm = dq_du.norm();
        let eps_singularity = 1e-6 * (grad_norm * dqdu_norm).max(1e-12);
        if r_prime.abs() < eps_singularity {
            let status = InverseSolveStatus::Singularity { dg_du: r_prime };
            return classify_or_fail(severity, status, q_k, u_k);
        }

        u_k = (u_k - (achieved - h) / r_prime).clamp(u_range.0, u_range.1);

        let q_prev = q_k.clone();
        let t_mech_next = (u_k - u_0) / nominal_rate;
        q_k = solve_position(mech, &q_prev, t_mech_next, 1e-10, 50)?.q;

        let delta_norm = body_translation_delta(mech, &q_k, &q_prev);
        if delta_norm > branch_threshold {
            let status = InverseSolveStatus::BranchJump { delta_q_norm: delta_norm };
            return classify_or_fail(severity, status, q_prev, u_k);
        }

        achieved = target.evaluate(mech, &q_k);
        residual = (achieved - h).abs();
        converged = residual < tol;
    }

    if !converged {
        let s = InverseSolveStatus::NonConvergent { iterations, residual };
        return classify_or_fail(severity, s, q_k, u_k);
    }

    Ok(InverseSolveResult {
        u: u_k,
        q: q_k,
        achieved,
        residual,
        iterations,
        status: InverseSolveStatus::Converged,
    })
}

/// In `Strict` mode, convert a non-`Converged` status to `Err(LinkageError::...)`.
/// In `Analysis` mode, return an `InverseSolveResult` populated with the failure status.
fn classify_or_fail(
    severity: Severity,
    status: InverseSolveStatus,
    partial_q: DVector<f64>,
    partial_u: f64,
) -> Result<InverseSolveResult, LinkageError> {
    match severity {
        Severity::Strict => Err(LinkageError::from(status)),
        Severity::Analysis => {
            // Achieved/residual fields are zero in failure cases — caller can
            // recompute from partial_q if needed for diagnostic display.
            Ok(InverseSolveResult {
                u: partial_u,
                q: partial_q,
                achieved: 0.0,
                residual: 0.0,
                iterations: 0,
                status,
            })
        }
    }
}

fn max_body_length(mech: &Mechanism) -> f64 {
    use crate::core::state::GROUND_ID;
    let mut max_len = 0.0_f64;
    for (id, body) in mech.bodies() {
        if id == GROUND_ID {
            continue;
        }
        // Approximate body extent as 2 × max joint distance from CG. We use
        // |cg_local| as a proxy; the simulator's bars store length in their
        // BlueprintBody. For trajectory purposes this is a heuristic only.
        let cg = body.cg_local;
        let extent = (cg.x * cg.x + cg.y * cg.y).sqrt() * 2.0;
        if extent > max_len {
            max_len = extent;
        }
    }
    // Floor: at least 1 cm to avoid tiny mechanisms being too sensitive.
    max_len.max(0.01)
}

/// Translational-only ‖Δq‖ for branch-jump detection.
///
/// The state vector q mixes meters (x, y) and radians (θ) per body, so a raw
/// ‖q_k − q_{k-1}‖ is unit-mismatched against `branch_jump_threshold` (which
/// is in meters). This helper extracts only the (x, y) components per body,
/// giving a true translational displacement comparable to body length.
///
/// See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §6.7
fn body_translation_delta(mech: &Mechanism, q_a: &DVector<f64>, q_b: &DVector<f64>) -> f64 {
    let state = mech.state();
    let mut sum_sq = 0.0_f64;
    for body_id in mech.body_order() {
        if let Ok((start, _)) = state.body_coord_range(body_id) {
            let dx = q_a[start] - q_b[start];
            let dy = q_a[start + 1] - q_b[start + 1];
            sum_sq += dx * dx + dy * dy;
        }
    }
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::inverse_kinematics::test_helpers::{build_fourbar, solve_at};
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    fn probe_angle_target_spans_expected_range() {
        let mech = build_fourbar();
        let q0 = solve_at(&mech, 0.0);
        let target = ControlTarget::angle("crank");
        let probe = workspace_probe(
            &mech, &q0, 0.0, 2.0 * PI, 0.0, 2.0 * PI, &target, 16,
        ).unwrap();
        // Fully-rotating canonical 4-bar should converge at all 16 samples; loosening
        // this would mask a regression. If a future linkage variant genuinely traverses
        // a non-convergent region, that test should use its own (looser) bound.
        assert_eq!(probe.u_samples.len(), 16, "all samples should converge for a fully-rotating crank");
        // crank angle should span roughly [-π/2, +3π/2] modulo 2π
        // simple check: g_max − g_min ≈ 2π (or close to it for full revolution)
        assert!(probe.g_max - probe.g_min > PI);
    }

    #[test]
    fn probe_records_min_and_max() {
        let mech = build_fourbar();
        let q0 = solve_at(&mech, 0.0);
        let target = ControlTarget::angle("crank");
        let probe = workspace_probe(
            &mech, &q0, 0.0, PI / 2.0, 0.0, 2.0 * PI, &target, 8,
        ).unwrap();
        let manual_min = probe.g_samples.iter().copied().fold(f64::INFINITY, f64::min);
        let manual_max = probe.g_samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_abs_diff_eq!(probe.g_min, manual_min, epsilon = 1e-12);
        assert_abs_diff_eq!(probe.g_max, manual_max, epsilon = 1e-12);
    }

    #[test]
    fn solve_for_target_angle_round_trip() {
        let mech = build_fourbar();
        let q0 = solve_at(&mech, 0.0);
        let target = ControlTarget::angle("crank");
        // Known crank angle: π/3
        let h = PI / 3.0;
        let res = solve_for_target(
            &mech, &q0, &target, h,
            Severity::Analysis,
            (0.0, 2.0 * PI), 0.0, 2.0 * PI,
            1e-8, 50, 64,
        ).unwrap();
        assert!(matches!(res.status, InverseSolveStatus::Converged));
        assert_abs_diff_eq!(res.achieved, h, epsilon = 1e-7);
    }

    #[test]
    fn solve_for_target_world_y_round_trip() {
        let mech = build_fourbar();
        let q0 = solve_at(&mech, 0.0);
        let target = ControlTarget::world_y("crank", [0.005, 0.0]);
        // Achievable target: y position of crank tip after rotation
        let h = 0.005; // achievable
        let res = solve_for_target(
            &mech, &q0, &target, h,
            Severity::Analysis,
            (0.0, 2.0 * PI), 0.0, 2.0 * PI,
            1e-8, 50, 64,
        ).unwrap();
        assert!(matches!(res.status, InverseSolveStatus::Converged));
        assert_abs_diff_eq!(res.achieved, h, epsilon = 1e-7);
    }

    #[test]
    fn reachability_failure_in_analysis_mode() {
        let mech = build_fourbar();
        let q0 = solve_at(&mech, 0.0);
        let target = ControlTarget::angle("crank");
        // Crank angle = 100 rad is unreachable in [0, 2π]
        let h = 100.0;
        let res = solve_for_target(
            &mech, &q0, &target, h,
            Severity::Analysis,
            (0.0, 2.0 * PI), 0.0, 2.0 * PI,
            1e-8, 50, 64,
        ).unwrap();
        assert!(matches!(res.status, InverseSolveStatus::Reachability { .. }));
    }

    #[test]
    fn reachability_failure_in_strict_mode_returns_err() {
        let mech = build_fourbar();
        let q0 = solve_at(&mech, 0.0);
        let target = ControlTarget::angle("crank");
        let h = 100.0;
        let res = solve_for_target(
            &mech, &q0, &target, h,
            Severity::Strict,
            (0.0, 2.0 * PI), 0.0, 2.0 * PI,
            1e-8, 50, 64,
        );
        assert!(matches!(res, Err(LinkageError::TrajectoryUnreachable { .. })));
    }
}
