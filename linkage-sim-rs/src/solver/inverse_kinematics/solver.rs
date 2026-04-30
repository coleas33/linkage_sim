//! Inverse Newton outer loop, bisection fallback, and workspace probe.
//!
//! See: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8.1
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §6

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;
use crate::error::LinkageError;
use crate::solver::kinematics::solve_position;

use super::control_target::ControlTarget;

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
}
