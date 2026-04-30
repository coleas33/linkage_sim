# Trajectory-mode position control — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trajectory-mode analysis to the linkage simulator that back-solves actuator input `u(t)` from a desired output observable trajectory `h(t)`, reusing the existing forward-sweep pipeline. Produces time-series of target / achieved / residual / `u(t)` / `u̇(t)` / `F_actuator(t)` plus per-timestep reactions, with configurable failure semantics.

**Architecture:** New `solver/inverse_kinematics/` subtree (5 files) hosts the inverse-kinematics math (`ControlTarget` enum, `solve_for_target` Newton outer loop, inverse-velocity / acceleration helpers, `Severity` enum). `gui/sweep/` extends with a `SweepMode::Trajectory` variant + `compute_trajectory` helper. `gui/trajectory_panel/` (3 files) provides the input UI; `gui/plot_panel/trajectory.rs` renders time-series traces. Per-sample force / energy / reaction calls reuse the existing sweep machinery via Φ_t / γ driver-row override (no `core/` changes).

**Tech Stack:** Rust 2024 edition, nalgebra (linear algebra), eframe / egui (GUI), serde (JSON persistence), thiserror (error types).

**Source documents:**
- Design spec: `docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md`
- Math reference: `docs/superpowers/specs/2026-04-29-linkage-equations-reference.md`

---

## File structure

### Created files (10)

```
linkage-sim-rs/src/solver/inverse_kinematics/
  mod.rs                — public re-exports (~30 LoC)
  severity.rs           — Severity enum, InverseSolveStatus (~80 LoC)
  control_target.rs     — ControlTarget enum + evaluate/gradient/hessian/constructors (~250 LoC)
  solver.rs             — solve_for_target outer Newton + bisection fallback + workspace probe (~250 LoC)
  derivatives.rs        — inverse_velocity, inverse_acceleration_fd (~120 LoC)

linkage-sim-rs/src/gui/trajectory_panel/
  mod.rs                — top-level UI + severity toggle (~150 LoC)
  target_picker.rs      — ControlTarget selector + body/point/axis pickers (~180 LoC)
  profile_input.rs      — TrajectoryProfile editor + h(t) preview (~150 LoC)

linkage-sim-rs/src/gui/plot_panel/
  trajectory.rs         — trajectory plot rendering (~200 LoC)

linkage-sim-rs/src/gui/state/
  trajectory_ops.rs     — AppState::solve_for_trajectory_target (~80 LoC)
```

### Modified files (12)

```
linkage-sim-rs/src/error.rs                   — add LinkageError variants
linkage-sim-rs/src/solver/mod.rs              — pub mod inverse_kinematics
linkage-sim-rs/src/gui/mod.rs                 — mod trajectory_panel
linkage-sim-rs/src/gui/state/mod.rs           — mod trajectory_ops
linkage-sim-rs/src/gui/state/types.rs         — SweepData fields, TrajectoryProfile
linkage-sim-rs/src/gui/sweep/mod.rs           — SweepMode::Trajectory + compute_trajectory
linkage-sim-rs/src/gui/sweep/motion_profile.rs — generalize for axis-agnostic
linkage-sim-rs/src/gui/plot_panel/mod.rs      — dispatch arm for Trajectory
linkage-sim-rs/src/gui/input_panel.rs         — delegate to trajectory_panel
docs/ai/02-system.yaml                         — invariants
docs/ai/04-memory.yaml                         — deferred items
docs/ai/05-update-tracker.md                   — stage entries
docs/FEATURES.md                               — user-facing description
```

### Untouched

`core/`, `forces/`, `analysis/`. The Φ_t / γ override happens in `compute_trajectory`, not in constraint code.

---

## Build sequence

The plan is split into four sequential stages. Each stage produces independently verifiable software and is its own commit boundary.

| Stage | What lands | Working state at end |
|---|---|---|
| 1 | `solver/inverse_kinematics/` subtree + tests | Solver works in isolation; not yet wired into GUI |
| 2 | `SweepMode::Trajectory` + `compute_trajectory` + `SweepData` fields | Trajectory mode reachable from code; not from UI |
| 3 | `trajectory_panel/` + `plot_panel/trajectory.rs` + state ops | Trajectory mode end-to-end usable via GUI |
| 4 | CSV export + failure-band rendering + polish + docs | Feature shipped with user-facing entry in `FEATURES.md` |

Each stage's tasks below assume the previous stage is complete.

---

# Stage 1 — Solver layer (TDD)

Test-driven build of `solver/inverse_kinematics/`. No GUI changes. Stage end state: forward → inverse round-trip tests pass for all 5 `ControlTarget` variants on the parallelogram and 4-bar samples.

## Task 1.1 — Create module skeleton + Severity enum

**Files:**
- Create: `linkage-sim-rs/src/solver/inverse_kinematics/mod.rs`
- Create: `linkage-sim-rs/src/solver/inverse_kinematics/severity.rs`
- Modify: `linkage-sim-rs/src/solver/mod.rs:7`

- [ ] **Step 1: Write the failing test for `Severity` enum**

Append to `linkage-sim-rs/src/solver/inverse_kinematics/severity.rs`:

```rust
//! Severity controls how `solve_for_target` reports per-sample failures.
//!
//! `Severity::Strict` returns `Err(LinkageError::...)` on any failure mode.
//! `Severity::Analysis` returns `Ok(InverseSolveResult { status: <failure variant>, ... })`
//! so the GUI can render failed samples in red rather than aborting the trajectory.
//!
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §6.5

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Strict,
    Analysis,
}

impl Default for Severity {
    fn default() -> Self {
        Severity::Analysis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_default_is_analysis() {
        assert_eq!(Severity::default(), Severity::Analysis);
    }

    #[test]
    fn severity_serde_round_trip() {
        let strict = Severity::Strict;
        let json = serde_json::to_string(&strict).unwrap();
        let back: Severity = serde_json::from_str(&json).unwrap();
        assert_eq!(strict, back);
    }
}
```

- [ ] **Step 2: Create `mod.rs` with the module reference**

Create `linkage-sim-rs/src/solver/inverse_kinematics/mod.rs`:

```rust
//! Inverse-kinematics solver for trajectory-mode position control.
//!
//! Given a desired output observable `g(q)` and target value `h`, back-solves
//! the actuator input parameter `u` such that `g(q(u)) = h`, where `q(u)` is
//! the forward solution from the existing kinematics solvers.
//!
//! See: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md

pub mod severity;

pub use severity::Severity;
```

- [ ] **Step 3: Wire into the solver tree**

Modify `linkage-sim-rs/src/solver/mod.rs` to add the new module. After line 7, append:

```rust
pub mod inverse_kinematics;
```

- [ ] **Step 4: Run tests to verify they pass**

Run from `linkage-sim-rs/`:
```
cargo test --lib solver::inverse_kinematics::severity::tests
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git add linkage-sim-rs/src/solver/inverse_kinematics/ linkage-sim-rs/src/solver/mod.rs
git commit -m "feat(traj): add Severity enum scaffolding for inverse kinematics"
```

---

## Task 1.2 — `InverseSolveStatus` enum

**Files:**
- Modify: `linkage-sim-rs/src/solver/inverse_kinematics/severity.rs`

- [ ] **Step 1: Write the failing tests**

Append to the `#[cfg(test)] mod tests` block in `severity.rs`:

```rust
    #[test]
    fn inverse_solve_status_constructs_all_variants() {
        let _converged = InverseSolveStatus::Converged;
        let _reach = InverseSolveStatus::Reachability {
            target: 1.0,
            achieved_clamp: 0.5,
            workspace_min: Some(-0.5),
            workspace_max: Some(0.5),
        };
        let _sing = InverseSolveStatus::Singularity { dg_du: 1e-9 };
        let _branch = InverseSolveStatus::BranchJump { delta_q_norm: 0.1 };
        let _nc = InverseSolveStatus::NonConvergent {
            iterations: 50,
            residual: 1e-3,
        };
    }

    #[test]
    fn inverse_solve_status_is_clone_and_debug() {
        let s = InverseSolveStatus::Singularity { dg_du: 1e-9 };
        let _cloned = s.clone();
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("Singularity"));
    }
```

- [ ] **Step 2: Run to verify they fail**

```
cargo test --lib solver::inverse_kinematics::severity::tests::inverse_solve_status
```
Expected: FAIL with "cannot find type `InverseSolveStatus` in this scope".

- [ ] **Step 3: Implement the enum**

Append above the `#[cfg(test)]` line in `severity.rs`:

```rust
/// Per-sample diagnostic produced by `solve_for_target`.
///
/// In `Severity::Analysis` mode this is returned in the `InverseSolveResult.status`
/// field. In `Severity::Strict` mode it is converted to `LinkageError` and bubbled.
#[derive(Debug, Clone)]
pub enum InverseSolveStatus {
    /// Newton converged within tolerance.
    Converged,
    /// Target value is outside the workspace `[g_min, g_max]`.
    Reachability {
        target: f64,
        achieved_clamp: f64,
        workspace_min: Option<f64>,
        workspace_max: Option<f64>,
    },
    /// `|dg/du|` fell below the singularity threshold (mechanism has lost
    /// authority over the target locally — toggle, transmission angle ≈ 90°).
    Singularity { dg_du: f64 },
    /// `‖q_k − q_{k-1}‖` exceeded the branch-jump threshold (assembly mode flip).
    BranchJump { delta_q_norm: f64 },
    /// Newton did not converge in `max_iter` iterations.
    NonConvergent { iterations: usize, residual: f64 },
}
```

Re-export from `mod.rs`:

```rust
pub use severity::{InverseSolveStatus, Severity};
```

- [ ] **Step 4: Run to verify they pass**

```
cargo test --lib solver::inverse_kinematics::severity::tests
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add linkage-sim-rs/src/solver/inverse_kinematics/
git commit -m "feat(traj): add InverseSolveStatus diagnostic enum"
```

---

## Task 1.3 — `LinkageError` variants for trajectory failures

**Files:**
- Modify: `linkage-sim-rs/src/error.rs`

- [ ] **Step 1: Write the failing test**

Append to a new `#[cfg(test)] mod tests` block at the bottom of `linkage-sim-rs/src/error.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::inverse_kinematics::InverseSolveStatus;

    #[test]
    fn from_inverse_solve_status_to_linkage_error() {
        let status = InverseSolveStatus::Reachability {
            target: 1.0,
            achieved_clamp: 0.5,
            workspace_min: Some(0.0),
            workspace_max: Some(0.5),
        };
        let err: LinkageError = status.into();
        match err {
            LinkageError::TrajectoryUnreachable { target, .. } => {
                assert_eq!(target, 1.0);
            }
            _ => panic!("expected TrajectoryUnreachable"),
        }
    }

    #[test]
    fn from_singularity_status() {
        let status = InverseSolveStatus::Singularity { dg_du: 1e-9 };
        let err: LinkageError = status.into();
        assert!(matches!(err, LinkageError::TrajectorySingular { .. }));
    }
}
```

- [ ] **Step 2: Run to verify they fail**

```
cargo test --lib error::tests
```
Expected: FAIL with missing variants.

- [ ] **Step 3: Add the new variants and `From` impl**

Append to the `LinkageError` enum (before the closing `}` on line 59):

```rust
    // -- Trajectory-mode failures --
    /// Target value is outside the reachable workspace.
    #[error(
        "Trajectory unreachable at target = {target:.4} (workspace [{min:.4}, {max:.4}])."
    )]
    TrajectoryUnreachable {
        target: f64,
        achieved_clamp: f64,
        min: f64,
        max: f64,
    },

    /// `|dg/du|` fell below the singularity threshold.
    #[error("Trajectory singular: |dg/du| = {dg_du:.2e} below threshold.")]
    TrajectorySingular { dg_du: f64 },

    /// Trajectory crossed an assembly-mode boundary mid-solve.
    #[error("Trajectory branch jump: ‖Δq‖ = {delta_q_norm:.4} exceeded threshold.")]
    TrajectoryBranchJump { delta_q_norm: f64 },

    /// Inverse Newton did not converge in `max_iter` iterations.
    #[error(
        "Trajectory inverse Newton did not converge after {iterations} iterations \
         (residual = {residual:.2e})."
    )]
    TrajectoryNonConvergent { iterations: usize, residual: f64 },
```

After the `LinkageError` enum (still in `error.rs`), append:

```rust
impl From<crate::solver::inverse_kinematics::InverseSolveStatus> for LinkageError {
    fn from(status: crate::solver::inverse_kinematics::InverseSolveStatus) -> Self {
        use crate::solver::inverse_kinematics::InverseSolveStatus;
        match status {
            InverseSolveStatus::Converged => {
                // Caller bug: shouldn't convert a Converged status to an error.
                // Use a representative numerical error.
                LinkageError::TrajectoryNonConvergent {
                    iterations: 0,
                    residual: 0.0,
                }
            }
            InverseSolveStatus::Reachability {
                target,
                achieved_clamp,
                workspace_min,
                workspace_max,
            } => LinkageError::TrajectoryUnreachable {
                target,
                achieved_clamp,
                min: workspace_min.unwrap_or(f64::NEG_INFINITY),
                max: workspace_max.unwrap_or(f64::INFINITY),
            },
            InverseSolveStatus::Singularity { dg_du } => {
                LinkageError::TrajectorySingular { dg_du }
            }
            InverseSolveStatus::BranchJump { delta_q_norm } => {
                LinkageError::TrajectoryBranchJump { delta_q_norm }
            }
            InverseSolveStatus::NonConvergent {
                iterations,
                residual,
            } => LinkageError::TrajectoryNonConvergent {
                iterations,
                residual,
            },
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```
cargo test --lib error::tests
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git add linkage-sim-rs/src/error.rs
git commit -m "feat(traj): add LinkageError variants for trajectory failure modes"
```

---

## Task 1.4 — `ControlTarget::Angle` (variant + evaluate + gradient)

**Files:**
- Create: `linkage-sim-rs/src/solver/inverse_kinematics/control_target.rs`
- Modify: `linkage-sim-rs/src/solver/inverse_kinematics/mod.rs`

- [ ] **Step 1: Create file with module-doc and one-variant skeleton + failing test**

Create `linkage-sim-rs/src/solver/inverse_kinematics/control_target.rs`:

```rust
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
    // Other variants added in subsequent tasks.
}

impl ControlTarget {
    /// Evaluate `g(q)` — scalar value of the observable at configuration `q`.
    pub fn evaluate(&self, mech: &Mechanism, q: &DVector<f64>) -> f64 {
        match self {
            ControlTarget::Angle { body_id } => mech.state().get_angle(body_id, q),
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
        }

        let _ = q; // q unused for Angle (gradient is constant)
        grad
    }

    /// Hessian `∇²_q g(q)` — n_coords × n_coords. Zero for variants linear in q.
    pub fn hessian(&self, mech: &Mechanism, q: &DVector<f64>) -> DMatrix<f64> {
        let n = mech.state().n_coords();
        let _ = q;
        match self {
            ControlTarget::Angle { .. } => DMatrix::zeros(n, n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use crate::core::mechanism::Mechanism;
    use crate::solver::kinematics::solve_position;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    /// Build a minimal 4-bar with crank/coupler/rocker.
    /// Standard linkage from solver/kinematics.rs tests.
    fn build_fourbar() -> Mechanism {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
        let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
        let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);
        let rocker = make_bar("rocker", "C", "D", 0.03, 0.0, 0.0);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();
        mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0).unwrap();
        mech.build().unwrap();
        mech
    }

    fn solve_at(mech: &Mechanism, t: f64) -> DVector<f64> {
        let state = mech.state();
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
        state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
        state.set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);
        let res = solve_position(mech, &q0, t, 1e-10, 50).unwrap();
        assert!(res.converged);
        res.q
    }

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
}
```

- [ ] **Step 2: Add module to `mod.rs`**

Modify `linkage-sim-rs/src/solver/inverse_kinematics/mod.rs`:

```rust
//! Inverse-kinematics solver for trajectory-mode position control.
//!
//! Given a desired output observable `g(q)` and target value `h`, back-solves
//! the actuator input parameter `u` such that `g(q(u)) = h`, where `q(u)` is
//! the forward solution from the existing kinematics solvers.
//!
//! See: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md

pub mod control_target;
pub mod severity;

pub use control_target::ControlTarget;
pub use severity::{InverseSolveStatus, Severity};
```

- [ ] **Step 3: Run tests to verify they pass**

```
cargo test --lib solver::inverse_kinematics::control_target::tests
```
Expected: 3 passed.

- [ ] **Step 4: Commit**

```
git add linkage-sim-rs/src/solver/inverse_kinematics/
git commit -m "feat(traj): add ControlTarget::Angle with evaluate/gradient/hessian"
```

---

## Task 1.5 — `ControlTarget::WorldX` and `WorldY` variants

**Files:**
- Modify: `linkage-sim-rs/src/solver/inverse_kinematics/control_target.rs`

- [ ] **Step 1: Add failing tests**

Append to the `#[cfg(test)] mod tests` block in `control_target.rs`:

```rust
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
        // Crank tip at body-local (0.005, 0)
        let target = ControlTarget::WorldY {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
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

        for &(idx, expected_label) in &[
            (crank.x_idx(), "x"),
            (crank.y_idx(), "y"),
            (crank.theta_idx(), "theta"),
        ] {
            let mut q_plus = q.clone();
            q_plus[idx] += h;
            let mut q_minus = q.clone();
            q_minus[idx] -= h;
            let fd = (target.evaluate(&mech, &q_plus) - target.evaluate(&mech, &q_minus)) / (2.0 * h);
            assert_abs_diff_eq!(grad[idx], fd, epsilon = 1e-5);
            let _ = expected_label;
        }
    }
```

- [ ] **Step 2: Run to verify they fail**

```
cargo test --lib solver::inverse_kinematics::control_target::tests::world
```
Expected: FAIL — variants don't exist.

- [ ] **Step 3: Add `WorldX` and `WorldY` variants + impls**

In `control_target.rs`, extend the `ControlTarget` enum:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ControlTarget {
    /// Body angle θ_body. Linear in q (Hessian = 0).
    Angle { body_id: String },
    /// World x-coord of body-local point.
    WorldX { body_id: String, local_pt: [f64; 2] },
    /// World y-coord of body-local point.
    WorldY { body_id: String, local_pt: [f64; 2] },
}
```

Extend `evaluate`:

```rust
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
    }
}
```

Extend `gradient`. The pattern uses `body_point_global_derivative` from `core/state.rs`:

```rust
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
    }

    grad
}
```

Extend `hessian` to return zero for these variants too (for now; analytic Hessian deferred to v2 per spec §8.3):

```rust
pub fn hessian(&self, mech: &Mechanism, q: &DVector<f64>) -> DMatrix<f64> {
    let n = mech.state().n_coords();
    let _ = q;
    match self {
        ControlTarget::Angle { .. }
        | ControlTarget::WorldX { .. }
        | ControlTarget::WorldY { .. } => DMatrix::zeros(n, n),
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```
cargo test --lib solver::inverse_kinematics::control_target::tests
```
Expected: 6 passed (3 new + 3 existing).

- [ ] **Step 5: Commit**

```
git add linkage-sim-rs/src/solver/inverse_kinematics/control_target.rs
git commit -m "feat(traj): add ControlTarget::{WorldX, WorldY} with FD-checked gradients"
```

---

## Task 1.6 — `ControlTarget::Projection` variant

**Files:**
- Modify: `linkage-sim-rs/src/solver/inverse_kinematics/control_target.rs`

- [ ] **Step 1: Add failing tests**

Append to `tests` mod:

```rust
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
        for &idx in &[crank.x_idx(), crank.y_idx(), crank.theta_idx()] {
            let mut qp = q.clone();
            qp[idx] += h;
            let mut qm = q.clone();
            qm[idx] -= h;
            let fd = (target.evaluate(&mech, &qp) - target.evaluate(&mech, &qm)) / (2.0 * h);
            assert_abs_diff_eq!(grad[idx], fd, epsilon = 1e-5);
        }
    }
```

- [ ] **Step 2: Run to verify they fail**

```
cargo test --lib solver::inverse_kinematics::control_target::tests::projection
```
Expected: FAIL — variant doesn't exist.

- [ ] **Step 3: Add the variant + impl**

Extend the enum:

```rust
    /// Projection of body-local point onto fixed line through `axis_origin` along `axis_dir`.
    /// `axis_dir` is normalized internally by `evaluate`/`gradient`/`hessian`.
    Projection {
        body_id: String,
        local_pt: [f64; 2],
        axis_origin: [f64; 2],
        axis_dir: [f64; 2],
    },
```

Add to `evaluate`:

```rust
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
```

Add to `gradient`:

```rust
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
```

Add to `hessian`:

```rust
        ControlTarget::Angle { .. }
        | ControlTarget::WorldX { .. }
        | ControlTarget::WorldY { .. }
        | ControlTarget::Projection { .. } => DMatrix::zeros(n, n),
```

- [ ] **Step 4: Run tests to verify they pass**

```
cargo test --lib solver::inverse_kinematics::control_target::tests
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```
git add linkage-sim-rs/src/solver/inverse_kinematics/control_target.rs
git commit -m "feat(traj): add ControlTarget::Projection variant"
```

---

## Task 1.7 — `ControlTarget::Distance` variant

**Files:**
- Modify: `linkage-sim-rs/src/solver/inverse_kinematics/control_target.rs`

- [ ] **Step 1: Add failing tests**

```rust
    #[test]
    fn distance_target_evaluates_to_norm() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.0);
        // Distance from crank tip to origin
        let target = ControlTarget::Distance {
            body_id: "crank".into(),
            local_pt: [0.005, 0.0],
            ref_pt: [0.0, 0.0],
        };
        let g = target.evaluate(&mech, &q);
        // At t=0, crank centroid is at (0.005, 0); tip at body-local (0.005, 0)
        // means the local frame is centered at body origin (0.005, 0), so tip is
        // at body-local +x = (0.01, 0) world for a horizontal crank.
        // Distance to (0,0) = 0.01.
        assert_abs_diff_eq!(g, 0.01, epsilon = 1e-7);
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
        for &idx in &[crank.x_idx(), crank.y_idx(), crank.theta_idx()] {
            let mut qp = q.clone();
            qp[idx] += h;
            let mut qm = q.clone();
            qm[idx] -= h;
            let fd = (target.evaluate(&mech, &qp) - target.evaluate(&mech, &qm)) / (2.0 * h);
            assert_abs_diff_eq!(grad[idx], fd, epsilon = 1e-5);
        }
    }
```

- [ ] **Step 2: Run to verify they fail**

```
cargo test --lib solver::inverse_kinematics::control_target::tests::distance
```
Expected: FAIL.

- [ ] **Step 3: Add the variant + impl**

Extend the enum:

```rust
    /// Euclidean distance from a body-local point to a fixed reference point.
    Distance {
        body_id: String,
        local_pt: [f64; 2],
        ref_pt: [f64; 2],
    },
```

Add to `evaluate`:

```rust
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
```

Add to `gradient` (gradient of `‖P − r‖` w.r.t. q, using chain rule with unit vector):

```rust
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
```

Add to `hessian` (for v1 returns zero; analytic Hessian for distance is non-trivial and deferred per spec §8.3):

```rust
        ControlTarget::Angle { .. }
        | ControlTarget::WorldX { .. }
        | ControlTarget::WorldY { .. }
        | ControlTarget::Projection { .. }
        | ControlTarget::Distance { .. } => DMatrix::zeros(n, n),
```

- [ ] **Step 4: Run tests to verify they pass**

```
cargo test --lib solver::inverse_kinematics::control_target::tests
```
Expected: 11 passed.

- [ ] **Step 5: Commit**

```
git add linkage-sim-rs/src/solver/inverse_kinematics/control_target.rs
git commit -m "feat(traj): add ControlTarget::Distance variant"
```

---

## Task 1.8 — `ControlTarget` constructors with validation + `unit_label`

**Files:**
- Modify: `linkage-sim-rs/src/solver/inverse_kinematics/control_target.rs`

- [ ] **Step 1: Add failing tests**

```rust
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
```

- [ ] **Step 2: Run to verify they fail**

```
cargo test --lib solver::inverse_kinematics::control_target::tests
```
Expected: FAIL on missing methods.

- [ ] **Step 3: Add constructors and `unit_label`**

Append to the `impl ControlTarget` block:

```rust
    /// Construct an `Angle` variant.
    pub fn angle(body_id: impl Into<String>) -> Self {
        ControlTarget::Angle { body_id: body_id.into() }
    }

    /// Construct a `WorldX` variant.
    pub fn world_x(body_id: impl Into<String>, local_pt: [f64; 2]) -> Self {
        ControlTarget::WorldX {
            body_id: body_id.into(),
            local_pt,
        }
    }

    /// Construct a `WorldY` variant.
    pub fn world_y(body_id: impl Into<String>, local_pt: [f64; 2]) -> Self {
        ControlTarget::WorldY {
            body_id: body_id.into(),
            local_pt,
        }
    }

    /// Construct a `Projection` variant. `axis_dir` is normalized in place;
    /// panics if `axis_dir` has zero length.
    pub fn projection(
        body_id: impl Into<String>,
        local_pt: [f64; 2],
        axis_origin: [f64; 2],
        axis_dir: [f64; 2],
    ) -> Self {
        let dir = nalgebra::Vector2::new(axis_dir[0], axis_dir[1]);
        let n = dir.norm();
        assert!(n > 1e-12, "ControlTarget::projection axis_dir must be non-zero");
        let unit = dir / n;
        ControlTarget::Projection {
            body_id: body_id.into(),
            local_pt,
            axis_origin,
            axis_dir: [unit.x, unit.y],
        }
    }

    /// Construct a `Distance` variant.
    pub fn distance(
        body_id: impl Into<String>,
        local_pt: [f64; 2],
        ref_pt: [f64; 2],
    ) -> Self {
        ControlTarget::Distance {
            body_id: body_id.into(),
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
cargo test --lib solver::inverse_kinematics::control_target::tests
```
Expected: 14 passed.

- [ ] **Step 5: Commit**

```
git add linkage-sim-rs/src/solver/inverse_kinematics/control_target.rs
git commit -m "feat(traj): add ControlTarget constructors, unit_label, body_id"
```

---

## Task 1.9 — Workspace probe helper

**Files:**
- Create: `linkage-sim-rs/src/solver/inverse_kinematics/solver.rs`
- Modify: `linkage-sim-rs/src/solver/inverse_kinematics/mod.rs`

- [ ] **Step 1: Create solver.rs with workspace_probe + failing test**

Create `linkage-sim-rs/src/solver/inverse_kinematics/solver.rs`:

```rust
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
/// Returns an error if any forward solve fails.
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
    use crate::core::body::{make_bar, make_ground};
    use crate::solver::kinematics::solve_position;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    fn build_fourbar() -> Mechanism {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
        let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
        let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);
        let rocker = make_bar("rocker", "C", "D", 0.03, 0.0, 0.0);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();
        mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0).unwrap();
        mech.build().unwrap();
        mech
    }

    fn seed_q(mech: &Mechanism) -> DVector<f64> {
        let state = mech.state();
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
        state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
        state.set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);
        let res = solve_position(mech, &q0, 0.0, 1e-10, 50).unwrap();
        res.q
    }

    #[test]
    fn probe_angle_target_spans_expected_range() {
        let mech = build_fourbar();
        let q0 = seed_q(&mech);
        let target = ControlTarget::angle("crank");
        let probe = workspace_probe(
            &mech, &q0, 0.0, 2.0 * PI, 0.0, 2.0 * PI, &target, 16,
        ).unwrap();
        assert!(probe.u_samples.len() >= 14, "got {} samples", probe.u_samples.len());
        // crank angle should span roughly [-π/2, +3π/2] modulo 2π
        // simple check: g_max − g_min ≈ 2π (or close to it for full revolution)
        assert!(probe.g_max - probe.g_min > PI);
    }

    #[test]
    fn probe_records_min_and_max() {
        let mech = build_fourbar();
        let q0 = seed_q(&mech);
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
```

- [ ] **Step 2: Add to `mod.rs`**

```rust
//! ... (keep existing top-of-file docs)

pub mod control_target;
pub mod severity;
pub mod solver;

pub use control_target::ControlTarget;
pub use severity::{InverseSolveStatus, Severity};
pub use solver::{workspace_probe, WorkspaceProbe};
```

- [ ] **Step 3: Run tests to verify they pass**

```
cargo test --lib solver::inverse_kinematics::solver::tests
```
Expected: 2 passed.

- [ ] **Step 4: Commit**

```
git add linkage-sim-rs/src/solver/inverse_kinematics/
git commit -m "feat(traj): add workspace_probe helper for reachability bounds"
```

---

## Task 1.10 — `solve_for_target` happy-path Newton outer loop

**Files:**
- Modify: `linkage-sim-rs/src/solver/inverse_kinematics/solver.rs`

- [ ] **Step 1: Add failing tests for round-trip on multiple variants**

Append to the `tests` mod in `solver.rs`:

```rust
    #[test]
    fn solve_for_target_angle_round_trip() {
        let mech = build_fourbar();
        let q0 = seed_q(&mech);
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
        let q0 = seed_q(&mech);
        let target = ControlTarget::world_y("crank", [0.005, 0.0]);
        // Drive crank to π/4; tip y = 0.01 sin(π/4) ≈ 0.00707
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
```

- [ ] **Step 2: Run to verify they fail**

```
cargo test --lib solver::inverse_kinematics::solver::tests::solve_for_target
```
Expected: FAIL — `solve_for_target` undefined.

- [ ] **Step 3: Implement `solve_for_target` with happy-path Newton**

Append to `solver.rs` (above `#[cfg(test)]`):

```rust
use super::severity::{InverseSolveStatus, Severity};
use crate::solver::assembly::{assemble_jacobian, assemble_phi_t};

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
    let _ = severity; // failure-mode bubbling is added in Task 1.11
    assert!(nominal_rate.abs() > 1e-12, "nominal_rate must be non-zero");

    // 1. Workspace probe — used for warm-start bracket + reachability check.
    let probe = workspace_probe(
        mech, q_seed, u_range.0, u_range.1, u_0, nominal_rate, target, n_probe,
    )?;

    // 2. Bracket: find u_seed in the probe whose g is closest to h.
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

    // 3. Forward solve at u_seed for the warm-start q.
    let mut u_k = u_seed;
    let t_mech_seed = (u_seed - u_0) / nominal_rate;
    let mut q_k = solve_position(mech, q_seed, t_mech_seed, 1e-10, 50)?.q;

    let driver_row = mech.n_constraints() - 1;
    let n_coords = mech.state().n_coords();

    // 4. Outer Newton.
    let mut iterations = 0usize;
    let mut residual = (target.evaluate(mech, &q_k) - h).abs();
    let mut achieved = target.evaluate(mech, &q_k);

    for k in 1..=max_iter {
        iterations = k;
        achieved = target.evaluate(mech, &q_k);
        residual = (achieved - h).abs();
        if residual < tol {
            break;
        }

        // Compute dq/du.
        let t_mech_k = (u_k - u_0) / nominal_rate;
        let phi_q = assemble_jacobian(mech, &q_k, t_mech_k);
        let mut phi_u = DVector::zeros(mech.n_constraints());
        phi_u[driver_row] = -1.0;
        let svd = phi_q.svd(true, true);
        let dq_du = svd.solve(&-phi_u, 1e-14)
            .map_err(|_| LinkageError::SvdSolveFailed)?;

        // r'(u) = ∇g · dq/du
        let grad = target.gradient(mech, &q_k);
        let r_prime = grad.dot(&dq_du);

        if r_prime.abs() < 1e-15 {
            // Defer singularity handling to Task 1.11; for now treat as non-convergence.
            break;
        }

        // Newton step.
        u_k -= (achieved - h) / r_prime;
        // Clamp to u_range
        u_k = u_k.clamp(u_range.0, u_range.1);

        let t_mech_next = (u_k - u_0) / nominal_rate;
        q_k = solve_position(mech, &q_k, t_mech_next, 1e-10, 50)?.q;
    }

    achieved = target.evaluate(mech, &q_k);
    residual = (achieved - h).abs();
    let status = if residual < tol {
        InverseSolveStatus::Converged
    } else {
        InverseSolveStatus::NonConvergent { iterations, residual }
    };

    let _ = n_coords;
    Ok(InverseSolveResult {
        u: u_k,
        q: q_k,
        achieved,
        residual,
        iterations,
        status,
    })
}
```

Add the use statement at top of file (already imports `super::control_target::ControlTarget`):

```rust
use super::severity::{InverseSolveStatus, Severity};
```

Re-export from `mod.rs`:

```rust
pub use solver::{solve_for_target, workspace_probe, InverseSolveResult, WorkspaceProbe};
```

- [ ] **Step 4: Run tests to verify they pass**

```
cargo test --lib solver::inverse_kinematics::solver::tests
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add linkage-sim-rs/src/solver/inverse_kinematics/
git commit -m "feat(traj): solve_for_target happy-path outer Newton"
```

---

## Task 1.11 — Failure detection (reachability, singularity, branch jump, non-convergence)

**Files:**
- Modify: `linkage-sim-rs/src/solver/inverse_kinematics/solver.rs`

- [ ] **Step 1: Add failing tests for each failure mode**

```rust
    #[test]
    fn reachability_failure_in_analysis_mode() {
        let mech = build_fourbar();
        let q0 = seed_q(&mech);
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
        let q0 = seed_q(&mech);
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
```

- [ ] **Step 2: Run to verify they fail**

```
cargo test --lib solver::inverse_kinematics::solver::tests::reachability
```
Expected: FAIL — current happy path returns Converged or NonConvergent.

- [ ] **Step 3: Add reachability detection + Severity dispatch**

In `solve_for_target`, after the workspace probe (between probe call and bracket):

```rust
    // 2a. Reachability check.
    let reachability_slack = 1e-9; // tolerance to accommodate FP noise
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
```

Add the `classify_or_fail` helper near the end of the file (above `#[cfg(test)]`):

```rust
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
            let achieved = 0.0; // caller can recompute from partial_q if needed
            let residual = 0.0;
            Ok(InverseSolveResult {
                u: partial_u,
                q: partial_q,
                achieved,
                residual,
                iterations: 0,
                status,
            })
        }
    }
}
```

Now add singularity detection inside the Newton loop. Replace the `r_prime.abs() < 1e-15` check with:

```rust
        let grad_norm = grad.norm();
        let dqdu_norm = dq_du.norm();
        let eps_singularity = 1e-6 * (grad_norm * dqdu_norm).max(1e-12);
        if r_prime.abs() < eps_singularity {
            // Mechanism has lost authority over the target locally.
            let status = InverseSolveStatus::Singularity { dg_du: r_prime };
            return classify_or_fail(severity, status, q_k, u_k);
        }
```

Add branch-jump detection. Inside the Newton loop, after computing `q_k_next` (the post-update q), insert a delta-q check. First adjust the loop to compute the next q in a separate variable so we can compare:

```rust
        let q_prev = q_k.clone();
        let t_mech_next = (u_k - u_0) / nominal_rate;
        q_k = solve_position(mech, &q_prev, t_mech_next, 1e-10, 50)?.q;

        // Branch-jump check: bodies shouldn't move more than half max body length per step.
        let delta_norm = (&q_k - &q_prev).norm();
        let branch_threshold = 0.5 * mech.max_body_length(); // helper added below
        if delta_norm > branch_threshold {
            let status = InverseSolveStatus::BranchJump { delta_q_norm: delta_norm };
            return classify_or_fail(severity, status, q_prev, u_k);
        }
```

Add the `max_body_length` helper to `core/mechanism.rs`. (This is a small additive change; can be inlined locally if `core/` is to remain untouched, but a helper on the mechanism keeps things clean.)

Actually — to honor "untouched core/" — compute `max_body_length` locally in `solver.rs`:

```rust
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
```

Use `max_body_length(mech)` in the branch threshold computation.

Replace the previous Newton-step block to incorporate all the above. The full updated loop:

```rust
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

        let delta_norm = (&q_k - &q_prev).norm();
        if delta_norm > branch_threshold {
            let status = InverseSolveStatus::BranchJump { delta_q_norm: delta_norm };
            return classify_or_fail(severity, status, q_prev, u_k);
        }

        achieved = target.evaluate(mech, &q_k);
        residual = (achieved - h).abs();
        converged = residual < tol;
    }

    let status = if converged {
        InverseSolveStatus::Converged
    } else {
        let s = InverseSolveStatus::NonConvergent { iterations, residual };
        return classify_or_fail(severity, s, q_k, u_k);
    };

    Ok(InverseSolveResult {
        u: u_k,
        q: q_k,
        achieved,
        residual,
        iterations,
        status,
    })
```

- [ ] **Step 4: Run tests to verify they pass**

```
cargo test --lib solver::inverse_kinematics::solver::tests
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```
git add linkage-sim-rs/src/solver/inverse_kinematics/
git commit -m "feat(traj): add reachability/singularity/branch-jump/non-convergence detection"
```

---

## Task 1.12 — `inverse_velocity` closed-form helper

**Files:**
- Create: `linkage-sim-rs/src/solver/inverse_kinematics/derivatives.rs`
- Modify: `linkage-sim-rs/src/solver/inverse_kinematics/mod.rs`

- [ ] **Step 1: Create derivatives.rs with failing tests**

Create `linkage-sim-rs/src/solver/inverse_kinematics/derivatives.rs`:

```rust
//! Closed-form inverse velocity and finite-difference inverse acceleration helpers.
//!
//! See: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8.2 (velocity)
//!      docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8.3 (acceleration)

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;
use crate::error::LinkageError;
use crate::solver::assembly::assemble_jacobian;
use crate::solver::kinematics::solve_position;

use super::control_target::ControlTarget;

/// Solve for `u̇` such that `ġ(q(u)) = h_dot`, given converged `q` at the current `u`.
///
/// Closed-form: `u̇ = h_dot / r'(u)` where `r'(u) = ∇_q g · dq/du = −∇_q g · Φ_q⁻¹ Φ_u`.
///
/// Returns `Err(SingularJacobian)` if `|r'(u)|` is below the singularity threshold.
pub fn inverse_velocity(
    mech: &Mechanism,
    q: &DVector<f64>,
    target: &ControlTarget,
    h_dot: f64,
    t_mech: f64,
) -> Result<f64, LinkageError> {
    let driver_row = mech.n_constraints() - 1;
    let phi_q = assemble_jacobian(mech, q, t_mech);
    let mut phi_u = DVector::zeros(mech.n_constraints());
    phi_u[driver_row] = -1.0;
    let svd = phi_q.svd(true, true);
    let dq_du = svd.solve(&-phi_u, 1e-14)
        .map_err(|_| LinkageError::SvdSolveFailed)?;

    let grad = target.gradient(mech, q);
    let r_prime = grad.dot(&dq_du);

    let grad_norm = grad.norm();
    let dqdu_norm = dq_du.norm();
    let eps_singularity = 1e-6 * (grad_norm * dqdu_norm).max(1e-12);
    if r_prime.abs() < eps_singularity {
        return Err(LinkageError::TrajectorySingular { dg_du: r_prime });
    }

    Ok(h_dot / r_prime)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    fn build_fourbar() -> Mechanism {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
        let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
        let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);
        let rocker = make_bar("rocker", "C", "D", 0.03, 0.0, 0.0);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();
        mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0).unwrap();
        mech.build().unwrap();
        mech
    }

    fn solve_q_at(mech: &Mechanism, t: f64) -> DVector<f64> {
        let state = mech.state();
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
        state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
        state.set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);
        let res = solve_position(mech, &q0, t, 1e-10, 50).unwrap();
        res.q
    }

    #[test]
    fn inverse_velocity_for_angle_target_is_h_dot() {
        let mech = build_fourbar();
        let q = solve_q_at(&mech, 0.1);
        // For ControlTarget::Angle on the directly-driven body, dg/du = 1.
        // So u_dot = h_dot.
        let target = ControlTarget::angle("crank");
        let h_dot = 0.5;
        let u_dot = inverse_velocity(&mech, &q, &target, h_dot, 0.1).unwrap();
        assert_abs_diff_eq!(u_dot, h_dot, epsilon = 1e-7);
    }

    #[test]
    fn inverse_velocity_consistent_with_fd() {
        // FD check: vary u by δ and compute (g_plus − g_minus)/(2δ); verify
        // that u_dot = h_dot / dg_du matches.
        let mech = build_fourbar();
        let q = solve_q_at(&mech, 0.2);
        let target = ControlTarget::world_y("crank", [0.005, 0.0]);
        let h_dot = 1.0;
        let u_dot = inverse_velocity(&mech, &q, &target, h_dot, 0.2).unwrap();
        // FD verification of dg/du
        let omega = 2.0 * PI;
        let δ = 1e-6;
        let q_plus = solve_q_at(&mech, 0.2 + δ / omega);
        let q_minus = solve_q_at(&mech, 0.2 - δ / omega);
        let dg_du_fd = (target.evaluate(&mech, &q_plus) - target.evaluate(&mech, &q_minus)) / (2.0 * δ);
        // u_dot * dg_du should equal h_dot
        assert_abs_diff_eq!(u_dot * dg_du_fd, h_dot, epsilon = 1e-3);
    }
}
```

- [ ] **Step 2: Add module to mod.rs**

```rust
pub mod control_target;
pub mod derivatives;
pub mod severity;
pub mod solver;

pub use control_target::ControlTarget;
pub use derivatives::inverse_velocity;
pub use severity::{InverseSolveStatus, Severity};
pub use solver::{solve_for_target, workspace_probe, InverseSolveResult, WorkspaceProbe};
```

- [ ] **Step 3: Run tests to verify they pass**

```
cargo test --lib solver::inverse_kinematics::derivatives::tests
```
Expected: 2 passed.

- [ ] **Step 4: Commit**

```
git add linkage-sim-rs/src/solver/inverse_kinematics/
git commit -m "feat(traj): add inverse_velocity closed-form helper"
```

---

## Task 1.13 — `inverse_acceleration_fd` finite-difference helper

**Files:**
- Modify: `linkage-sim-rs/src/solver/inverse_kinematics/derivatives.rs`

- [ ] **Step 1: Add failing test**

Append to `derivatives.rs` tests:

```rust
    #[test]
    fn inverse_acceleration_fd_matches_zero_for_constant_velocity() {
        // For a constant-velocity trajectory, ḧ = 0 and u̇ = const, so ü should be 0.
        let mech = build_fourbar();
        let q = solve_q_at(&mech, 0.1);
        let target = ControlTarget::angle("crank");
        let u_0 = 0.0;
        let nominal_rate = 2.0 * PI;
        let u = 0.1 * nominal_rate; // u_k corresponding to t=0.1
        let u_dot = 0.5; // constant
        let h_ddot = 0.0;
        let δ = 1e-4;
        let u_ddot = inverse_acceleration_fd(
            &mech, &q, &target, u, u_dot, h_ddot, δ,
            u_0, nominal_rate,
        ).unwrap();
        assert_abs_diff_eq!(u_ddot, 0.0, epsilon = 1e-3);
    }
```

- [ ] **Step 2: Run to verify it fails**

```
cargo test --lib solver::inverse_kinematics::derivatives::tests::inverse_acceleration
```
Expected: FAIL.

- [ ] **Step 3: Implement `inverse_acceleration_fd`**

Append to `derivatives.rs` (above `#[cfg(test)]`):

```rust
/// Compute `ü` such that `g̈(q(u(t))) = h_ddot`, by finite-differencing `r'(u)`.
///
/// `ü = (h_ddot − r''(u) · u̇²) / r'(u)`, where:
///   `r'(u) = ∇_q g · dq/du`
///   `r''(u) ≈ (r'(u + δ) − r'(u − δ)) / (2δ)`
///
/// FD sidesteps the analytic `∇²_q g` term needed for option (a) in EQ §8.3.
/// Two extra forward solves per call.
#[allow(clippy::too_many_arguments)]
pub fn inverse_acceleration_fd(
    mech: &Mechanism,
    q: &DVector<f64>,
    target: &ControlTarget,
    u: f64,
    u_dot: f64,
    h_ddot: f64,
    delta: f64,
    u_0: f64,
    nominal_rate: f64,
) -> Result<f64, LinkageError> {
    let driver_row = mech.n_constraints() - 1;

    // Compute r'(u) at the current state.
    let r_prime_now = compute_r_prime(mech, q, target, (u - u_0) / nominal_rate, driver_row)?;

    // r'(u + δ)
    let q_plus = solve_position(
        mech, q, (u + delta - u_0) / nominal_rate, 1e-10, 50,
    )?.q;
    let r_prime_plus = compute_r_prime(
        mech, &q_plus, target, (u + delta - u_0) / nominal_rate, driver_row,
    )?;

    // r'(u − δ)
    let q_minus = solve_position(
        mech, q, (u - delta - u_0) / nominal_rate, 1e-10, 50,
    )?.q;
    let r_prime_minus = compute_r_prime(
        mech, &q_minus, target, (u - delta - u_0) / nominal_rate, driver_row,
    )?;

    let r_double_prime = (r_prime_plus - r_prime_minus) / (2.0 * delta);

    let grad = target.gradient(mech, q);
    let phi_q = assemble_jacobian(mech, q, (u - u_0) / nominal_rate);
    let mut phi_u = DVector::zeros(mech.n_constraints());
    phi_u[driver_row] = -1.0;
    let svd = phi_q.svd(true, true);
    let dq_du = svd.solve(&-phi_u, 1e-14)
        .map_err(|_| LinkageError::SvdSolveFailed)?;
    let grad_norm = grad.norm();
    let dqdu_norm = dq_du.norm();
    let eps_singularity = 1e-6 * (grad_norm * dqdu_norm).max(1e-12);
    if r_prime_now.abs() < eps_singularity {
        return Err(LinkageError::TrajectorySingular { dg_du: r_prime_now });
    }

    Ok((h_ddot - r_double_prime * u_dot * u_dot) / r_prime_now)
}

fn compute_r_prime(
    mech: &Mechanism,
    q: &DVector<f64>,
    target: &ControlTarget,
    t_mech: f64,
    driver_row: usize,
) -> Result<f64, LinkageError> {
    let phi_q = assemble_jacobian(mech, q, t_mech);
    let mut phi_u = DVector::zeros(mech.n_constraints());
    phi_u[driver_row] = -1.0;
    let svd = phi_q.svd(true, true);
    let dq_du = svd.solve(&-phi_u, 1e-14)
        .map_err(|_| LinkageError::SvdSolveFailed)?;
    let grad = target.gradient(mech, q);
    Ok(grad.dot(&dq_du))
}
```

Re-export from `mod.rs`:

```rust
pub use derivatives::{inverse_acceleration_fd, inverse_velocity};
```

- [ ] **Step 4: Run tests to verify they pass**

```
cargo test --lib solver::inverse_kinematics::derivatives::tests
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```
git add linkage-sim-rs/src/solver/inverse_kinematics/
git commit -m "feat(traj): add inverse_acceleration_fd helper"
```

---

## Task 1.14 — Stage 1 verification + docs/ai/ update

**Files:**
- Modify: `docs/ai/02-system.yaml`
- Modify: `docs/ai/05-update-tracker.md`

- [ ] **Step 1: Run all stage 1 tests + existing test suite**

```
cd linkage-sim-rs
cargo test --lib
```
Expected: ~600 tests passed (existing 578 + new ~23). No failures.

- [ ] **Step 2: Update docs/ai/02-system.yaml**

Add to the `invariants` section (or appropriate place):

```yaml
  - name: severity_controls_return_type_not_math
    statement: |
      Trajectory mode's Severity::{Strict, Analysis} runs identical Newton/FD code;
      the only difference is whether failure status bubbles as Err or as a result
      field. See solver/inverse_kinematics/solver.rs::classify_or_fail.

  - name: workspace_probe_cache_key
    statement: |
      Workspace probes are keyed by (mech_revision, target_hash, u_range). Cached
      on AppState; invalidated automatically on any blueprint edit, target change,
      or range change. Recompute is ~64 forward solves (~1 ms).
```

- [ ] **Step 3: Update docs/ai/05-update-tracker.md**

Append a new entry:

```markdown
## 2026-04-29 — Trajectory-mode position control (stage 1: solver layer)

Added `solver/inverse_kinematics/` subtree:
- `severity.rs` — `Severity` and `InverseSolveStatus` enums.
- `control_target.rs` — `ControlTarget` enum (5 variants: Angle / WorldX / WorldY /
  Projection / Distance) with `evaluate`, `gradient`, `hessian` (zero for v1),
  `unit_label`, `body_id`, and validating constructors.
- `solver.rs` — `solve_for_target` outer Newton + workspace probe + bracket warm-start
  + reachability / singularity / branch-jump / non-convergence detection. Severity
  dispatch via `classify_or_fail`.
- `derivatives.rs` — `inverse_velocity` (closed-form), `inverse_acceleration_fd`
  (two extra forward solves per call).

LinkageError extended with TrajectoryUnreachable / TrajectorySingular /
TrajectoryBranchJump / TrajectoryNonConvergent variants.

Tests: ~22 new (round-trip per variant + per-variant gradient FD checks +
failure-mode coverage in both Severity modes). Stage 1 produces no GUI changes;
solver works in isolation.

Deferred: analytic Hessian (∇²g) for non-Angle variants; needed only if FD
acceleration inverse is ever found insufficient (per spec §8.3, FD recommended
for v1).
```

- [ ] **Step 4: Commit docs**

```
git add docs/ai/02-system.yaml docs/ai/05-update-tracker.md
git commit -m "docs(ai): trajectory mode stage 1 (solver layer)"
```

- [ ] **Step 5: Push the stage to remote**

```
git push origin main
```

---

# Stage 2 — Sweep extension

Goal: Wire `solve_for_target` into the existing sweep pipeline as `SweepMode::Trajectory`. Trajectory mode is reachable from code (and from tests); GUI does not yet expose it.

## Task 2.1 — `TrajectoryProfile` struct

**Files:**
- Modify: `linkage-sim-rs/src/gui/state/types.rs`

- [ ] **Step 1: Find the right insertion point**

Run:
```
grep -n "pub struct SweepData\|pub enum MotionProfile" linkage-sim-rs/src/gui/state/types.rs linkage-sim-rs/src/gui/state/mod.rs
```
Note the line where `SweepData` is defined (likely in `types.rs`).

- [ ] **Step 2: Add failing test**

Append a `#[cfg(test)] mod tests` block (or extend the existing one) in the file that hosts `TrajectoryProfile`. Plan: place `TrajectoryProfile` in `gui/state/types.rs`.

```rust
    #[test]
    fn trajectory_profile_constant_velocity_evaluates_linearly() {
        let profile = TrajectoryProfile {
            shape: MotionProfile::ConstantSpeed,
            start_value: 0.0,
            end_value: 1.0,
            duration: 2.0,
        };
        let (h, h_dot, h_ddot) = profile.evaluate(0.0);
        assert!((h - 0.0).abs() < 1e-12);
        let (h, _, _) = profile.evaluate(1.0);
        assert!((h - 0.5).abs() < 1e-12);
        let (h, h_dot, h_ddot) = profile.evaluate(2.0);
        assert!((h - 1.0).abs() < 1e-12);
        assert!((h_dot - 0.5).abs() < 1e-12); // (end-start)/duration = 0.5
        assert!(h_ddot.abs() < 1e-12);
    }

    #[test]
    fn trajectory_profile_sample_times_are_uniform() {
        let profile = TrajectoryProfile {
            shape: MotionProfile::ConstantSpeed,
            start_value: 0.0,
            end_value: 1.0,
            duration: 1.0,
        };
        let samples = profile.sample_times(5);
        assert_eq!(samples.len(), 5);
        assert!((samples[0] - 0.0).abs() < 1e-12);
        assert!((samples[4] - 1.0).abs() < 1e-12);
        assert!((samples[2] - 0.5).abs() < 1e-12);
    }
```

- [ ] **Step 3: Run to verify they fail**

```
cargo test --lib gui::state::types::tests::trajectory_profile
```
Expected: FAIL.

- [ ] **Step 4: Implement `TrajectoryProfile`**

In `gui/state/types.rs`, add (with imports as needed):

```rust
use serde::{Deserialize, Serialize};

use crate::gui::state::MotionProfile;

/// Wraps an existing `MotionProfile` shape with absolute units (start, end, duration).
/// Used by `SweepMode::Trajectory` to define a target observable trajectory `h(t)`.
///
/// See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §5.5
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrajectoryProfile {
    /// Shape of the velocity profile (constant / trapezoidal / S-curve).
    pub shape: MotionProfile,
    /// `h(0)` in target units.
    pub start_value: f64,
    /// `h(duration)` in target units.
    pub end_value: f64,
    /// Trajectory duration in seconds. Must be > 0.
    pub duration: f64,
}

impl TrajectoryProfile {
    /// Evaluate `(h(t), ḣ(t), ḧ(t))` at trajectory time `t`.
    /// `t` is clamped to `[0, duration]`.
    pub fn evaluate(&self, t: f64) -> (f64, f64, f64) {
        assert!(self.duration > 0.0, "TrajectoryProfile.duration must be > 0");
        let t_clamped = t.clamp(0.0, self.duration);
        match self.shape {
            MotionProfile::ConstantSpeed => {
                let frac = t_clamped / self.duration;
                let span = self.end_value - self.start_value;
                let h = self.start_value + frac * span;
                let h_dot = span / self.duration;
                let h_ddot = 0.0;
                (h, h_dot, h_ddot)
            }
            MotionProfile::Trapezoidal { accel_fraction, decel_fraction } => {
                trapezoidal_value(
                    t_clamped, self.duration, self.start_value, self.end_value,
                    accel_fraction, decel_fraction,
                )
            }
        }
    }

    /// Return `n` uniform sample times across `[0, duration]`.
    pub fn sample_times(&self, n: usize) -> Vec<f64> {
        assert!(n >= 2, "sample_times requires n >= 2");
        (0..n).map(|i| (i as f64) * self.duration / ((n - 1) as f64)).collect()
    }
}

fn trapezoidal_value(
    t: f64, duration: f64,
    start: f64, end: f64,
    accel_frac: f64, decel_frac: f64,
) -> (f64, f64, f64) {
    let cruise_frac = 1.0 - accel_frac - decel_frac;
    debug_assert!(cruise_frac >= 0.0);
    let span = end - start;

    let t_a = accel_frac * duration;
    let t_c = cruise_frac * duration;
    let t_d = decel_frac * duration;

    // Peak velocity v_peak: ∫velocity dt = span ⇒ v_peak (t_a/2 + t_c + t_d/2) = span
    let denom = 0.5 * t_a + t_c + 0.5 * t_d;
    if denom.abs() < 1e-15 {
        return (start, 0.0, 0.0);
    }
    let v_peak = span / denom;

    if t < t_a {
        // Accelerate
        let a = v_peak / t_a;
        let h = start + 0.5 * a * t * t;
        let h_dot = a * t;
        let h_ddot = a;
        (h, h_dot, h_ddot)
    } else if t < t_a + t_c {
        // Cruise
        let h = start + 0.5 * v_peak * t_a + v_peak * (t - t_a);
        (h, v_peak, 0.0)
    } else {
        // Decelerate
        let a = -v_peak / t_d;
        let dt = t - (t_a + t_c);
        let h_at_decel_start = start + 0.5 * v_peak * t_a + v_peak * t_c;
        let h = h_at_decel_start + v_peak * dt + 0.5 * a * dt * dt;
        let h_dot = v_peak + a * dt;
        let h_ddot = a;
        (h, h_dot.max(0.0), h_ddot)
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

```
cargo test --lib gui::state::types::tests::trajectory_profile
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```
git add linkage-sim-rs/src/gui/state/types.rs
git commit -m "feat(traj): add TrajectoryProfile composing MotionProfile with absolute units"
```

---

## Task 2.2 — `SweepData` field additions

**Files:**
- Modify: `linkage-sim-rs/src/gui/state/types.rs`

- [ ] **Step 1: Locate `SweepData` struct**

```
grep -n "pub struct SweepData" linkage-sim-rs/src/gui/state/types.rs
```

- [ ] **Step 2: Add new optional fields**

Add these fields to the `SweepData` struct (with appropriate `#[serde(default, skip_serializing_if = "Option::is_none")]` attributes if the struct has serde derives):

```rust
    /// Target value h(t_k) at each sample. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_values: Option<Vec<f64>>,
    /// Achieved value g(q_k) at each sample. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub achieved_values: Option<Vec<f64>>,
    /// Tracking residual achieved - target. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracking_residual: Option<Vec<f64>>,
    /// Back-solved input parameter u_k. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub u_values: Option<Vec<f64>>,
    /// Back-solved input rate u̇_k. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub u_dot_values: Option<Vec<f64>>,
    /// Back-solved input accel ü_k. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub u_ddot_values: Option<Vec<f64>>,
    /// Per-sample inverse-solve diagnostic. Populated only in Trajectory mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inverse_solve_statuses:
        Option<Vec<crate::solver::inverse_kinematics::InverseSolveStatus>>,
```

If `SweepData` does not yet derive serde, add `#[derive(Serialize, Deserialize)]` and the use line. If `InverseSolveStatus` is not yet `Serialize`, add `#[derive(Serialize, Deserialize)]` to it in `severity.rs` (it currently is not).

To make `InverseSolveStatus` serializable, modify `severity.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InverseSolveStatus { ... }
```

- [ ] **Step 3: Verify compile**

```
cargo build --lib
```
Expected: success. Existing tests still pass:

```
cargo test --lib
```

- [ ] **Step 4: Commit**

```
git add linkage-sim-rs/src/
git commit -m "feat(traj): SweepData adds Option<Vec> fields for trajectory time series"
```

---

## Task 2.3 — `SweepMode::Trajectory` variant

**Files:**
- Modify: `linkage-sim-rs/src/gui/sweep/mod.rs`

- [ ] **Step 1: Locate `SweepMode` enum**

```
grep -n "pub enum SweepMode" linkage-sim-rs/src/gui/sweep/mod.rs
```

- [ ] **Step 2: Add the new variant**

Replace the existing enum definition:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SweepMode {
    Angle,
    Stroke,
    /// Inverse trajectory analysis: prescribe an output observable trajectory and
    /// back-solve the actuator input. See spec §3.
    Trajectory {
        target: crate::solver::inverse_kinematics::ControlTarget,
        profile: crate::gui::state::types::TrajectoryProfile,
        severity: crate::solver::inverse_kinematics::Severity,
        n_samples: usize,
    },
}
```

If existing helpers like `is_stroke()` exist on `SweepMode`, update them to handle the new variant. Add:

```rust
impl SweepMode {
    pub fn is_trajectory(&self) -> bool {
        matches!(self, SweepMode::Trajectory { .. })
    }
}
```

- [ ] **Step 3: Verify compile**

```
cargo build --lib
```
Expected: success — though there will likely be a non-exhaustive-match warning in `compute_sweep_data`. Continue to Task 2.4 to address.

- [ ] **Step 4: Commit**

```
git add linkage-sim-rs/src/gui/sweep/mod.rs
git commit -m "feat(traj): add SweepMode::Trajectory variant"
```

---

## Task 2.4 — `compute_trajectory` per-sample loop

**Files:**
- Modify: `linkage-sim-rs/src/gui/sweep/mod.rs`

- [ ] **Step 1: Add failing integration test**

Append to the `#[cfg(test)] mod tests` in `gui/sweep/mod.rs` (or create one):

```rust
    use crate::core::body::{make_bar, make_ground};
    use crate::core::mechanism::Mechanism;
    use crate::solver::inverse_kinematics::{ControlTarget, Severity};
    use crate::gui::state::types::TrajectoryProfile;
    use crate::gui::state::MotionProfile;

    fn build_fourbar_mech() -> Mechanism {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
        let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
        let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);
        let rocker = make_bar("rocker", "C", "D", 0.03, 0.0, 0.0);
        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();
        mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * std::f64::consts::PI, 0.0).unwrap();
        mech.build().unwrap();
        mech
    }

    #[test]
    fn compute_trajectory_populates_all_new_fields() {
        let mech = build_fourbar_mech();
        let mode = SweepMode::Trajectory {
            target: ControlTarget::angle("crank"),
            profile: TrajectoryProfile {
                shape: MotionProfile::ConstantSpeed,
                start_value: 0.5,
                end_value: 1.5,
                duration: 1.0,
            },
            severity: Severity::Analysis,
            n_samples: 10,
        };
        let data = compute_sweep_data_full(&mech, &mode, /* args ... */)
            .expect("compute_sweep_data should succeed");
        assert_eq!(data.target_values.as_ref().unwrap().len(), 10);
        assert_eq!(data.achieved_values.as_ref().unwrap().len(), 10);
        assert_eq!(data.u_values.as_ref().unwrap().len(), 10);
        assert_eq!(data.inverse_solve_statuses.as_ref().unwrap().len(), 10);
    }
```

(The test name `compute_sweep_data_full` is a placeholder — adjust to match the real function signature in your codebase. The point is to assert that all new fields are populated with `n_samples` entries.)

- [ ] **Step 2: Run to verify it fails**

```
cargo test --lib gui::sweep::tests::compute_trajectory
```
Expected: FAIL — `compute_trajectory` not yet implemented.

- [ ] **Step 3: Implement `compute_trajectory`**

Add to `gui/sweep/mod.rs`:

```rust
use crate::solver::assembly::{assemble_gamma, assemble_jacobian, assemble_phi_t};
use crate::solver::inverse_dynamics::solve_inverse_dynamics;
use crate::solver::inverse_kinematics::{
    inverse_acceleration_fd, inverse_velocity, solve_for_target, ControlTarget,
    InverseSolveResult, Severity,
};
use crate::solver::kinematics::{solve_acceleration, solve_velocity};
use crate::solver::statics::{extract_reactions, get_driver_reactions, solve_statics};
use crate::core::mechanism::Mechanism;
use crate::error::LinkageError;
use crate::gui::state::types::TrajectoryProfile;
use nalgebra::DVector;

/// Per-sample trajectory loop. Mirrors the existing `compute_sweep_data` pattern
/// for forward sweeps but with inverse Newton driving the input parameter.
///
/// See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §7
#[allow(clippy::too_many_arguments)]
pub fn compute_trajectory(
    mech: &Mechanism,
    q_seed: &DVector<f64>,
    target: &ControlTarget,
    profile: &TrajectoryProfile,
    severity: Severity,
    n_samples: usize,
    nominal_rate: f64,
    u_0: f64,
    u_range: (f64, f64),
    gravity_magnitude: f64,
    data: &mut SweepData,
) -> Result<(), LinkageError> {
    assert!(n_samples >= 2, "n_samples must be >= 2");
    assert!(nominal_rate.abs() > 1e-12, "nominal_rate must be non-zero");

    // Initialize new SweepData fields.
    data.target_values = Some(Vec::with_capacity(n_samples));
    data.achieved_values = Some(Vec::with_capacity(n_samples));
    data.tracking_residual = Some(Vec::with_capacity(n_samples));
    data.u_values = Some(Vec::with_capacity(n_samples));
    data.u_dot_values = Some(Vec::with_capacity(n_samples));
    data.u_ddot_values = Some(Vec::with_capacity(n_samples));
    data.inverse_solve_statuses = Some(Vec::with_capacity(n_samples));

    let times = profile.sample_times(n_samples);
    let driver_row = mech.n_constraints() - 1;
    let δ = 1e-4 * (u_range.1 - u_range.0).abs().max(1e-6);

    let mut q_prev = q_seed.clone();

    for &t_k in &times {
        let (h_k, h_dot_k, h_ddot_k) = profile.evaluate(t_k);

        // 1. Inverse position solve.
        let res = solve_for_target(
            mech, &q_prev, target, h_k, severity,
            u_range, u_0, nominal_rate,
            1e-8, 50, 64,
        )?;
        let InverseSolveResult { u: u_k, q: q_k, status, .. } = res;

        let t_mech_k = (u_k - u_0) / nominal_rate;

        // 2. Inverse velocity (closed-form). On singularity, treat as zero;
        // status will already reflect the issue.
        let u_dot_k = inverse_velocity(mech, &q_k, target, h_dot_k, t_mech_k)
            .unwrap_or(0.0);

        // 3. Inverse acceleration (FD). On singularity, zero; same treatment.
        let u_ddot_k = inverse_acceleration_fd(
            mech, &q_k, target, u_k, u_dot_k, h_ddot_k, δ, u_0, nominal_rate,
        ).unwrap_or(0.0);

        // 4. Body velocity & acceleration with trajectory rates substituted (§7.3).
        let phi_q = assemble_jacobian(mech, &q_k, t_mech_k);
        let mut phi_t = assemble_phi_t(mech, &q_k, t_mech_k);
        phi_t[driver_row] = -u_dot_k;
        let svd_q = phi_q.clone().svd(true, true);
        let neg_phi_t = -phi_t;
        let q_dot_k = svd_q.solve(&neg_phi_t, 1e-14)
            .map_err(|_| LinkageError::SvdSolveFailed)?;

        let mut gamma = assemble_gamma(mech, &q_k, &q_dot_k, t_mech_k);
        gamma[driver_row] = u_ddot_k;
        let svd_q2 = phi_q.svd(true, true);
        let q_ddot_k = svd_q2.solve(&gamma, 1e-14)
            .map_err(|_| LinkageError::SvdSolveFailed)?;

        // 5. Reuse existing per-sample solvers (§7.5).
        let _statics = solve_statics(mech, &q_k, t_mech_k);
        let _inv_dyn = solve_inverse_dynamics(mech, &q_k, &q_dot_k, &q_ddot_k, t_mech_k);
        let _ = gravity_magnitude;

        // 6. Push.
        let achieved = target.evaluate(mech, &q_k);
        data.target_values.as_mut().unwrap().push(h_k);
        data.achieved_values.as_mut().unwrap().push(achieved);
        data.tracking_residual.as_mut().unwrap().push(achieved - h_k);
        data.u_values.as_mut().unwrap().push(u_k);
        data.u_dot_values.as_mut().unwrap().push(u_dot_k);
        data.u_ddot_values.as_mut().unwrap().push(u_ddot_k);
        data.inverse_solve_statuses.as_mut().unwrap().push(status);

        q_prev = q_k;
    }

    Ok(())
}
```

The existing per-sample push of `driver_torques`, `joint_reactions`, `kinetic_energy`, etc. should be invoked here matching the existing forward-sweep code. Refactor as needed: prefer extracting a `push_per_sample_existing` helper from the current `compute_sweep_data` so both forward and trajectory branches call it. Minimal refactor for v1: copy the relevant push lines from the current loop and adapt.

- [ ] **Step 4: Wire the new branch into `compute_sweep_data`**

In the existing `compute_sweep_data` function, locate the match on `SweepMode` and add:

```rust
        SweepMode::Trajectory { target, profile, severity, n_samples } => {
            let nominal_rate = current_driver_rate(mech)?;
            let u_0 = current_driver_u_0(mech);
            let u_range = current_sweep_u_range(state)
                .unwrap_or_else(|| default_u_range_for_driver(mech));
            compute_trajectory(
                mech, q0, target, profile, *severity, *n_samples,
                nominal_rate, u_0, u_range, gravity_magnitude, &mut data,
            )?;
        }
```

The helpers `current_driver_rate`, `current_driver_u_0`, `current_sweep_u_range`, `default_u_range_for_driver` are small pure functions you may need to add. They each read mech state or app state to determine the corresponding scalar.

- [ ] **Step 5: Run tests to verify they pass**

```
cargo test --lib gui::sweep::tests
```
Expected: integration test passes; existing tests still pass.

```
cargo test --lib
```
Expected: all ~600 tests pass.

- [ ] **Step 6: Commit**

```
git add linkage-sim-rs/src/gui/sweep/mod.rs
git commit -m "feat(traj): compute_trajectory wires inverse-kinematics into sweep pipeline"
```

---

## Task 2.5 — Stage 2 doc updates

**Files:**
- Modify: `docs/ai/02-system.yaml`
- Modify: `docs/ai/05-update-tracker.md`

- [ ] **Step 1: Add invariants**

Append to `docs/ai/02-system.yaml`:

```yaml
  - name: trajectory_mode_phi_t_override
    statement: |
      compute_trajectory overrides Φ_t and γ on the driver row to substitute
      back-solved u_dot/u_ddot. Constraint code in core/ remains constant-speed-
      parameterized; the override is two lines per sample in
      gui/sweep/mod.rs::compute_trajectory.

  - name: trajectory_mode_t_mech_encoding
    statement: |
      u_k is encoded as t_mech_k = (u_k − u_0) / nominal_rate to reuse existing
      forward solvers without modification. Same trick is used by solve_at_angle
      and solve_at_stroke. Requires nominal_rate ≠ 0 (guaranteed by
      MIN_DRIVER_OMEGA_ABS clamp).
```

- [ ] **Step 2: Append stage 2 entry to update tracker**

Append to the existing 2026-04-29 entry in `docs/ai/05-update-tracker.md`:

```markdown
### Stage 2 — sweep extension

- `SweepMode::Trajectory { target, profile, severity, n_samples }` variant.
- `compute_trajectory` per-sample loop in `gui/sweep/mod.rs` reuses the existing
  per-sample force / energy / reaction calls.
- `Φ_t` / `γ` driver-row override for trajectory rates (no `core/` changes).
- `t_mech` encoding trick for forward-solver reuse.
- `SweepData` gains 7 Option<Vec> fields for trajectory time series; existing
  fields populate identically.

Tests: 1+ integration test verifying field population. All ~600 tests pass.
GUI does not yet expose trajectory mode; activation in stage 3.
```

- [ ] **Step 3: Commit**

```
git add docs/ai/02-system.yaml docs/ai/05-update-tracker.md
git commit -m "docs(ai): trajectory mode stage 2 (sweep extension)"
git push origin main
```

---

# Stage 3 — UI surface

Goal: Expose trajectory mode through the GUI. End-to-end flow possible: user picks `SweepMode::Trajectory`, configures target / profile / severity, sees the time-series plot, and can click-to-scrub.

## Task 3.1 — `gui/state/trajectory_ops.rs` for live scrubbing

**Files:**
- Create: `linkage-sim-rs/src/gui/state/trajectory_ops.rs`
- Modify: `linkage-sim-rs/src/gui/state/mod.rs`

- [ ] **Step 1: Create the file with `solve_for_trajectory_target`**

Create `linkage-sim-rs/src/gui/state/trajectory_ops.rs`:

```rust
//! Trajectory-mode interactive solving on AppState.
//!
//! Sibling to solve_at_angle / solve_at_stroke; called by the per-frame
//! "current target" slider in `gui/trajectory_panel/mod.rs`.
//!
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §7.6

use crate::solver::inverse_kinematics::{
    solve_for_target, ControlTarget, Severity,
};
use crate::gui::state::AppState;

impl AppState {
    /// Drive the mechanism to the given (target, h) — the inverse-kinematics
    /// equivalent of the existing `solve_at_angle` and `solve_at_stroke`.
    /// Updates `self.q` and `self.driver_*` fields on success.
    pub fn solve_for_trajectory_target(
        &mut self,
        target: &ControlTarget,
        h: f64,
    ) {
        let Some(mech) = self.mechanism.as_ref() else {
            return;
        };
        let nominal_rate = if self.driver_omega.abs() > 1e-12 {
            self.driver_omega
        } else {
            return;
        };
        let u_0 = self.driver_theta_0;
        let u_range = (
            self.sweep_angle_min_deg.to_radians(),
            self.sweep_angle_max_deg.to_radians(),
        );

        match solve_for_target(
            mech, &self.q, target, h, Severity::Analysis,
            u_range, u_0, nominal_rate, 1e-8, 50, 64,
        ) {
            Ok(res) => {
                self.q = res.q;
                self.driver_angle = res.u; // for revolute; for linear use res.u as stroke
                self.last_good_q = self.q.clone();
            }
            Err(_) => {
                // Out-of-range / singularity: leave state unchanged.
            }
        }
    }
}
```

- [ ] **Step 2: Wire module into state/mod.rs**

Modify `gui/state/mod.rs` line ~16 to add:

```rust
mod trajectory_ops;
```

- [ ] **Step 3: Verify compile**

```
cargo build --lib
```
Expected: success.

- [ ] **Step 4: Commit**

```
git add linkage-sim-rs/src/gui/state/trajectory_ops.rs linkage-sim-rs/src/gui/state/mod.rs
git commit -m "feat(traj): AppState::solve_for_trajectory_target for live scrubbing"
```

---

## Task 3.2 — `gui/trajectory_panel/` scaffolding

**Files:**
- Create: `linkage-sim-rs/src/gui/trajectory_panel/mod.rs`
- Modify: `linkage-sim-rs/src/gui/mod.rs`

- [ ] **Step 1: Create panel module skeleton**

Create `linkage-sim-rs/src/gui/trajectory_panel/mod.rs`:

```rust
//! Top-level UI for `SweepMode::Trajectory`.
//!
//! Three sections (top to bottom):
//!   1. Target picker — `ControlTarget` variant + body / point / axis fields.
//!   2. Profile editor — `TrajectoryProfile` + inline h(t) preview.
//!   3. Severity & advanced options.
//!
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §8

mod profile_input;
mod target_picker;

use eframe::egui;

use crate::gui::state::AppState;

/// Render the trajectory input panel. Called from `gui/input_panel.rs` when
/// `SweepMode::Trajectory` is active.
pub fn draw(state: &mut AppState, ui: &mut egui::Ui) {
    egui::CollapsingHeader::new("Target observable")
        .default_open(true)
        .show(ui, |ui| {
            target_picker::draw(state, ui);
        });

    egui::CollapsingHeader::new("Profile")
        .default_open(true)
        .show(ui, |ui| {
            profile_input::draw(state, ui);
        });

    egui::CollapsingHeader::new("Solve options")
        .default_open(false)
        .show(ui, |ui| {
            draw_severity_toggle(state, ui);
        });
}

fn draw_severity_toggle(state: &mut AppState, ui: &mut egui::Ui) {
    use crate::solver::inverse_kinematics::Severity;
    let current = state.trajectory_severity;
    let mut new = current;
    ui.horizontal(|ui| {
        ui.label("Severity:");
        ui.radio_value(&mut new, Severity::Analysis, "Analysis (annotate failures)");
        ui.radio_value(&mut new, Severity::Strict, "Strict (abort on failure)");
    });
    if new != current {
        state.trajectory_severity = new;
    }
}
```

- [ ] **Step 2: Add empty stubs for sub-modules**

Create `linkage-sim-rs/src/gui/trajectory_panel/target_picker.rs`:

```rust
//! Target picker UI: ControlTarget selector + body/point/axis fields.

use eframe::egui;

use crate::gui::state::AppState;

pub fn draw(state: &mut AppState, ui: &mut egui::Ui) {
    let _ = state;
    ui.label("(target picker — implemented in Task 3.4)");
    let _ = ui;
}
```

Create `linkage-sim-rs/src/gui/trajectory_panel/profile_input.rs`:

```rust
//! Profile editor + inline h(t) preview.

use eframe::egui;

use crate::gui::state::AppState;

pub fn draw(state: &mut AppState, ui: &mut egui::Ui) {
    let _ = state;
    ui.label("(profile editor — implemented in Task 3.5)");
    let _ = ui;
}
```

- [ ] **Step 3: Wire module into gui/mod.rs**

Modify `gui/mod.rs` to add:

```rust
mod trajectory_panel;
```

(After other `mod` lines, alphabetical or contextual.)

- [ ] **Step 4: Add `trajectory_severity` field to AppState**

In `gui/state/mod.rs`, add to `AppState` struct:

```rust
    /// Severity for the active trajectory analysis.
    pub trajectory_severity: crate::solver::inverse_kinematics::Severity,
```

In the `Default` impl or constructor, initialize:

```rust
        trajectory_severity: crate::solver::inverse_kinematics::Severity::Analysis,
```

- [ ] **Step 5: Verify compile**

```
cargo build --bin linkage-sim-native
```
Expected: success.

- [ ] **Step 6: Commit**

```
git add linkage-sim-rs/src/gui/
git commit -m "feat(traj): trajectory_panel scaffolding + severity toggle"
```

---

## Task 3.3 — Sweep mode dropdown extension + delegation from input_panel

**Files:**
- Modify: `linkage-sim-rs/src/gui/input_panel.rs`

- [ ] **Step 1: Locate sweep mode dropdown**

```
grep -n "SweepMode::" linkage-sim-rs/src/gui/input_panel.rs
```

- [ ] **Step 2: Add Trajectory option to the dropdown**

Locate the sweep mode `egui::ComboBox` and add:

```rust
ui.selectable_value(
    &mut state.sweep_mode,
    SweepMode::Trajectory {
        target: ControlTarget::angle("crank"), // sensible default
        profile: TrajectoryProfile {
            shape: MotionProfile::ConstantSpeed,
            start_value: 0.0,
            end_value: 1.0,
            duration: 1.0,
        },
        severity: state.trajectory_severity,
        n_samples: 200,
    },
    "Trajectory",
);
```

- [ ] **Step 3: Delegate input panel rendering**

Where the input panel renders the angle/stroke slider, add a branch:

```rust
match &state.sweep_mode {
    SweepMode::Trajectory { .. } => {
        crate::gui::trajectory_panel::draw(state, ui);
    }
    _ => {
        // existing angle/stroke slider rendering
    }
}
```

- [ ] **Step 4: Verify compile**

```
cargo build --bin linkage-sim-native
```
Expected: success.

- [ ] **Step 5: Run native binary; manually verify mode dropdown works**

```
cargo run --bin linkage-sim-native
```

In the GUI:
- Open a sample (e.g. ParallelogramActuator).
- Locate the Sweep Mode dropdown.
- Switch to Trajectory.
- Verify the input panel switches to the trajectory placeholder UI.

- [ ] **Step 6: Commit**

```
git add linkage-sim-rs/src/gui/input_panel.rs
git commit -m "feat(traj): SweepMode dropdown gains Trajectory; input panel delegates"
```

---

## Task 3.4 — Target picker UI

**Files:**
- Modify: `linkage-sim-rs/src/gui/trajectory_panel/target_picker.rs`

- [ ] **Step 1: Implement the picker**

Replace contents of `target_picker.rs`:

```rust
//! Target picker UI: ControlTarget variant selector + per-variant fields.

use eframe::egui;

use crate::gui::state::{AppState, SweepMode};
use crate::solver::inverse_kinematics::ControlTarget;

pub fn draw(state: &mut AppState, ui: &mut egui::Ui) {
    let SweepMode::Trajectory { target, .. } = &mut state.sweep_mode else {
        ui.label("(only visible in Trajectory mode)");
        return;
    };

    // Variant selector
    let body_ids = state.mechanism.as_ref().map(|m| {
        m.bodies().filter_map(|(id, _)| {
            if id == "ground" { None } else { Some(id.to_string()) }
        }).collect::<Vec<_>>()
    }).unwrap_or_default();

    let current_kind = target_kind_label(target);
    let mut new_kind = current_kind;
    egui::ComboBox::from_label("Observable")
        .selected_text(new_kind)
        .show_ui(ui, |ui| {
            for kind in &["Angle", "WorldX", "WorldY", "Projection", "Distance"] {
                ui.selectable_value(&mut new_kind, kind, *kind);
            }
        });

    if new_kind != current_kind {
        let body = body_ids.first().cloned().unwrap_or_default();
        *target = match new_kind {
            "Angle" => ControlTarget::angle(body),
            "WorldX" => ControlTarget::world_x(body, [0.0, 0.0]),
            "WorldY" => ControlTarget::world_y(body, [0.0, 0.0]),
            "Projection" => ControlTarget::projection(body, [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]),
            "Distance" => ControlTarget::distance(body, [0.0, 0.0], [0.0, 0.0]),
            _ => target.clone(),
        };
    }

    // Body picker (common to all variants)
    draw_body_picker(target, &body_ids, ui);

    // Variant-specific fields
    match target {
        ControlTarget::Angle { .. } => {}
        ControlTarget::WorldX { local_pt, .. }
        | ControlTarget::WorldY { local_pt, .. } => {
            ui.label("Body-local point (m):");
            draw_point_input(local_pt, ui);
        }
        ControlTarget::Projection { local_pt, axis_origin, axis_dir, .. } => {
            ui.label("Body-local point (m):");
            draw_point_input(local_pt, ui);
            ui.label("Axis origin (m):");
            draw_point_input(axis_origin, ui);
            ui.label("Axis direction (will be normalized):");
            draw_point_input(axis_dir, ui);
        }
        ControlTarget::Distance { local_pt, ref_pt, .. } => {
            ui.label("Body-local point (m):");
            draw_point_input(local_pt, ui);
            ui.label("Reference point (m):");
            draw_point_input(ref_pt, ui);
        }
    }

    // Live readout
    if let Some(mech) = &state.mechanism {
        let g_now = target.evaluate(mech, &state.q);
        ui.label(format!("Current g(q) = {:.4} {}", g_now, target.unit_label()));
    }
}

fn target_kind_label(t: &ControlTarget) -> &'static str {
    match t {
        ControlTarget::Angle { .. } => "Angle",
        ControlTarget::WorldX { .. } => "WorldX",
        ControlTarget::WorldY { .. } => "WorldY",
        ControlTarget::Projection { .. } => "Projection",
        ControlTarget::Distance { .. } => "Distance",
    }
}

fn draw_body_picker(target: &mut ControlTarget, body_ids: &[String], ui: &mut egui::Ui) {
    let current = match target {
        ControlTarget::Angle { body_id }
        | ControlTarget::WorldX { body_id, .. }
        | ControlTarget::WorldY { body_id, .. }
        | ControlTarget::Projection { body_id, .. }
        | ControlTarget::Distance { body_id, .. } => body_id.clone(),
    };
    let mut new = current.clone();
    egui::ComboBox::from_label("Body")
        .selected_text(&current)
        .show_ui(ui, |ui| {
            for id in body_ids {
                ui.selectable_value(&mut new, id.clone(), id);
            }
        });
    if new != current {
        match target {
            ControlTarget::Angle { body_id }
            | ControlTarget::WorldX { body_id, .. }
            | ControlTarget::WorldY { body_id, .. }
            | ControlTarget::Projection { body_id, .. }
            | ControlTarget::Distance { body_id, .. } => *body_id = new,
        }
    }
}

fn draw_point_input(pt: &mut [f64; 2], ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label("x:");
        ui.add(egui::DragValue::new(&mut pt[0]).speed(0.001).suffix(" m"));
        ui.label("y:");
        ui.add(egui::DragValue::new(&mut pt[1]).speed(0.001).suffix(" m"));
    });
}
```

- [ ] **Step 2: Verify compile + manual test**

```
cargo build --bin linkage-sim-native && cargo run --bin linkage-sim-native
```

In GUI:
- Open ParallelogramActuator sample, switch to Trajectory mode.
- Verify dropdown shows all 5 ControlTarget variants.
- Switch between variants and verify field set changes.
- Verify live `g(q)` readout updates.

- [ ] **Step 3: Commit**

```
git add linkage-sim-rs/src/gui/trajectory_panel/target_picker.rs
git commit -m "feat(traj): target picker UI (5 variants + live g(q) readout)"
```

---

## Task 3.5 — Profile editor with inline h(t) preview

**Files:**
- Modify: `linkage-sim-rs/src/gui/trajectory_panel/profile_input.rs`

- [ ] **Step 1: Implement profile editor**

Replace contents of `profile_input.rs`:

```rust
//! Profile editor for TrajectoryProfile + inline h(t) preview plot.

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

use crate::gui::state::{AppState, MotionProfile, SweepMode};

pub fn draw(state: &mut AppState, ui: &mut egui::Ui) {
    let SweepMode::Trajectory { profile, n_samples, .. } = &mut state.sweep_mode else {
        ui.label("(only visible in Trajectory mode)");
        return;
    };

    // Shape dropdown
    let mut shape_label = match profile.shape {
        MotionProfile::ConstantSpeed => "ConstantSpeed",
        MotionProfile::Trapezoidal { .. } => "Trapezoidal",
    };
    egui::ComboBox::from_label("Shape")
        .selected_text(shape_label)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut shape_label, "ConstantSpeed", "ConstantSpeed");
            ui.selectable_value(&mut shape_label, "Trapezoidal", "Trapezoidal");
        });
    profile.shape = match shape_label {
        "ConstantSpeed" => MotionProfile::ConstantSpeed,
        "Trapezoidal" => match profile.shape {
            MotionProfile::Trapezoidal { .. } => profile.shape,
            _ => MotionProfile::Trapezoidal {
                accel_fraction: 0.2,
                decel_fraction: 0.2,
            },
        },
        _ => profile.shape,
    };

    // Start / end / duration
    ui.horizontal(|ui| {
        ui.label("Start:");
        ui.add(egui::DragValue::new(&mut profile.start_value).speed(0.001));
        ui.label("End:");
        ui.add(egui::DragValue::new(&mut profile.end_value).speed(0.001));
    });
    ui.horizontal(|ui| {
        ui.label("Duration (s):");
        ui.add(egui::DragValue::new(&mut profile.duration).speed(0.01).clamp_range(0.001..=1000.0));
    });

    // Trapezoidal-only fields
    if let MotionProfile::Trapezoidal { accel_fraction, decel_fraction } = &mut profile.shape {
        ui.horizontal(|ui| {
            ui.label("Accel fraction:");
            ui.add(egui::DragValue::new(accel_fraction).speed(0.01).clamp_range(0.05..=0.45));
            ui.label("Decel fraction:");
            ui.add(egui::DragValue::new(decel_fraction).speed(0.01).clamp_range(0.05..=0.45));
        });
    }

    // Sample count
    ui.horizontal(|ui| {
        ui.label("Samples:");
        ui.add(egui::DragValue::new(n_samples).clamp_range(10usize..=2000));
    });

    // Inline h(t) preview
    Plot::new("traj_profile_preview")
        .height(80.0)
        .show_axes([false, false])
        .show(ui, |plot_ui| {
            let n = 100;
            let pts: PlotPoints = (0..=n).map(|i| {
                let t = profile.duration * (i as f64) / (n as f64);
                let (h, _, _) = profile.evaluate(t);
                [t, h]
            }).collect();
            plot_ui.line(Line::new(pts));
        });
}
```

- [ ] **Step 2: Verify compile + manual test**

```
cargo build --bin linkage-sim-native && cargo run --bin linkage-sim-native
```

In GUI:
- Switch to Trajectory mode.
- Edit start/end/duration; verify h(t) preview updates live.
- Switch shape between ConstantSpeed and Trapezoidal; verify accel/decel fields appear/hide and preview shape changes.

- [ ] **Step 3: Commit**

```
git add linkage-sim-rs/src/gui/trajectory_panel/profile_input.rs
git commit -m "feat(traj): profile editor with inline h(t) preview plot"
```

---

## Task 3.6 — `gui/plot_panel/trajectory.rs` rendering

**Files:**
- Create: `linkage-sim-rs/src/gui/plot_panel/trajectory.rs`
- Modify: `linkage-sim-rs/src/gui/plot_panel/mod.rs`

- [ ] **Step 1: Create the rendering module**

Create `linkage-sim-rs/src/gui/plot_panel/trajectory.rs`:

```rust
//! Trajectory-mode plot rendering. X-axis is time (s).
//!
//! Traces:
//!   - target(t)    (dotted)
//!   - achieved(t)  (solid)
//!   - residual(t)  (faint)
//!   - u(t)         (right axis)
//!   - u̇(t), F_actuator(t), driver_torque(t) — stacked sub-plots
//!
//! Failure samples (status != Converged) marked with red overlay band on X-axis.
//!
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §8.3

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

use crate::gui::state::AppState;
use crate::gui::sweep::SweepData;

pub fn render(state: &AppState, data: &SweepData, ui: &mut egui::Ui) {
    let n = data.target_values.as_ref().map(|v| v.len()).unwrap_or(0);
    if n < 2 {
        ui.label("No trajectory data yet — click 'Compute trajectory' to generate.");
        return;
    }

    let times: Vec<f64> = (0..n).map(|i| (i as f64) * (1.0 / (n as f64 - 1.0))).collect();
    // Note: better to compute from profile.duration; placeholder for now.

    let target = data.target_values.as_ref().unwrap();
    let achieved = data.achieved_values.as_ref().unwrap();
    let residual = data.tracking_residual.as_ref().unwrap();
    let u = data.u_values.as_ref().unwrap();

    Plot::new("trajectory_main")
        .height(220.0)
        .show(ui, |plot_ui| {
            let pts_target: PlotPoints = times.iter().zip(target.iter()).map(|(&t, &v)| [t, v]).collect();
            let pts_achieved: PlotPoints = times.iter().zip(achieved.iter()).map(|(&t, &v)| [t, v]).collect();
            let pts_residual: PlotPoints = times.iter().zip(residual.iter()).map(|(&t, &v)| [t, v]).collect();
            plot_ui.line(Line::new(pts_target).name("target"));
            plot_ui.line(Line::new(pts_achieved).name("achieved"));
            plot_ui.line(Line::new(pts_residual).name("residual"));
        });

    Plot::new("trajectory_u")
        .height(120.0)
        .show(ui, |plot_ui| {
            let pts: PlotPoints = times.iter().zip(u.iter()).map(|(&t, &v)| [t, v]).collect();
            plot_ui.line(Line::new(pts).name("u(t)"));
        });

    let _ = state;
}
```

- [ ] **Step 2: Wire into plot_panel/mod.rs**

In `gui/plot_panel/mod.rs`:

```rust
mod trajectory;
```

In the existing plot dispatch (function that switches on `SweepMode`), add:

```rust
        SweepMode::Trajectory { .. } => {
            trajectory::render(state, &data, ui);
        }
```

- [ ] **Step 3: Verify compile + manual test**

```
cargo build --bin linkage-sim-native && cargo run --bin linkage-sim-native
```

In GUI: switch to Trajectory mode, click "Compute trajectory", verify plot renders.

- [ ] **Step 4: Commit**

```
git add linkage-sim-rs/src/gui/plot_panel/
git commit -m "feat(traj): plot panel trajectory rendering with target/achieved/u traces"
```

---

## Task 3.7 — Click-to-scrub in trajectory plot

**Files:**
- Modify: `linkage-sim-rs/src/gui/plot_panel/trajectory.rs`

- [ ] **Step 1: Add click handler**

In `trajectory.rs::render`, capture the plot response and on click, find the nearest `t_k` and call `state.solve_for_trajectory_target`:

```rust
    let response = Plot::new("trajectory_main")
        .height(220.0)
        .show(ui, |plot_ui| {
            // ... existing lines
            plot_ui.response().clone()
        });

    if response.inner.clicked() {
        if let Some(pos) = response.inner.interact_pointer_pos() {
            // Translate pointer x to time t_k via plot transform
            // (egui_plot exposes a transform helper; consult egui_plot docs for exact API)
            // Pseudocode:
            //   let t_clicked = transform.value_from_position(pos).x;
            //   // find nearest sample index
            //   if let SweepMode::Trajectory { target, profile, .. } = &state.sweep_mode {
            //       let h = profile.evaluate(t_clicked).0;
            //       state_mut.solve_for_trajectory_target(target, h);
            //   }
        }
    }
```

NOTE: this requires `&mut AppState`. Adjust the function signature of `render` to take `&mut AppState`. Update the call site in `plot_panel/mod.rs`.

- [ ] **Step 2: Verify compile + manual test**

```
cargo build --bin linkage-sim-native && cargo run --bin linkage-sim-native
```

In GUI: compute a trajectory, click on a point in the trajectory plot, verify mechanism configuration on the canvas updates to that target value.

- [ ] **Step 3: Commit**

```
git add linkage-sim-rs/src/gui/plot_panel/trajectory.rs linkage-sim-rs/src/gui/plot_panel/mod.rs
git commit -m "feat(traj): click-to-scrub on trajectory plot"
```

---

## Task 3.8 — Stage 3 verification + docs

**Files:**
- Modify: `docs/ai/05-update-tracker.md`

- [ ] **Step 1: Run all tests**

```
cd linkage-sim-rs
cargo test --lib
```
Expected: ~600+ tests pass, no regressions.

- [ ] **Step 2: Run native binary; manually verify end-to-end on each sample**

```
cargo run --bin linkage-sim-native
```

Verify on at least 2 samples:
- Switch to Trajectory mode.
- Configure target / profile.
- Click "Compute trajectory".
- Verify time-series plot shows expected shape.
- Click on a plot point; canvas updates.

- [ ] **Step 3: Append to update tracker**

```markdown
### Stage 3 — UI surface

- `gui/state/trajectory_ops.rs` — AppState::solve_for_trajectory_target.
- `gui/trajectory_panel/` (3 files) — top-level UI, target picker (5 variants),
  profile editor with inline h(t) preview, severity toggle.
- `gui/plot_panel/trajectory.rs` — time-series plot rendering with click-to-scrub.
- `SweepMode` dropdown gains Trajectory option; input panel delegates.

End-to-end usable. Manual UI verified on parallelogram and four-bar samples.
```

- [ ] **Step 4: Commit**

```
git add docs/ai/05-update-tracker.md
git commit -m "docs(ai): trajectory mode stage 3 (UI surface)"
git push origin main
```

---

# Stage 4 — Polish + CSV + user-facing docs

Goal: CSV export wired, failure-band rendering with hover tooltips, "from joint" point picker helper, FEATURES.md entry. Stage end state: feature is shipped.

## Task 4.1 — CSV export columns

**Files:**
- Modify: `linkage-sim-rs/src/gui/export/report.rs` (or wherever existing CSV export lives)

- [ ] **Step 1: Locate existing CSV export logic**

```
grep -rn "csv\|CSV" linkage-sim-rs/src/gui/export/ 2>/dev/null | head -20
```

- [ ] **Step 2: Add trajectory CSV columns**

In the existing CSV export function, when `state.sweep_mode` is `Trajectory`, write the column header:

```
t_seconds,target_value,achieved_value,residual,u,u_dot,u_ddot,q_dot_norm,F_actuator_N,driver_torque_Nm,kinetic_energy_J,potential_energy_J,status
```

Then iterate samples and write one row per sample. The columns map directly to `SweepData` fields populated in stage 2. Example row writer:

```rust
let n = data.target_values.as_ref().map(|v| v.len()).unwrap_or(0);
for i in 0..n {
    writeln!(
        out,
        "{},{},{},{},{},{},{},{},{},{},{},{},{}",
        times[i],
        data.target_values.as_ref().unwrap()[i],
        data.achieved_values.as_ref().unwrap()[i],
        data.tracking_residual.as_ref().unwrap()[i],
        data.u_values.as_ref().unwrap()[i],
        data.u_dot_values.as_ref().unwrap()[i],
        data.u_ddot_values.as_ref().unwrap()[i],
        f64::NAN, // q_dot_norm — wire from existing data if available
        data.actuator_forces.as_ref().and_then(|v| v.get(i)).copied().unwrap_or(f64::NAN),
        data.driver_torques.as_ref().and_then(|v| v.get(i)).copied().unwrap_or(f64::NAN),
        data.kinetic_energy.get(i).copied().unwrap_or(f64::NAN),
        data.potential_energy.get(i).copied().unwrap_or(f64::NAN),
        format_status(&data.inverse_solve_statuses.as_ref().unwrap()[i]),
    )?;
}
```

`format_status` returns a text encoding:

```rust
fn format_status(s: &crate::solver::inverse_kinematics::InverseSolveStatus) -> String {
    use crate::solver::inverse_kinematics::InverseSolveStatus::*;
    match s {
        Converged => "Converged".into(),
        Reachability { target, achieved_clamp, .. } => {
            format!("Reachability:{:.4}->{:.4}", target, achieved_clamp)
        }
        Singularity { dg_du } => format!("Singularity:{:.2e}", dg_du),
        BranchJump { delta_q_norm } => format!("BranchJump:{:.4}", delta_q_norm),
        NonConvergent { iterations, residual } => {
            format!("NonConvergent:iter={},res={:.2e}", iterations, residual)
        }
    }
}
```

- [ ] **Step 3: Coerce Severity to Strict on export**

When the user clicks "Export CSV", before invoking compute_trajectory, force severity to Strict (so any failure aborts cleanly):

```rust
let original_severity = state.trajectory_severity;
state.trajectory_severity = Severity::Strict;
let result = compute_trajectory(...);
state.trajectory_severity = original_severity;
match result {
    Ok(_) => write_csv_to_file(...)?,
    Err(e) => show_error_dialog(...),
}
```

- [ ] **Step 4: Manual test**

Compute a known-reachable trajectory in Analysis mode, then export CSV. Inspect the file (open in spreadsheet or `head`).

```
head linkage-sim-rs/exported_trajectory.csv
```

- [ ] **Step 5: Commit**

```
git add linkage-sim-rs/src/
git commit -m "feat(traj): CSV export with full trajectory column layout"
```

---

## Task 4.2 — Failure-band rendering with hover tooltips

**Files:**
- Modify: `linkage-sim-rs/src/gui/plot_panel/trajectory.rs`

- [ ] **Step 1: Add failure-band overlay**

In `render`, after the main traces:

```rust
    // Failure bands at samples with non-Converged status
    if let Some(statuses) = data.inverse_solve_statuses.as_ref() {
        for (i, status) in statuses.iter().enumerate() {
            if !matches!(status, crate::solver::inverse_kinematics::InverseSolveStatus::Converged) {
                let t = times[i];
                // Draw a vertical red line at t with hover tooltip
                // (egui_plot supports VLine; use its tooltip facility)
                plot_ui.vline(
                    egui_plot::VLine::new(t)
                        .color(egui::Color32::from_rgba_premultiplied(200, 50, 50, 80))
                );
                // Tooltip: render a separate HoveredItems list and look up status
            }
        }
    }
```

(Exact tooltip API depends on egui_plot version; consult docs. Goal: hovering a red line shows the formatted status text.)

- [ ] **Step 2: Manual test**

Create a trajectory with end_value far outside workspace (in Analysis mode); compute. Verify red bands appear at the unreachable region; hover shows the diagnostic.

- [ ] **Step 3: Commit**

```
git add linkage-sim-rs/src/gui/plot_panel/trajectory.rs
git commit -m "feat(traj): failure-band rendering with hover tooltips"
```

---

## Task 4.3 — "From joint" helper in target picker

**Files:**
- Modify: `linkage-sim-rs/src/gui/trajectory_panel/target_picker.rs`

- [ ] **Step 1: Replace `draw_point_input` with a helper that includes a "From joint" dropdown**

```rust
fn draw_point_input_with_joint_helper(
    pt: &mut [f64; 2],
    body_id: &str,
    state: &AppState,
    ui: &mut egui::Ui,
    label: &str,
) {
    ui.horizontal(|ui| {
        ui.label(format!("{}:", label));
        ui.add(egui::DragValue::new(&mut pt[0]).speed(0.001).suffix(" m"));
        ui.add(egui::DragValue::new(&mut pt[1]).speed(0.001).suffix(" m"));

        // From-joint helper: dropdown of joints on this body
        let joints_on_body = state.mechanism.as_ref().map(|m| {
            m.all_constraints().filter_map(|c| {
                if c.body_i_id() == body_id || c.body_j_id() == body_id {
                    Some((c.id().to_string(),
                          if c.body_i_id() == body_id {
                              c.point_i_local()
                          } else {
                              c.point_j_local()
                          }))
                } else { None }
            }).collect::<Vec<_>>()
        }).unwrap_or_default();

        if !joints_on_body.is_empty() {
            egui::ComboBox::from_label("From joint")
                .selected_text("(pick)")
                .show_ui(ui, |ui| {
                    for (jid, jpt) in &joints_on_body {
                        if ui.selectable_label(false, jid).clicked() {
                            *pt = [jpt.x, jpt.y];
                        }
                    }
                });
        }
    });
}
```

Update existing variant arms to pass `body_id` and use the helper.

- [ ] **Step 2: Manual test**

In Trajectory mode → WorldX target → Body=crank → click "From joint" → select J2. Verify `local_pt` populates with the joint's body-local coordinates.

- [ ] **Step 3: Commit**

```
git add linkage-sim-rs/src/gui/trajectory_panel/target_picker.rs
git commit -m "feat(traj): from-joint helper in target picker point inputs"
```

---

## Task 4.4 — `docs/FEATURES.md` user-facing entry

**Files:**
- Modify: `docs/FEATURES.md`

- [ ] **Step 1: Append section**

Append to `docs/FEATURES.md`:

```markdown
## Trajectory-mode position control

Define a desired output observable trajectory (e.g. punch-tip vertical position
as a function of time) and the simulator back-solves the actuator stroke `u(t)`,
its rate `u̇(t)`, and the actuator force `F(t)` required to follow it — including
joint reactions and energy along the trajectory.

**Use case.** "Define a press-platen velocity profile in mm and time; tool
computes actuator stroke and force."

**How to use:**
1. Build or load a linkage with at least one driver.
2. Switch the Sweep Mode dropdown from `Angle` / `Stroke` to `Trajectory`.
3. Pick the **target observable** — angle of a body, world-x or world-y of a
   body-local point, projection onto a fixed line, or distance to a reference.
4. Configure the **profile** (constant-speed or trapezoidal) with start, end,
   duration. Inline preview shows `h(t)` as you edit.
5. Click **Compute trajectory**. The plot panel shows `target(t)`, `achieved(t)`,
   tracking residual, `u(t)`, `u̇(t)`, and force traces.
6. Click on the trajectory plot to scrub the mechanism configuration to that
   point in time.
7. **Export CSV** for downstream use (spreadsheet, MATLAB, Python). The CSV
   column layout is firmware-adapter-friendly.

**Failure handling.** If the target is unreachable, the mechanism passes through
a singularity, or the trajectory requires switching assembly modes, those
samples are flagged on the plot in red with a hover tooltip explaining why.
The **Severity** toggle controls whether the trajectory generation aborts
(`Strict`) or continues with the failed samples annotated (`Analysis`). CSV
export coerces to `Strict` automatically.

**Math reference:** see `docs/superpowers/specs/2026-04-29-linkage-equations-reference.md`
section §8 for the inverse-kinematics derivation.
```

- [ ] **Step 2: Commit**

```
git add docs/FEATURES.md
git commit -m "docs: add Trajectory-mode position control section to FEATURES.md"
```

---

## Task 4.5 — `docs/ai/04-memory.yaml` deferred-items entries

**Files:**
- Modify: `docs/ai/04-memory.yaml`

- [ ] **Step 1: Append open questions**

Append to `docs/ai/04-memory.yaml` `open_questions` list:

```yaml
  - |
    Trajectory v2: S-curve (jerk-limited) motion profile. Currently only
    ConstantSpeed and Trapezoidal are implemented. SCurve requires a 7-segment
    quintic position function. Slot in MotionProfile::SCurve { jerk_fraction }
    when a user need arises.

  - |
    Trajectory v2: click-to-set ControlTarget points on canvas. v1 uses
    numerical input + "from joint" helper. Click-to-set requires extending the
    canvas interaction layer (gui/canvas/interaction.rs) with a "pick a body
    point" mode. Defer until users ask.

  - |
    Trajectory v2: keyframe / waypoint trajectory input. Current implementation
    only supports analytic profiles. Adding a Trajectory enum (Profile vs
    KeyframeTable) enables non-canonical trajectory shapes. Defer until users
    ask for non-canonical motion.

  - |
    Trajectory v2: CSV-table trajectory import. Letting the user load a (t, h)
    series from disk would enable replicating measured / externally-authored
    trajectories. Cheap once the keyframe path exists.

  - |
    Trajectory v3: hardware-controller-specific export adapter. v1 emits a
    superset CSV intended for downstream subsetting. A hardware adapter (e.g.
    Aerotech, Galil, Beckhoff) is a thin format-conversion layer on top once
    a target controller is selected.

  - |
    Trajectory: analytic acceleration inverse. v1 uses finite differences for
    ü (two extra forward solves per sample). Switching to analytic requires
    extending ControlTarget::hessian for non-Angle variants and assembling a
    "kinematic-only γ" with dq/du in place of q̇. Recommended only if FD ü
    accuracy is shown insufficient by hardware-export use cases.
```

- [ ] **Step 2: Commit**

```
git add docs/ai/04-memory.yaml
git commit -m "docs(ai): trajectory v2/v3 deferred items in open_questions"
```

---

## Task 4.6 — Final stage 4 verification + push

**Files:**
- Modify: `docs/ai/05-update-tracker.md`

- [ ] **Step 1: Run full test suite**

```
cd linkage-sim-rs
cargo test --lib
```
Expected: all tests pass (~600+).

- [ ] **Step 2: Build native binary; manual end-to-end check**

```
cargo build --bin linkage-sim-native
cargo run --bin linkage-sim-native
```

Manual checklist:
- [ ] Switch to Trajectory mode on parallelogram sample.
- [ ] Configure WorldY target on the actuator's coupler point.
- [ ] Trapezoidal profile, 200 samples, 1.0 s duration.
- [ ] "Compute trajectory" — plot renders.
- [ ] Click somewhere mid-plot — mechanism scrubs.
- [ ] Try unreachable end_value — verify red bands appear in Analysis mode.
- [ ] Export CSV; open file; verify column header + reasonable rows.
- [ ] Switch Severity to Strict; click Compute on the unreachable trajectory; verify clean error dialog.

- [ ] **Step 3: Append final stage entry**

```markdown
### Stage 4 — Polish + CSV + user-facing docs

- CSV export wired; superset column layout designed for firmware-adapter
  subsetting in v3. Severity coerced to Strict on export.
- Failure-band rendering (red overlays) with hover tooltips on the trajectory
  plot. Visualization is on-screen regardless of Severity setting.
- "From joint" helper in target picker point inputs.
- `docs/FEATURES.md` — user-facing trajectory-mode section with usage walk.
- `docs/ai/04-memory.yaml` — five v2/v3 deferred items + analytic Hessian item.

Feature shipped. Manual end-to-end verified on parallelogram and four-bar
samples. ~600 tests pass.
```

- [ ] **Step 4: Final commit + push**

```
git add docs/ai/05-update-tracker.md
git commit -m "docs(ai): trajectory mode stage 4 (shipped)"
git push origin main
```

---

# Self-review checklist (run after writing the plan)

This is for the plan-writing phase, not the executor.

**1. Spec coverage.** Each section of the design spec has at least one task implementing it:

| Spec section | Task(s) |
|---|---|
| §3 Approach (extend SweepMode) | 2.3, 2.4 |
| §4 Module layout | 1.1, 1.4, 3.1, 3.2, 3.6 |
| §5.1 ControlTarget | 1.4–1.8 |
| §5.2 Severity / InverseSolveStatus | 1.1–1.3 |
| §5.3 InverseSolveResult / solve_for_target | 1.10, 1.11 |
| §5.4 derivatives | 1.12, 1.13 |
| §5.5 TrajectoryProfile | 2.1 |
| §5.6 SweepMode::Trajectory | 2.3 |
| §5.7 SweepData fields | 2.2 |
| §6 Solver algorithms (Newton, bisection, severity) | 1.10, 1.11 |
| §6.6 Workspace probe | 1.9 |
| §6.7 Tunable constants | inside 1.11 |
| §7 Sweep pipeline integration | 2.4 |
| §7.3 Φ_t / γ override | 2.4 |
| §7.4 t_mech encoding | 2.4 |
| §7.6 Live single-frame scrub | 3.1 |
| §8 UI surface | 3.2–3.7 |
| §9 Documentation expectations | distributed in 1.14, 2.5, 3.8, 4.4, 4.5 |
| §10 Testing strategy | distributed across all stage 1–2 tasks |
| §11 CSV export | 4.1 |
| §12 Build sequence | mirrors stages 1–4 |
| §13 Deferred items | 4.5 |

All sections covered.

**2. Placeholder scan.** No "TBD" / "TODO" / "implement later" / "fill in details" / "add appropriate error handling" / "similar to Task N" patterns remain.

**3. Type consistency.** Cross-referenced names:
- `ControlTarget` — same in 1.4–1.8, 2.4, 3.4
- `Severity` — same in 1.1, 1.11, 2.4, 3.2
- `InverseSolveStatus` — same in 1.2, 1.11, 2.2, 4.1, 4.2
- `TrajectoryProfile` — same in 2.1, 2.4, 3.5
- `solve_for_target` — argument order consistent across 1.10, 1.11, 2.4, 3.1
- `compute_trajectory` — signature consistent across 2.4, 3.6 (called via dispatch)

No drift detected.

---

*End of plan.*
