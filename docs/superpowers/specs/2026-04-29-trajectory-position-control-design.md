# Trajectory-mode position control — design spec

**Date:** 2026-04-29
**Status:** Design — ready for implementation review.
**Companion:** `2026-04-29-linkage-equations-reference.md` (math layer; cited as **EQ §X** below).

---

## 1. Purpose & scope

Add a **trajectory-mode** to the linkage analyzer in which the user specifies an *output observable* (e.g. punch-tip vertical position, link angle, projection onto a fixed line) as a function of time, and the tool back-solves the actuator input parameter `u(t)` together with `u̇(t), ü(t), F_actuator(t)` and the full kinematic / kinetic state along the trajectory.

The feature is the **inverse** of the existing forward sweep: instead of "given driver input, compute output", it answers "given desired output, compute driver input — and tell me when my mechanism can't deliver."

**In scope (v1):**
- Five `ControlTarget` variants covering the common 1-D observables (angle, world-x, world-y, projection-onto-line, distance-to-point).
- Analytic motion-profile trajectories (constant-velocity, trapezoidal; S-curve deferred to v2).
- Inverse position + closed-form inverse velocity + finite-difference inverse acceleration.
- Per-sample reuse of existing statics / inverse-dynamics so `F_actuator(t)` and joint reactions are computed automatically.
- Configurable failure semantics (`Strict` for export, `Analysis` for on-screen).
- Time-series plot panel with target / achieved / residual / `u` / `u̇` / forces.
- CSV export designed for downstream firmware-adapter use.

**Out of scope (deferred):**
- Click-to-set `ControlTarget` points on the canvas (v2).
- S-curve / jerk-limited motion profile (v2).
- Keyframe / waypoint trajectory input (v2).
- CSV-table trajectory import (v2).
- Hardware-controller-specific export formats (v2 — adapter on top of the v1 CSV).
- True 2-DOF position control (requires structural addition of a second driver — separate project).
- The R1b refactor (collapsing dual-purpose driver scalars into `DriverKind` payload, see `docs/ai/04-memory.yaml`) is independent and not blocked by this work.

---

## 2. Requirements

Resolved during brainstorming:

| Q | Decision |
|---|---|
| **Q1: What is being controlled?** | Configurable per analysis via `ControlTarget` enum: Angle / WorldX / WorldY / Projection / Distance. Same solver and data model regardless of variant. |
| **Q2: How is the trajectory specified?** | Analytic motion profile only for v1. `TrajectoryProfile` composes the existing `MotionProfile` shape (`ConstantSpeed` / `Trapezoidal`) with absolute `start_value`, `end_value`, `duration`. |
| **Q3: Deliverables?** | All of: time-series plots, tracking residual, per-timestep actuator force, CSV export, reaction forces / joint torques per timestep. Firmware-format export deferred to v2. |
| **Q4: Failure semantics?** | Configurable `Severity::{Strict, Analysis}`. Strict aborts on any failure (for CSV / firmware export). Analysis annotates failed samples with status payload (for on-screen). Failure regions rendered red with hover tooltips *regardless* of severity. |

---

## 3. Approach choice — extend SweepMode

Among three considered architectures:

- **A.** New parallel `gui/trajectory/` pipeline (clean isolation, ~200 LoC of structural duplication).
- **B. (chosen)** Extend `SweepMode` with a `Trajectory` variant; reuse the sweep pipeline.
- **C.** Refactor sweep into a generic `AnalysisMode` abstraction first.

**Rationale for B:**
- The existing sweep pipeline already loops over an input parameter, evaluates per-sample (statics, inverse-dynamics, energy, reactions), accumulates into `SweepData`, and renders. Trajectory mode reuses every one of these.
- The math difference (forward solve vs. inverse Newton) is localized: one new match arm in `compute_sweep_data`. The math itself is fully decoupled into `solver/inverse_kinematics/`.
- C is over-engineered for one new mode (premature abstraction). A introduces real duplication that violates DRY without offsetting risk.
- If a fourth analysis mode ever appears, *that* is the natural moment for C; we'll have two real modes to draw the abstraction from rather than one.

---

## 4. Module layout

### 4.1 New: `solver/inverse_kinematics/`

```
solver/inverse_kinematics/
  mod.rs              — public re-exports
  control_target.rs   — ControlTarget enum + per-variant g, ∇g, ∇²g
  solver.rs           — solve_for_target outer Newton + bisection fallback
  derivatives.rs      — inverse_velocity (closed-form), inverse_acceleration_fd
  severity.rs         — Severity enum, InverseSolveStatus, error variants
```

No GUI dependencies. Tested standalone.

### 4.2 New: `gui/trajectory_panel/`

```
gui/trajectory_panel/
  mod.rs              — top-level input UI, severity toggle, advanced options
  target_picker.rs    — ControlTarget selector + body / point / axis pickers
  profile_input.rs    — MotionProfile + start/end/duration editor with inline h(t) preview
```

Active only when `SweepMode::Trajectory` is selected.

### 4.3 New: `gui/plot_panel/trajectory.rs`

Renders trajectory traces (target, achieved, residual, `u`, `u̇`, `F_actuator`) with time as X-axis. Wires into the existing `gui/plot_panel/mod.rs` mode dispatch.

### 4.4 New: `gui/state/trajectory_ops.rs`

`AppState::solve_for_trajectory_target(target, h)` for interactive scrubbing — the live single-frame analogue of one trajectory iteration. Mirrors the existing `solve_at_angle` / `solve_at_stroke` in `gui/state/mod.rs`.

### 4.5 Modified (additive)

| File | Change |
|---|---|
| `gui/sweep/mod.rs` | `SweepMode::Trajectory { target, profile, severity, n_samples }` variant; new match arm dispatching to `compute_trajectory` (new helper, sibling of existing per-sample loop). Existing `Angle` / `Stroke` paths untouched. |
| `gui/sweep/motion_profile.rs` | Small generalization so the trapezoidal evaluator works against an arbitrary scalar axis. Existing callers preserved. |
| `gui/state/types.rs` | `SweepData` gains seven `Option<Vec<f64>>` time-series fields (`target_values`, `achieved_values`, `tracking_residual`, `u_values`, `u_dot_values`, `u_ddot_values`) plus `Option<Vec<InverseSolveStatus>>` for diagnostics. All `None` outside trajectory mode. |
| `gui/plot_panel/mod.rs` | Mode dispatch adds `Trajectory` branch → `plot_panel::trajectory::render`. |
| `gui/input_panel.rs` | When `SweepMode::Trajectory` is active, delegates the input section to `gui::trajectory_panel`. |

### 4.6 Untouched

`core/`, `forces/`, `analysis/`. The Φ_t / γ driver-row substitution that trajectory mode needs (§7) lives in `compute_trajectory`, not in `core/constraint/`.

---

## 5. Data model

### 5.1 `ControlTarget` (in `solver/inverse_kinematics/control_target.rs`)

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum ControlTarget {
    Angle { body_id: String },
    WorldX { body_id: String, local_pt: [f64; 2] },
    WorldY { body_id: String, local_pt: [f64; 2] },
    Projection {
        body_id: String,
        local_pt: [f64; 2],
        axis_origin: [f64; 2],
        axis_dir: [f64; 2],   // normalized on construction
    },
    Distance {
        body_id: String,
        local_pt: [f64; 2],
        ref_pt: [f64; 2],
    },
}

impl ControlTarget {
    pub fn evaluate(&self, mech: &Mechanism, q: &DVector<f64>) -> f64;
    pub fn gradient(&self, mech: &Mechanism, q: &DVector<f64>) -> DVector<f64>;
    pub fn hessian(&self, mech: &Mechanism, q: &DVector<f64>) -> DMatrix<f64>;
    pub fn unit_label(&self) -> &'static str; // "rad", "m", "m" (proj), "m" (distance)
}

// Constructor functions enforce input validity:
//   - body_id exists in the mechanism
//   - local_pt is finite
//   - For Projection: axis_dir non-zero, normalized in-place
// Direct struct construction is allowed but the caller is then responsible
// for invariants. Recommended path: ControlTarget::angle(...), ::world_x(...),
// ::projection(...), ::distance(...).
```

Per-variant math definitions and Hessian shapes documented in **EQ §8** (specifically §8.1 for the gradient table; §8.3 for the Hessian-of-observable contributions to acceleration inverse).

### 5.2 `Severity` and `InverseSolveStatus` (in `severity.rs`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    Strict,    // Hard-error on any per-sample failure. CSV / firmware export default.
    Analysis,  // Annotate failed sample with status payload. On-screen default.
}

#[derive(Debug, Clone)]
pub enum InverseSolveStatus {
    Converged,
    Reachability {
        target: f64,
        achieved_clamp: f64,
        workspace_min: Option<f64>,
        workspace_max: Option<f64>,
    },
    Singularity { dg_du: f64 },
    BranchJump { delta_q_norm: f64 },
    NonConvergent { iterations: usize, residual: f64 },
}
```

### 5.3 `InverseSolveResult` (in `solver.rs`)

```rust
pub struct InverseSolveResult {
    pub u: f64,
    pub q: DVector<f64>,
    pub achieved: f64,
    pub residual: f64,
    pub iterations: usize,
    pub status: InverseSolveStatus,
}

pub fn solve_for_target(
    mech: &Mechanism,
    q0: &DVector<f64>,
    target: &ControlTarget,
    h: f64,
    severity: Severity,
    tol: f64,
    max_iter: usize,
) -> Result<InverseSolveResult, LinkageError>;
```

### 5.4 `inverse_velocity`, `inverse_acceleration_fd` (in `derivatives.rs`)

```rust
pub fn inverse_velocity(
    mech: &Mechanism,
    q: &DVector<f64>,
    target: &ControlTarget,
    h_dot: f64,
) -> Result<f64, LinkageError>;

pub fn inverse_acceleration_fd(
    mech: &Mechanism,
    q: &DVector<f64>,
    target: &ControlTarget,
    u: f64,
    u_dot: f64,
    h_ddot: f64,
    delta: f64,
) -> Result<f64, LinkageError>;
```

Math: **EQ §8.2** and **EQ §8.3**.

### 5.5 `TrajectoryProfile` (in `gui/state/types.rs` or near `MotionProfile`)

The existing `MotionProfile` enum (`gui/state/mod.rs:45`) is preserved as the *shape* descriptor. New struct adds absolute units:

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrajectoryProfile {
    pub shape: MotionProfile,    // existing — ConstantSpeed | Trapezoidal{ accel_fraction, decel_fraction }
    pub start_value: f64,        // h(0) in target units
    pub end_value: f64,          // h(duration)
    pub duration: f64,           // seconds
}

impl TrajectoryProfile {
    /// Returns (h(t), ḣ(t), ḧ(t)).
    pub fn evaluate(&self, t: f64) -> (f64, f64, f64);
    pub fn sample_times(&self, n: usize) -> Vec<f64>;
}
```

### 5.6 `SweepMode` extension (in `gui/sweep/mod.rs`)

```rust
pub enum SweepMode {
    Angle,
    Stroke,
    Trajectory {
        target: ControlTarget,
        profile: TrajectoryProfile,
        severity: Severity,
        n_samples: usize,
    },
}
```

### 5.7 `SweepData` field additions (in `gui/state/types.rs`)

All `Option<Vec<f64>>`, populated only in trajectory mode:

```rust
pub target_values: Option<Vec<f64>>,
pub achieved_values: Option<Vec<f64>>,
pub tracking_residual: Option<Vec<f64>>,
pub u_values: Option<Vec<f64>>,
pub u_dot_values: Option<Vec<f64>>,
pub u_ddot_values: Option<Vec<f64>>,
pub inverse_solve_statuses: Option<Vec<InverseSolveStatus>>,
```

Existing fields (driver_torques, joint_reactions, kinetic_energy, etc.) populate identically in trajectory mode.

---

## 6. Solver layer algorithms

### 6.1 `solve_for_target` outer Newton loop

```text
fn solve_for_target(mech, q0, target, h, severity, tol, max_iter):
    1. Workspace probe: cached scan of u over [u_min, u_max] in N_probe ≈ 64
       samples. Yields (u_k, g(q(u_k))) curve and g_min, g_max.
    2. Reachability check: if h ∉ [g_min, g_max] (with slack), set status =
       Reachability { workspace_min, workspace_max, achieved_clamp = clamp(h) }.
       Strict → return Err. Analysis → continue with clamped target so a sample
       is still produced (with the failure flagged in the status field).
    3. Bracket: bisect the probe table to find u_seed with g(q(u_seed)) ≈ h.
       Becomes the warm-start.
    4. Outer Newton (k = 1..max_iter):
         q_k       = solve_position(mech, q_{k-1}, u_k)              // forward solve, EQ §3
         r_k       = target.evaluate(q_k) - h
         if |r_k| < tol: converged, break
         dq_du     = -Φ_q⁻¹ · Φ_u                                    // 1 linear solve
         r_prime_k = target.gradient(q_k) · dq_du
         if |r_prime_k| < ε_singularity:
           status = Singularity { dg_du: r_prime_k }; fall through to bisection
         u_{k+1}   = u_k - r_k / r_prime_k
         if u_{k+1} ∉ [u_min, u_max]: clamp and flag near-boundary
         if ‖q_k - q_{k-1}‖ > branch_jump_threshold:
           status = BranchJump; revert q_k to q_{k-1}; fall through to bisection
    5. If outer Newton stalled, run bisection inside the bracket as fallback.
    6. Return InverseSolveResult.
```

Math reference: **EQ §8.1**.

**Cost per outer iteration:** one warm-started forward solve (~3-5 SVDs) + one linear solve for `dq/du`. Outer Newton typically converges in 2-4 iterations. ~10-25 SVDs per trajectory sample. nalgebra 9×9 SVD ≈ 1-3 µs → 1000-sample trajectory ≲ 50 ms wallclock.

### 6.2 Bisection fallback (private helper in `solver.rs`)

When Newton stalls, singularity is detected, or branch-jump occurs: refine `(u_lo, u_hi)` from the workspace probe until `|r(u_mid)| < tol` or sub-step. Linear convergence, immune to local divergence. ~10 forward solves per fallback. Reported as `Converged` with a `bisected: true` flag on the result.

### 6.3 `inverse_velocity` (closed-form)

```text
fn inverse_velocity(mech, q, target, h_dot):
    dq_du   = -Φ_q⁻¹ · Φ_u
    r_prime = target.gradient(q) · dq_du
    if |r_prime| < ε_singularity: return Err(SingularInverse)
    return h_dot / r_prime
```

Math: **EQ §8.2**.

### 6.4 `inverse_acceleration_fd` (finite-difference)

```text
fn inverse_acceleration_fd(mech, q, target, u, u_dot, h_ddot, δ):
    Solve forward at u + δ → q_plus  ; r_prime_plus  = ∇g · dq/du at u + δ
    Solve forward at u − δ → q_minus ; r_prime_minus = ∇g · dq/du at u − δ
    r_double_prime = (r_prime_plus − r_prime_minus) / (2δ)
    r_prime_now    = ∇g · dq/du at u
    return (h_ddot − r_double_prime · u_dot²) / r_prime_now
```

Math: **EQ §8.3** (with explicit note that FD sidesteps both the `∇²g` and `d²q/du²` analytic terms).

### 6.5 Severity dispatch

```rust
fn classify_or_fail<T>(status: InverseSolveStatus, severity: Severity, partial: T)
    -> Result<(InverseSolveStatus, T), LinkageError>
{
    match severity {
        Severity::Strict   => Err(LinkageError::from(status)),
        Severity::Analysis => Ok((status, partial)),
    }
}
```

Same Newton loop and FD helpers in both modes; only the wrapper differs.

### 6.6 Workspace-probe cache

Probe results live on `AppState` keyed by `(mech_revision, target_hash, u_range)`. `mech_revision` is a counter incremented inside the existing `rebuild()` (see `gui/state/blueprint_ops.rs`). `target_hash` is a deterministic hash of the `ControlTarget` value. `u_range` is the `(u_min, u_max)` interval (see §6.6.1). Recompute is ~64 forward solves (~1 ms typically); invalidation is automatic on any blueprint edit, target change, or range change.

#### 6.6.1 Where `u_min`, `u_max` come from

The probe scans the driver's input parameter over an explicit range. Source of that range, in priority order:

1. **Existing sweep-range UI**, if set — the user has already declared "I care about this angle/stroke window"; the trajectory probe stays inside it. This makes reachability bounds align with what the user sees in the existing sweep plot.
2. **Default fallback**: for a revolute driver, `[u_0_initial − π, u_0_initial + π]` (full revolution centered on current). For a linear driver, `[u_0_initial − 0.1, u_0_initial + 0.1]` (±100 mm centered on current).

User can widen / narrow via the existing sweep-range input — no new UI required. If the trajectory's reachable workspace is genuinely outside the configured sweep range, that is itself a `Reachability` failure with an actionable diagnostic ("trajectory needs u ∈ [0.05, 0.20] but configured range is [0.0, 0.10] — widen the range or relax the trajectory").

### 6.7 Tunable constants

| Constant | Default | Rationale |
|---|---|---|
| `branch_jump_threshold` | `0.5 × max(body_length)` | "Bodies shouldn't move more than half their longest dimension between adjacent timesteps." Conservative; tunable per-mechanism if needed. |
| `ε_singularity` | `1e-6 × ‖∇g‖ × ‖dq/du‖` | Relative threshold; avoids false-positives on ill-scaled units. |
| `N_probe` | `64` | Enough resolution for a 1-DOF curve; ~1 ms compute. |
| `δ` (FD step) | `1e-4 × (u_max − u_min)` | Empirically robust at double precision. |
| `tol` | `1e-8` | Matches existing `solve_position` default. |
| `max_iter` | `50` | Newton-on-Newton; 50 is generous given typical 2-4 outer iter convergence. |

---

## 7. Sweep pipeline integration

### 7.1 Dispatch in `compute_sweep_data`

```rust
match sweep_mode {
    SweepMode::Angle  => { /* unchanged */ }
    SweepMode::Stroke => { /* unchanged */ }
    SweepMode::Trajectory { target, profile, severity, n_samples } => {
        compute_trajectory(mech, q0, target, profile, severity, *n_samples, &mut data)?
    }
}
```

`compute_trajectory` is a new function in `gui/sweep/mod.rs` alongside the existing per-sample loop. Will be extracted to `gui/sweep/trajectory.rs` if it grows past ~250 LoC during implementation.

### 7.2 Per-sample loop in `compute_trajectory`

```text
nominal_rate = current_driver_rate(mech)        // ω for revolute, v for linear
driver_row   = mech.n_constraints() - 1
q_prev       = q0.clone()
u_0_initial  = current_driver_value(mech)

for k in 0..n_samples:
    t_k = profile.duration * k / (n_samples - 1)
    (h_k, h_dot_k, h_ddot_k) = profile.evaluate(t_k)

    // 1. Inverse position
    res    = solve_for_target(mech, &q_prev, target, h_k, severity, tol, max_iter)?
    q_k    = res.q;  u_k = res.u;  status = res.status

    // 2. Encode u_k in mechanism's t-frame
    t_mech_k = (u_k - u_0_initial) / nominal_rate

    // 3. Inverse velocity & acceleration
    u_dot_k  = inverse_velocity(mech, &q_k, target, h_dot_k)?
    u_ddot_k = inverse_acceleration_fd(mech, &q_k, target, u_k, u_dot_k, h_ddot_k, δ)?

    // 4. Body velocity / acceleration with trajectory rates substituted (see §7.3)
    phi_q              = assemble_jacobian(mech, &q_k, t_mech_k)
    let mut phi_t      = assemble_phi_t(mech, &q_k, t_mech_k)
    phi_t[driver_row]  = -u_dot_k                      // override
    q_dot_k            = phi_q.svd().solve(&-phi_t)?

    let mut gamma      = assemble_gamma(mech, &q_k, &q_dot_k, t_mech_k)
    gamma[driver_row]  = u_ddot_k                       // override
    q_ddot_k           = phi_q.svd().solve(&gamma)?

    // 5. Existing per-sample machinery — unchanged
    statics  = solve_statics(mech, &q_k, t_mech_k)
    inv_dyn  = solve_inverse_dynamics(mech, &q_k, &q_dot_k, &q_ddot_k, t_mech_k)
    energy   = compute_energy_state_mech(mech, &q_k, &q_dot_k, gravity)
    react    = extract_reactions(mech, &statics)
    F_act    = ...                                       // already wired for stroke mode

    // 6. Push to SweepData (new + existing fields)
    push_trajectory_fields(&mut data, t_k, h_k, target.evaluate(&q_k),
                           u_k, u_dot_k, u_ddot_k, status)
    push_existing_fields(&mut data, q_k, q_dot_k, statics, inv_dyn, energy, react, F_act)

    q_prev = q_k
```

### 7.3 The Φ_t / γ driver-row override (rationale)

The existing constraint code computes `phi_t = -ω` and `gamma_driver_row = 0` because the driver is parameterized as constant-speed (`f(t) = θ_0 + ωt` ⟹ `f''(t) = 0`). For trajectory mode the *back-solved* input rates `u_dot_k` and `u_ddot_k` differ each sample.

**Two options considered:**

- **(A)** Plumb a "trajectory rate" through every constraint's `phi_t` / `gamma`. Touches `core/`, breaks the constant-speed parameterization invariant. **Rejected.**
- **(B, chosen)** Run `assemble_phi_t` / `assemble_gamma` normally, then override the driver row. Two lines per sample. `core/` untouched.

**Why this is safe:** the driver row is structurally identical between constant-speed and trajectory parameterizations — same column non-zeros in `Φ_q`. Only the time-derivative scalars differ, and those live at known indices (`driver_row = mech.n_constraints() - 1`, since drivers are added last).

**Correctness:** for constant-speed driver `f(t) = θ_0 + ω t`, `f'(t) = ω`, `f''(t) = 0`. For trajectory mode at sample `k`, the equivalent is a (locally) time-varying parameterization with `f'(t_k) = u_dot_k`, `f''(t_k) = u_ddot_k`. The `Φ_q` is unchanged (it depends only on `q`, not on the parameterization), so substituting the new rate into `Φ_t` and the new acceleration into `γ` produces correct `q̇` and `q̈` — see **EQ §4** and **EQ §5** for the underlying equations.

### 7.4 The `t_mech` encoding trick

The existing `solve_position(mech, q0, t)` expects a time-frame argument compatible with the driver's parameterization. To re-use it without modification, trajectory mode encodes the desired `u_k` as `t_mech_k = (u_k - u_0_initial) / nominal_rate`. This makes `f(t_mech_k) = θ_0 + ω·t_mech_k = u_0_initial + (u_k - u_0_initial) = u_k`, which is exactly what we want.

Same trick is already used by `solve_at_angle` (`gui/state/mod.rs:885`) and `solve_at_stroke` (`gui/state/mod.rs:1033`) — the trajectory branch is just a deeper application of the same idea. `MIN_DRIVER_OMEGA_ABS = 0.01` (`core/driver.rs:188`) guarantees `nominal_rate ≠ 0`.

### 7.5 Reuse of existing solvers

`solve_statics`, `solve_inverse_dynamics`, `extract_reactions`, `compute_energy_state_mech`, and the actuator-force computation are called **with no modifications**. They consume `(q, q_dot, q_ddot, t)` and don't care that `q_dot` came from a back-solved rate rather than `ω`. This is the principal payoff of choosing approach B.

### 7.6 Live single-frame scrub

`AppState::solve_for_trajectory_target(target, h)` (in the new `gui/state/trajectory_ops.rs`) is the per-frame analogue: identical to one iteration of §7.2 but called from the input panel when the user drags a "current target" slider. Updates `self.q`, `self.driver_*`. Mirrors `solve_at_angle` / `solve_at_stroke`.

---

## 8. UI surface

### 8.1 Sweep-mode dropdown

Existing `SweepMode::{Angle, Stroke}` selector adds a `Trajectory` entry. When selected:

- Input panel switches to the trajectory editor.
- Sweep-range degrees / mm controls hide.
- Plot X-axis switches to time (s).
- "Run sweep" button label changes to "Compute trajectory".

### 8.2 Trajectory input panel — `gui/trajectory_panel/`

Three top-to-bottom sections.

**Target observable** (`target_picker.rs`)

Radio group with five options matching `ControlTarget` variants. Only the relevant fields show for the selected variant:

- *Angle*: body dropdown.
- *WorldX* / *WorldY*: body dropdown + body-local point (two `f64` inputs, with a "from joint" helper dropdown that fills in coordinates of an existing joint on that body).
- *Projection*: body + local_pt + axis_origin (two `f64`) + axis_dir (two `f64`, normalized on submit).
- *Distance*: body + local_pt + ref_pt (two `f64`).

A small live readout below shows `g(q_current) = <value>` so the user sees the observable's *current* value before defining a target trajectory for it. Click-to-set on canvas is deferred to v2.

**Profile editor** (`profile_input.rs`)

Reuses the existing `MotionProfile` enum (no new shape types for v1):

- Shape dropdown: `ConstantSpeed` / `Trapezoidal{accel_fraction, decel_fraction}`. (`SCurve` greyed with "v2".)
- `start_value: f64` (target units, follows selected `ControlTarget`).
- `end_value: f64` (same units).
- `duration: f64` (seconds).
- For trapezoidal: `accel_fraction`, `decel_fraction` (existing fields).
- `n_samples: usize` (default 200).

A tiny inline plot beneath the form shows `h(t)` as the user edits values — purely visual feedback, no solver involvement. ~30 LoC of egui plot.

**Severity & solve options** (in `mod.rs`)

- Toggle: `Severity::{Strict, Analysis}`. Default `Analysis`. CSV export coerces to `Strict` automatically.
- Tolerance / max-iterations under collapsible "Advanced" with sensible defaults.

### 8.3 Plot panel — `gui/plot_panel/trajectory.rs`

X-axis: time (s). Y-axis grouping follows existing axis convention:

- **Left axis** — `target(t)` (dotted), `achieved(t)` (solid), `tracking_residual(t)` (faint). Observable units.
- **Right axis** — `u(t)` (rad or m).
- **Stacked sub-plot 1** — `u̇(t)` (rad/s or m/s).
- **Stacked sub-plot 2** — `F_actuator(t)` (N) and `driver_torque(t)` (N·m).

**Failure rendering.** Samples with `status != Converged` get a red overlay band on the X-axis at `t_k`. Hovering shows the `InverseSolveStatus` payload (e.g. `"Reachability failure: target 0.092m exceeds workspace max 0.087m"`). Visualization is on-screen *regardless* of `Severity` setting — the toggle controls trajectory generation, not failure visibility.

**Click-to-scrub.** Existing sweep-plot click handler is extended: clicking on a time `t_k` calls `state.solve_for_trajectory_target(target, profile.evaluate(t_k).0)`. Same UX as current angle/stroke scrubbing.

### 8.4 Live-frame slider

When `SweepMode::Trajectory` is active, the existing angle/stroke slider is *replaced* by a "current-target" slider with range `[start_value, end_value]`. Dragging it updates `state.q` via `solve_for_trajectory_target`. Lets the user explore the workspace before committing to a full trajectory compute.

### 8.5 Persistence

`SweepMode` is already part of the JSON save/load. The new `Trajectory { target, profile, severity, n_samples }` variant inherits this for free — `ControlTarget`, `TrajectoryProfile`, and `Severity` carry serde annotations from §5. No new save/load wiring.

---

## 9. Documentation expectations

Per the standing rule that docs ride with code, the work is structured so docs land *as part of* each PR, not as cleanup.

### 9.1 Spec doc (this document)

Cross-references the equations reference for math; cross-referenced *back* from each new module's top-of-file `//!` block.

### 9.2 Module docs

Every new file gets a `//!` block stating: purpose, key invariants, and a `// See: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §X` cross-reference.

| Module | Cross-ref |
|---|---|
| `solver/inverse_kinematics/control_target.rs` | EQ §8 — definitions + per-variant `g`, `∇g`, `∇²g` |
| `solver/inverse_kinematics/solver.rs` | EQ §8.1 — Newton outer loop |
| `solver/inverse_kinematics/derivatives.rs` | EQ §8.2 (closed-form velocity), EQ §8.3 (FD acceleration) |
| `solver/inverse_kinematics/severity.rs` | This spec §6.5 |
| `gui/trajectory_panel/*` | This spec §8 |

### 9.3 Inline comments (only where non-obvious)

- The `phi_t[driver_row] = -u_dot_k` and `gamma[driver_row] = u_ddot_k` overrides in `compute_trajectory` (this spec §7.3).
- The `t_mech_k = (u_k - u_0_initial) / nominal_rate` encoding (§7.4).
- The workspace-probe cache key `(mech_revision, target_hash)` (§6.6).
- The `branch_jump_threshold = 0.5 * max(body_length)` heuristic (§6.7).
- The `δ = 1e-4 × workspace_span` for FD acceleration inverse (§6.7).

### 9.4 Repo-level doc updates (per-stage)

Each implementation stage's PR includes the doc updates relevant to its scope:

**Stage 1 (solver only):**
- `docs/ai/02-system.yaml` — add invariants:
  - `severity_controls_return_type_not_math`: "Strict and Analysis severities run the same Newton/FD code; they differ only in whether failure status bubbles as Err or as a result field."
  - `workspace_probe_cache_key`: "(mech_revision, target_hash, u_range). Invalidated on any blueprint edit, target change, or range change."
- `docs/ai/05-update-tracker.md` — entry: "Trajectory-mode position control: stage 1 (solver layer)."

**Stage 2 (sweep extension):**
- `docs/ai/02-system.yaml` — add invariants:
  - `trajectory_mode_phi_t_override`: "compute_trajectory overrides Φ_t and γ on the driver row to substitute back-solved u_dot/u_ddot; constraint code in core/ remains constant-speed-parameterized."
  - `trajectory_mode_t_mech_encoding`: "u_k is encoded as t_mech_k = (u_k − u_0)/nominal_rate to reuse existing forward solvers."
- `docs/ai/05-update-tracker.md` — append: "stage 2 (sweep dispatch)."

**Stage 3 (UI surface):**
- `docs/ai/05-update-tracker.md` — append: "stage 3 (UI: trajectory_panel + plot_panel/trajectory)."

**Stage 4 (polish + CSV):**
- `docs/FEATURES.md` — new "Trajectory-mode position control" subsection with a one-screen description and a worked use case ("Define a press-platen velocity profile in mm and time; tool computes actuator stroke and force.").
- `docs/ai/05-update-tracker.md` — final entry: "stage 4 (CSV + polish), feature shipped."
- `docs/ai/04-memory.yaml` — open questions: `trajectory_v2_scurve`, `trajectory_v2_canvas_point_picking`, `trajectory_v2_keyframe_input`, `trajectory_v2_csv_import`, `trajectory_v3_firmware_export_adapter`. Existing R1b entry is unaffected.

### 9.5 Tests as documentation

Every `ControlTarget` variant gets a *forward → inverse round-trip test* in `solver/inverse_kinematics/control_target.rs` `#[cfg(test)] mod tests`: pick a target value `h`, run `solve_for_target` → get `u`, then forward-solve at `u` and verify `g(q) ≈ h`. Test name and body double as a worked example for that variant.

---

## 10. Testing strategy

### 10.1 Unit tests — `solver/inverse_kinematics/`

| File | Tests |
|---|---|
| `control_target.rs` | Forward-inverse round-trip for each of 5 variants on a 4-bar with revolute driver and a parallelogram with linear driver. ~10 tests. |
| `solver.rs` | Newton convergence on a known target, bisection-fallback when Newton stalls, branch-jump detection, singularity detection, max-iter behavior. ~6 tests. |
| `derivatives.rs` | `inverse_velocity` agrees with FD of `solve_for_target`; `inverse_acceleration_fd` agrees with second FD; both error correctly on singularities. ~4 tests. |
| `severity.rs` | `Strict` returns `Err` on each failure mode; `Analysis` returns `Ok` with matching status. ~4 tests. |

### 10.2 Failure-handling test matrix

| Status | Construction | Strict expects | Analysis expects |
|---|---|---|---|
| `Reachability` | End_value outside workspace probe range | `Err(LinkageError::TrajectoryUnreachable)` | `Ok` with status, achieved_clamp populated |
| `Singularity` | 4-bar with collinear coupler/rocker (transmission angle 0°) | `Err` | `Ok` with `dg_du < ε` |
| `BranchJump` | Crank trajectory crossing a known toggle position | `Err` | `Ok` with `delta_q_norm` |
| `NonConvergent` | Pathological `tol`/`max_iter` combination | `Err` | `Ok` with iteration count |

Plus one **multi-failure trajectory test** that constructs a trajectory hitting all four modes in sequence and verifies the on-screen plot would show four distinct red bands (assertion on `data.inverse_solve_statuses`).

### 10.3 Integration tests — `gui/sweep/mod.rs` (or `gui/sweep/trajectory.rs` if extracted)

- Trajectory in `Analysis` mode produces all expected fields populated and matching length `n_samples`.
- Trajectory in `Strict` mode aborts at the first failed sample with a clear error.
- `compute_trajectory` reuses existing per-sample machinery without breaking forward-sweep tests.

### 10.4 Test count summary

~18 new unit tests + ~3 integration tests = ~21 new tests for v1. Existing 578 tests (current count) must continue to pass.

---

## 11. CSV export column layout

Designed as a **superset** that any future firmware-export adapter can subset from. Order is left-to-right "what we wanted, what we got, what the actuator did":

```
t_seconds, target_value, achieved_value, residual,
u, u_dot, u_ddot,
q_dot_norm, F_actuator_N, driver_torque_Nm,
kinetic_energy_J, potential_energy_J,
status
```

| Column | Units | Notes |
|---|---|---|
| `t_seconds` | s | Trajectory time. |
| `target_value` | observable units | From `TrajectoryProfile.evaluate`. |
| `achieved_value` | observable units | `target.evaluate(q_k)`. |
| `residual` | observable units | `achieved − target`. |
| `u` | rad or m | Input parameter; units depend on driver kind (header row identifies). |
| `u_dot`, `u_ddot` | rad/s, rad/s² (or m/s, m/s²) | Back-solved input rates. |
| `q_dot_norm` | varies | Norm of body-coordinate velocities; useful for sanity checks. |
| `F_actuator_N` | N | Populated for linear drivers; `NaN` for revolute. |
| `driver_torque_Nm` | N·m | Populated for revolute drivers; `NaN` for linear. |
| `kinetic_energy_J`, `potential_energy_J` | J | From existing energy computation. |
| `status` | text | `"Converged"`, `"Reachability:0.092m>0.087m"`, `"Singularity:dg_du=2.3e-7"`, etc. |

**Behavior under severity:**

- `Strict` mode aborts CSV generation on the first failed sample; returns clear error with offending `t_seconds`.
- `Analysis` mode writes failure rows with `status` filled and numeric columns possibly `NaN` or last-good values.

---

## 12. Build sequence — staged PRs

| Stage | Lands | Risk | Tests |
|---|---|---|---|
| **1. Solver only** | `solver/inverse_kinematics/` subtree (4 files), no GUI changes, not yet wired into `compute_sweep_data`. | Zero — pure addition. | ~18 unit tests (round-trip + failure modes). |
| **2. Sweep extension** | `SweepMode::Trajectory` variant, `compute_trajectory` in `gui/sweep/mod.rs`, `SweepData` field additions. UI doesn't yet expose trajectory mode. | Low — additive. Existing forward-sweep paths unchanged. | ~3 integration tests against canned trajectories. Existing sweep tests still pass. |
| **3. UI surface** | `gui/trajectory_panel/` (3 files), `gui/plot_panel/trajectory.rs`, `gui/state/trajectory_ops.rs`, mode-dropdown extension, click-to-scrub wiring. End-to-end usable. | Medium — touches input panel and plot dispatch. UI smoke test required. | Manual UI walkthrough on parallelogram + slider-crank samples. |
| **4. Polish + CSV** | CSV export, severity Strict/Analysis enforcement at export boundary, failure-band rendering with hover tooltips, inline `h(t)` preview plot, "from joint" point-picker helper. `docs/FEATURES.md` user-facing entry. | Low — UI-only refinements. | Manual export verification; round-trip CSV through a minimal Python parser for sanity. |

Stage 5 (firmware export adapter) is **deferred** until a target controller is picked. It is a thin format conversion on top of the §11 CSV layout and does not gate the rest.

Total work estimate: ~1500 LoC of new code + ~500 LoC of tests + the doc updates above. Realistic three to four sessions of focused work.

---

## 13. Open questions / deferred items

| Item | Status | Notes |
|---|---|---|
| S-curve / jerk-limited motion profile | v2 | Greyed-out option in shape dropdown. |
| Click-to-set `ControlTarget` points on canvas | v2 | Numerical input + "from joint" helper sufficient for v1. |
| Keyframe / waypoint trajectory input | v2 | Defer until users ask for non-canonical profiles. |
| CSV-table trajectory import | v2 | Cheap to add once `Trajectory` enum is introduced. |
| Hardware-controller-specific export formats | v2 | Adapter on top of §11 CSV; needs target controller spec. |
| Analytic acceleration inverse | future | FD is sufficient for v1; analytic only if controllers demand >1e-6 ü accuracy. |
| 2-DOF position control (true xy) | future | Requires structural mechanism change (second driver). Different scope. |
| **R1b** — collapse driver scalars into `DriverKind` payload | tracked in `docs/ai/04-memory.yaml` | Independent of this work. May make trajectory code slightly cleaner if landed first; not blocking. |

---

## 14. References

- `docs/superpowers/specs/2026-04-29-linkage-equations-reference.md` — math layer (constraint catalog, position/velocity/acceleration solvers, statics, inverse-kinematics derivation).
- `docs/ai/02-system.yaml` — stable invariants and limitations.
- `docs/ai/04-memory.yaml` — open questions, including R1b.
- `docs/ai/05-update-tracker.md` — decision log.
- `docs/FEATURES.md` — user-facing feature catalog.

---

*End of design spec. Implementation plan to follow as a separate document under `docs/superpowers/plans/`.*
