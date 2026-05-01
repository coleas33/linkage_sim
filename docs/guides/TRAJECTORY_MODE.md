# Trajectory Mode (Inverse Position Control)

The two `Sweep mode` options most users start with — **Angle** and **Stroke** — are *forward* analyses: you prescribe the actuator input (crank angle or actuator stroke) and the solver computes everything else (poses, forces, energies). **Trajectory** mode flips that: you prescribe a desired output observable as a function of time (e.g. "I want the coupler tip's X coordinate to follow this curve over 1 second"), and the solver back-solves the actuator input `u(t)` that makes it happen.

This is what you use for:
- **Pick-and-place**: prescribe end-effector position, get the actuator command.
- **Path tracing**: drive a coupler point along a target curve and read out the required servo profile.
- **Profile design**: compare trapezoidal vs. S-curve actuator profiles to a fixed end-effector trajectory.
- **Reachability checks**: see whether the requested motion is mechanically possible at every sample.

If you've used inverse kinematics in robotics, the math is the same. The simulator implements a Newton outer loop on `r(u) = g(q(u)) − h(t) = 0`, where `g` is the chosen observable, `q(u)` is the constrained pose at input `u`, and `h(t)` is your prescribed target.

The math is documented in detail in [`docs/superpowers/specs/2026-04-29-linkage-equations-reference.md`](../superpowers/specs/2026-04-29-linkage-equations-reference.md) §8.

---

## Quick Start (5 minutes)

This walkthrough drives the **4-Bar Crank-Rocker** sample's crank angle along a prescribed time profile. You'll see target and achieved curves overlay perfectly because `Angle` on the directly-driven body is the trivial case (`dg/du = 1`) — perfect for verifying the pipeline works before moving on to non-trivial targets.

### Step 1 — Load the sample

`File → Load Sample → 4-Bar Crank-Rocker`

You should see a 3-body 4-bar linkage. The bottom status bar reads `4-Bar Crank-Rocker, 3B 4J DOF=0` and the canvas shows the bars in their initial pose.

### Step 2 — Switch to Trajectory mode

Top-right toolbar: **Sweep mode → Trajectory**.

The bottom status bar now reads `No trajectory data yet — switch to Trajectory mode and click Compute trajectory.` That's expected; switching modes doesn't auto-compute.

### Step 3 — Configure the target

Scroll the left panel down past the Property panel. You'll see four collapsing sections:

1. **Target observable** — what you want to control. Defaults to `Angle of crank`. Leave it.
2. **Profile** — the time function for the target. Defaults to `ConstantSpeed` from `0.0` to `1.0` over `1.0 s`. Leave it.
3. **Visualization** — motion ribbon and playback options.
4. **Failure handling** — Continue (default) or Abort on solver failure.

### Step 4 — Compute

Click the blue **▶ Compute trajectory** button at the bottom of the left panel.

A trajectory plot replaces the bottom-of-screen plot tabs. You should see:

- **Orange dashed line** — `target(t)`: rises linearly from 0 to 1 rad over 1 s.
- **Cyan solid line** — `achieved(t)`: should overlay the orange dashed line *exactly*.
- **Faint red line** — `residual(t) = achieved − target`: should be a flat zero at the bottom.
- **Green line** (stacked plot below) — `u(t)`: also rises linearly, identical to target.

If achieved tracks target and residual is ≈ 0 everywhere, **the inverse-kinematics pipeline is working correctly on this trivial case**. This is the equivalent of `2 + 2 = 4` for the trajectory solver.

### Step 5 — Click-to-scrub

Click anywhere on the trajectory plot. The mechanism on the canvas snaps to the configuration corresponding to that time point. A vertical gold line on the plot marks your scrub position. This is the fastest way to inspect mid-trajectory poses without animating.

### Step 6 — Animated playback

In the **Visualization** section of the left panel:

- **▶ Play trajectory** — animates the canvas through `q(t)` at the trajectory's actual time scale.
- **⏹ Stop** — resets to `t = 0`.
- **Speed multiplier** — 0.05× to 4×.
- **Loop** — wraps back to `t = 0` at the end.

This animates at the *real* time scale (not the constant-omega kinematic animation). A 1 s trajectory with `Speed = 1.0×` plays back in exactly 1 s.

---

## Reading the Trajectory Plot

```
┌──────────────────────────────────────────────┐
│  target / achieved / residual                │ ← main plot
│   - target(t)  (orange dashed)               │
│   - achieved(t) (cyan)                       │
│   - residual(t) = achieved - target (red)    │
│                                              │
├──────────────────────────────────────────────┤
│  u(t)  (green)                               │ ← stacked
│   - back-solved actuator input               │
└──────────────────────────────────────────────┘
```

**Legend interpretation:**

- `target` and `achieved` overlap → solver tracked the requested output exactly.
- `target` and `achieved` diverge → either the target is unreachable (solver clamped to workspace) or the solver got stuck (singularity, branch jump, non-convergence).
- `u(t)` is what you'd command an actuator to do to produce this motion. Smooth `u` → easy on hardware. Spiky `u` near workspace boundaries → torque/velocity demand spikes.

---

## When the Solve Fails

Failed samples appear as red vertical bands in the plot, with single-letter glyphs at the top of each band:

| Glyph | Meaning | Typical cause |
|-------|---------|---------------|
| **R** | **R**eachability — target outside the mechanism's workspace at this time | Profile end-value beyond physical reach; check workspace bounds. |
| **S** | **S**ingularity — `\|dg/du\|` near zero | Mechanism is at a dead point or branch boundary; the requested observable is locally insensitive to the input. |
| **B** | **B**ranch jump — pose moved farther than the local Newton step expects | Mechanism flipped to a different assembly mode mid-trajectory. |
| **N** | **N**on-convergent — Newton outer loop ran out of iterations | Numerical conditioning issue; try a finer profile or check the target observable. |

Below the plot, a collapsing section lists every failed sample with its full diagnostic payload (workspace bounds, `dg/du`, residual norm, etc.).

In **Failure handling → Continue** mode (default), failures get annotated and the trajectory continues. In **Abort** mode, the first failure aborts the entire trajectory. Use Abort when correctness matters more than partial results (e.g. firmware export).

---

## Choosing a Target Observable

Five `ControlTarget` variants are exposed in the **Target observable** picker:

| Variant | What it controls | Units |
|---------|------------------|-------|
| **Angle** | A body's orientation angle θ | rad |
| **WorldX** | World-frame X coordinate of a point on a body | m |
| **WorldY** | World-frame Y coordinate of a point on a body | m |
| **Projection** | Signed distance of a body point along a fixed world-frame unit vector | m |
| **Distance** | Distance between two body points | m |

`Angle` on the directly-driven body is the trivial case. The interesting ones are **WorldX / WorldY / Projection** on a coupler point — those are the "drive the end-effector to (x, y) at time t" use cases.

The **Pick** button next to each `body / point / axis` field lets you click a point on the canvas to fill it in, instead of typing IDs.

---

## Choosing a Profile

The **Profile** section lets you pick between two top-level shapes:

### Profile (analytic)

| Shape | Velocity profile | When to use |
|-------|------------------|-------------|
| **ConstantSpeed** | `h(t) = start + (end − start) · (t / duration)` | Simplest case; `h_dot` is constant, `h_ddot` is zero. |
| **Trapezoidal** | Accelerate → cruise → decelerate, with `accel_fraction` and `decel_fraction` | Industry-standard for actuator commands. |
| **SCurve** | Quintic ease-in-out, `s(τ) = τ³(10 − 15τ + 6τ²)` | Continuous jerk; smoother than trapezoidal. |

### Keyframes (table)

A waypoint table with `(t, value)` rows, linearly interpolated between waypoints. Use when the desired motion isn't a closed-form profile (e.g. a CAM-data import). The Keyframes mode supports **Drag a `.csv` onto the canvas** to import waypoints — column 1 is `t_seconds`, column 2 is the target value. See `compute_trajectory_populates_all_trajectory_fields` in the test suite for the exact integration semantics.

---

## Verifying Correctness

Beyond the visual check in Step 4 (target and achieved overlap), three deeper sanity checks:

1. **Lib test `compute_trajectory_populates_all_trajectory_fields`** — exercises the full pipeline on the canonical 4-Bar with `Angle` target, asserts `achieved[k] = target[k]` within `1e-6`, `u_dot[k] = (end − start) / duration`, and that the per-sample pose snapshot's θ_crank exactly matches the target. Run with `cargo test --features native --lib gui::sweep::tests::compute_trajectory_populates_all_trajectory_fields`.

2. **Cross-check with a forward sweep** — switch to **Angle** mode, set the crank to a specific angle, and read out the body angles. Then switch to **Trajectory** with `Angle of crank` target and a profile that hits that same angle. The `q(t)` snapshot at that time should match.

3. **Per-sample pose snapshot** — exporting **File → Export Sweep CSV** in Trajectory mode appends `q_x_<body>_m`, `q_y_<body>_m`, `q_theta_<body>_rad` columns. Plot the body's θ in your favorite tool against the target column; they should overlap for the directly-driven body.

---

## Advanced Features

### Comparison overlay

In the **Visualization** section, click **💾 Save as comparison**. The current trajectory is frozen as a faded dotted overlay on the plot. Now change something — switch the profile from ConstantSpeed to Trapezoidal, or change the target body, or adjust the duration — and click Compute again. Both trajectories render simultaneously: the live one solid, the snapshot dotted. Useful for A/B'ing two profiles or two targets without exporting CSVs.

### Motion ribbon

Toggle **Show motion ribbon (ghost poses)** in the Visualization section. The canvas renders N evenly-spaced ghost poses of the mechanism behind the live pose. Useful for visualizing the swept path without animating.

### Per-sample exports

Three exports populate the trajectory data:

- **Export Sweep CSV** — time-series with target / achieved / u / u̇ / ü / F_actuator / driver_torque / status, plus per-body pose columns.
- **Export firmware (JSON)** — same data in a machine-friendly schema (`linkage-traj-firmware-v1`); intended as the input for downstream actuator controllers. Includes a `pose` map per sample for replay.
- **Generate Report (HTML)** — embeds the trajectory plot in an HTML report alongside the mechanism schematic, dimensions, and Grashof classification.

The CSV / JSON exports include the per-body pose snapshot so downstream tools can replay `q(t)` without rerunning the inverse solve.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| "Compute trajectory" button does nothing | Mechanism has no driver, or the target body doesn't exist | Add a driver via the Driver panel; verify target body in the **Target observable** section. |
| All samples fail with **R** | Profile end-value outside reachable workspace | Reduce the profile range, or pick a different target body. |
| Spiky `u(t)` near a sample | Solver near a singularity at that time | Inspect the canvas at that scrub position; if the linkage is at a dead point, redesign the profile to avoid it. |
| `u(t)` is constant zero | Target observable is constant (e.g. `WorldX` of a ground-attached point) | Pick a moving body. |
| Trajectory plot stays empty after Compute | A panic occurred during solve (rare); check the error panel | Check console / error log; report the failure mode. |

If you see a panic in the WASM build's console, please file an issue with the panic message — the trajectory pipeline has 668+ lib tests but novel mechanism topologies can still hit untested code paths.
