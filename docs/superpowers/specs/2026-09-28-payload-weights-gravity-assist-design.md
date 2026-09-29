# Payload Weights and Gravity Assist — Design

**Date:** 2026-09-28
**Status:** Approved by user in interactive brainstorm (all four design sections)
**Scope:** `linkage-sim-rs/` (solver analysis, sweep, GUI), `docs/ai/backlog.yaml`

## Purpose

The user models a mechanism that **lifts a robot**. Parts of the robot's mass,
and parts of the linkage's own mass, work in the actuator's favour over some of
the stroke and against it over the rest. The simulator must make that visible
and trustworthy:

- Weights can be placed on links and **dragged** to where the real masses sit.
- The user can see, at every angle, **which weight helps and which hurts**, and
  by how much, in actuator force and in actuator power.
- The user can see where in the cycle the **load drives the actuator**
  (braking) versus where the actuator drives the load (motoring).

Success: with a robot-lift model loaded, the user drags weights around and the
canvas colours, the breakdown plot, and the braking bands match physical
intuition, with numbers verified against an independent energy check.

## What already works (evidence, 2026-09-28 scouting run)

- A weight is a point mass on a body: `PointMassJson { mass, local_pos }`
  (`src/io/schema.rs:213-216`), folded into the body's composite mass, CG and
  Izz at build time (`src/core/body.rs:117-137`, `src/io/from_json.rs:122-126`).
- Gravity acts at every composite CG (`src/forces/elements/evaluation.rs:41-59`),
  so weights enter statics, the two-pass actuator solve, and the power-balance
  actuator force `F = -P_applied / Ldot` (`src/solver/reactions.rs:372-403`).
- **Empirical probe:** 50 kg on the ParallelogramActuator rocker and on the
  ChebyshevLambdaActuator coupler, two positions each, sizing mode. The solver's
  change in driver torque, actuator force, and actuator power matched the
  independent value `-m g·v` (finite-differenced mass-point velocity) to a max
  relative error of 1.2e-6, with correct sign at 100% of valid angles. Moving the
  weight changed results exactly as predicted; on a purely translating coupler,
  position had no effect (3.7e-11 N·m).
- GUI today: a `+ Mass` tool (hard-coded 1 kg), numeric mass and body-local X/Y
  in the link editor, one-click Reposition, and Move to Link.

## Key physics statement (drives the readout design)

In the quasi-static, frictionless model, the actuator force needed to hold a
payload at a pose is **the same going up and coming down**. Direction changes
**who does the work**: lifting, the actuator does positive work (motoring);
lowering, the load does work on the actuator (braking). Gravity "helping"
therefore appears as **negative actuator power**, not as a smaller force. On the
force plot this can look inverted: at 200° on the parallelogram, gravity assists
yet the retracting actuator pushes about 500 N harder to hold the weight back.
Direction-dependent friction is out of scope (see Out of scope).

## Decomposition

### Track 1 — correctness fixes (backlog items, fix-campaign batches)

These are defects that corrupt the numbers once a user edits the model. They go
into `docs/ai/backlog.yaml` with the probe evidence and run through the existing
fix loop. **Track 2 implementation starts only after the compound-actuator fix
(first item) is merged.**

1. **Compound actuator geometry.** Any rebuild expands a mount-point actuator
   into cylinder + rod bodies. (a) `remap_force_to_compound`
   (`src/forces/compound.rs:306`) anchors the force at the cylinder base
   `(0,0)` instead of the slide point `(half_len,0)` (`compound.rs:124-126`), so
   actuator length becomes `|L - L0/2|` and the plotted force flips sign at 83 of
   361 parallelogram angles. (b) `initial_length`/`half_len`
   (`compound.rs:175-176`) are computed from body-local points of two different
   bodies (`src/io/from_json.rs:25-33, 272-273`), producing a phantom end-stop
   force up to ~1245 N on the Chebyshev sample. Likely root cause of **BL-017**.
2. **Save/autosave/share double-counts point masses.** `mechanism_to_json`
   writes composite mass (`src/gui/state/file_io.rs:143`), then point masses are
   re-attached (`:150`) and applied again on load. Verified 51 kg → 101 kg.
3. **`set_body_mass` / `set_body_izz` overwrite composite values** in the live
   mechanism (`src/gui/state/blueprint_ops.rs:449-486`), dropping point masses
   from gravity until the next rebuild.
4. **Point-mass edits are not undoable.** `update_point_mass`
   (`blueprint_ops.rs:956-970`) never calls `push_undo`; Move to Link records two
   undo entries and two rebuilds.
5. **Stored-force mode plots `F_required − F_stored`** (`src/gui/sweep/mod.rs:647-649`,
   `src/solver/reactions.rs:579`) while the label says "(computed)".
6. **Inverse dynamics lacks the velocity-quadratic term.**
   `solve_inverse_dynamics` uses only `M(q)·q̈` (`src/solver/inverse_dynamics.rs:67-70`)
   although `M` depends on configuration (`src/solver/assembly.rs:133-145`).
   Verified error `2·m·s_y·ω²` = 3158 N·m with an off-pivot rocker mass. Corrupts
   the "With Inertia" actuator curve. Independent of Track 2; may ride along.

### Track 2 — payload feature (this spec, then an implementation plan)

Sections below.

## Track 2 design

### 1. Data model

- `PointMassJson` gains `id: String` (stable, e.g. `"W1"`) and
  `label: Option<String>`, both `#[serde(default)]`. Files without ids load
  unchanged; ids are assigned on load (`W<n>`, unique per mechanism) so every
  weight is addressable by id rather than by list index.
- Link self-weight is **not stored**: the breakdown derives one entry per
  non-ground body from the blueprint base `mass` and `cg_local`, named after the
  body's label (or id).
- Loader validation: a point mass on ground is rejected with a warning, and a
  negative mass is rejected instead of silently ignored
  (`src/core/body.rs:121-123` currently ignores it).

### 2. Breakdown physics

New pure-function module `src/analysis/gravity_breakdown.rs`. Inputs: built
mechanism, `q`, `q_dot`, gravity vector, the weight list (id, name, body,
body-local position, mass, including derived link self-weights), and the
actuator stroke speed `Ldot` (or driver rate `ω` when there is no actuator).

For each weight *i* at one sweep sample:

- **Gravity power** `P_g,i = m_i · (g · v_i)`, with `v_i` from
  `State::body_point_velocity` (`src/core/state.rs:211`). `P_g,i > 0` ⇒ the
  weight is **helping**.
- **Actuator power share** `P_act,i = −P_g,i`.
- **Actuator force share** `F_i = −P_g,i / Ldot` (same power balance as
  `reactions.rs:372-403`, one weight at a time).
- **Non-gravity loads** `F_other = F_total − Σ F_i` (force zones, springs,
  external loads), so entries always sum to the actuator's required force.
- **No actuator:** the same shares are expressed as driver-torque shares,
  `T_i = −P_g,i / ω`.

The gravity vector is read from the mechanism's gravity element (single source;
respects `mounting_angle`). Gravity loads are linear, so superposition is exact.

Power shares are in watts at the driver's configured speed (velocities come
from the sweep's constant-rate velocity solve); force shares are
speed-independent.

**Motoring vs braking:** actuator power `P_act = F_total · Ldot`; braking when
`P_act < −τ`, with `τ = 1e-6 · max|P_act|` over the sweep (named constant).

**Helping vs hurting per weight:** helping when `P_g,i > δ_i`, hurting when
`P_g,i < −δ_i`, neutral otherwise, with `δ_i = 0.01 · max|P_g,i|` over the sweep
for that weight (the weight is momentarily moving horizontally).

**Edge cases:**

- **Near stroke reversal:** when `|Ldot| < ε_rel · max|Ldot|` over the sweep,
  with named constant `ε_rel = 0.01`, force shares are `NaN` (plotted as a gap;
  hover shows "-"); power shares and the helping/hurting classification remain
  defined, because they do not divide by `Ldot`.
- **Solver failure rows:** per-weight series get `NaN` via `push_nan_row`
  (`src/gui/sweep/mod.rs:843`), preserving the all-channels-same-length invariant
  (`docs/ai/02-system.yaml`).
- **Stored-force mode:** shares are defined against the **required** force
  (after Track 1 item 5).
- **Stroke-mode sweeps with a `LinearDriver` and no `LinearActuator`
  element** get the same breakdown, as shares of the driver force (basis
  `DriverTorque`, plotted as "Driver Force Share"). This was added during
  implementation because it falls out of the same power balance; it was
  originally out of scope (accepted deviation, 2026-09-29).
- **Out of scope for v1:** a per-weight split of the "With Inertia" curve.

**Integration:** the sweep loop calls the module once per converged sample and
stores `SweepData::weight_breakdown: Option<WeightBreakdown>` (sources, basis,
per-source gravity power, force share and power share, other and total force
and power, braking flags), every series aligned with `angles_deg`.

### 3. GUI

**Placing.** The `+ Mass` tool keeps its two-click flow (link, then location).
While active, a mass field appears in the toolbar, defaulting to the last mass
used (not 1 kg). The drop point snaps to the grid when snapping is on. New
weights get default names (`W1`, `W2`, …).

**Selecting and dragging.**

- Weights get hit targets (new point-mass hit test in
  `src/gui/canvas/hit_testing.rs`) and highlight on hover.
- `SelectedEntity` (`src/gui/state/types.rs:124`) gains a `Weight` variant
  keyed by `(body_id, weight_id)`.
- Press-drag shows a live preview; the blueprint mutation and rebuild happen
  **once on release** (`drag_stopped()`), per the existing lesson that per-frame
  rebuilds are too expensive.
- On release the nearest link wins, within the existing 60 px pick radius:
  if a different link is nearer than the weight's own link, the weight
  reattaches there, preserving its world position; otherwise it moves on its
  current link (accepted deviation, 2026-09-29: prevents accidental
  reattachment while dragging along the weight's own link). Each drag is **one** undo step
  (`AppState::mutate_and_rebuild`). Delete removes the selected weight.

**Property panel.** A selected weight shows name, mass, owning link, and
body-local position. The link editor's weights section is always visible and
gains an **Add weight** button.

**Canvas readout at the current pose.**

- Each weight draws an arrow in the gravity direction, length scaled by mass;
  **green** when helping, **red** when hurting, **gray** when neutral (per the
  `δ_i` rule in the breakdown physics section).
- Hover or selection shows name, mass, and current force share (no permanent
  labels, to avoid clutter).
- The actuator label gains words: push/pull from the sign convention
  (positive = extension push, `src/forces/elements/evaluation.rs:424-425`) and
  motoring/braking from the sign of `P_act`, e.g. "1.2 kN push, braking".

**Plots.**

- Actuator Force and Actuator Power plots shade the angle ranges where the
  actuator is braking.
- New `PlotTab::WeightBreakdown` (`src/gui/plot_panel/mod.rs:16`): one line per
  weight (link self-weights included), plus `other` and `total`; a toggle
  switches force share / power share; colours match the canvas arrows. Lines,
  not stacked areas, because shares carry mixed signs.
- Fix the two tooltips that state "positive = tension"
  (`src/gui/plot_panel/mod.rs:184`, `src/gui/property_panel/force_editor.rs:972`);
  the code's convention is positive = extension.

**Unchanged by design:** health utilisation and motor sizing keep using force
magnitudes (`abs`), which is correct for capacity sizing: the actuator must hold
the load in either direction.

### 4. Testing and validation

The breakdown is `risk: physics`: every physics change gets the
`fbd-math-reviewer` pass.

**Physics tests.**

- Hand-computed: single bar on a pivot with one weight; `P_g` equals
  `m·g·(vertical velocity)` at several angles, ascending and descending.
- Independent reference (probe method): each weight's `P_g` matches the
  finite-difference change of its potential energy between adjacent sweep angles,
  on ParallelogramActuator and ChebyshevLambdaActuator with two weights each.
- **Sum invariant:** at every valid angle, `Σ F_i + F_other = F_total` and the
  power equivalent; link-self-weight plus point-mass entries equal the model's
  total gravity load.
- **Mutation check:** flip the gravity sign inside the breakdown, confirm the
  tests go red, restore.
- Classification: descending weight ⇒ helping; ascending ⇒ hurting;
  load-driven actuator ⇒ braking.
- Edge cases: near reversal ⇒ force share `NaN`, power share finite;
  forced-failure sweep ⇒ every weight series has `angles_deg.len()` entries.

**Data tests.** An old file without ids loads, gets ids, and round-trips ids,
names, and masses intact (mass-doubling itself is covered by Track 1 item 2).

**GUI tests (headless egui, as in `gui::input_panel::tests`).**

- Hit test finds a weight under the cursor and misses empty space.
- A simulated drag moves the weight on release, records exactly one undo step,
  and reattaches when dropped over another link.
- Placement mass field defaults to the last mass used.
- Actuator label produces correct push/pull and motoring/braking words for each
  sign combination.

**Gate and hands-on.** Every change passes `scripts/gate.sh` and a
fresh-context review. Before merge, `gui-smoke` runs against the WASM build, and
the user runs a hands-on checklist on a robot-lift model.

## Out of scope

- Weights that move during the cycle (sliding or swinging payloads).
- Direction-dependent friction or efficiency losses (up-stroke vs down-stroke
  force difference).
- Parametric position study (peak force vs weight position).
- Per-weight split of the "With Inertia" curve.
