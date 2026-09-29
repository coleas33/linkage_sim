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
- On release the weight lands at the drop point (snapped to the grid when
  snapping is on). The nearest link within the existing 60 px pick radius
  takes it, its own link included: a drop nearest a different link reattaches
  the weight there, and a drop nearer its own link, or with no link in range,
  moves it on its current link (accepted deviation, 2026-09-29: prevents
  accidental reattachment while dragging along the weight's own link). Ground
  and compound actuator bodies never take a weight. Each drag is **one** undo
  step (`AppState::mutate_and_rebuild`). Delete removes the selected weight.

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

## Hands-on checklist (robot-lift model)

Run this after the automated gate passes, before merging Track 2. Tick each
box in the real GUI. A box that fails goes into `docs/ai/backlog.yaml` with
the angle and the numbers seen. The numbers quoted come from the sample's
default sweep (0-360 deg in 1 deg steps) with W1 at the rocker tip.

**Caution (known issue):** while the model is still the loaded sample, that
is until section 6 reopens it from a file, changing the driver rebuilds the
stock sample and discards every weight and every other edit. The driver
changes on right-click of a ground pivot joint (**Set as Driver**, **Set
Driver to …**) or when a load case with another driver joint is applied.
Ctrl+Z brings everything back, and a model reopened from a file keeps its
weights. Section 6 checks the undo.

### 0. Build and smoke test

1. From `linkage-sim-rs/`: `bash scripts/gate.sh` prints `GATE PASS`. The
   tests rewrite `docs/chebyshev_lambda/*.png`; restore them with
   `git checkout -- ../docs/chebyshev_lambda/`.
2. Build and serve the WASM app from `linkage-sim-rs/`:
   `bash scripts/build_web.sh` (needs the `wasm32-unknown-unknown` target
   and `wasm-bindgen-cli`), then `bash scripts/serve_web.sh` in a second
   shell. It serves http://localhost:8080; leave it running.
3. In Claude Code, run the `gui-smoke` skill (workflow
   `.claude/workflows/gui-smoke.js`; default URL http://localhost:8080,
   override with args `{"url": "..."}`). It must report `passed: true`: the
   page loaded, a `<canvas>` is present, and the console shows no errors.
4. Do sections 1-6 in the served page, or natively with
   `cargo run --release --bin linkage-gui` from `linkage-sim-rs/`.

### 1. Model the lift

- [ ] File > Load Sample > **Parallelogram + Actuator**. Leave the
      actuator's stored force **F** at the sample's 50 N: every plot and the
      actuator label show the required force whatever F is (section 6
      checks this).
- [ ] Click **+ Mass**: a mass field appears in the toolbar. Set it to 50 kg
      and click the rocker. The canvas hint reads "Click to place weight W1
      (50 kg) on 'rocker' (Esc to cancel)", and a preview circle shows where
      the weight lands (on the grid when snapping is on). Click the rocker's
      tip, the end joined to the coupler: **W1** appears, selected, and the
      tool returns to Select. If its **Position** does not read X 0 mm,
      Y 0 mm (the tip), type those values.
- [ ] Click **+ Mass** again: the field shows 50 kg, the last mass used. Set
      20 kg, click the coupler, then a point on it: **W2** appears. Type
      "Robot torso" in its **Name** field and press Enter.
- [ ] With W2 selected, the property panel shows "Weight Robot torso (W2)",
      "Link: coupler" and its Name, Mass and body-local Position fields.
      Pick the coupler in the Link Editor: its **Weights (1)** section lists
      W2 and has an **Add weight** button. The rocker's lists W1.
- [ ] Pick the rocker in the Link Editor and click **Add weight**: W3
      (20 kg, the last mass used) appears at the rocker's centre of mass.
      One Ctrl+Z removes it.
- [ ] Click W2's Mass field, type 25 and press Esc: the mass stays 20 kg.
      Type 25 and press Enter: it is 25 kg, and one Ctrl+Z puts back 20 kg.
      While typing, Backspace edits the text and does not delete the
      selected weight.

### 2. Select, drag, reattach, delete

- [ ] Hover a weight: it gets a highlight ring and a grab cursor. Click W1
      at the rocker tip: W1 is selected, not the joint under it. Shift+click
      W2: both are selected; Shift+click W2 again drops it from the
      selection.
- [ ] Drag W2 along the coupler: a dashed line and a marker follow the
      pointer (on the grid when snapping is on), and the mechanism does not
      re-solve until release. After release, one Ctrl+Z puts W2 back.
- [ ] Drag W2 over the rocker and release nearer the rocker than the
      coupler, within 60 px: the rocker is highlighted while dragging, and W2
      moves to the rocker at the drop point with the same id, name and mass.
      One Ctrl+Z reverts it.
- [ ] Start dragging W2, then press Esc, or release outside the canvas:
      nothing changes.
- [ ] Select W1 and press Delete: it is removed. Ctrl+Z restores it as W1.

### 3. Canvas readout at the current pose

Set the crank angle with the **Crank Angle** slider, or by clicking a plot.

- [ ] At 45 deg both weight arrows are red (being lifted), W1's longer than
      W2's (the length grows with the mass). The actuator label reads
      "7.9 kN push, motoring".
- [ ] At 135 deg both arrows are green (coming down); the label reads
      "1.3 kN push, braking".
- [ ] At 90 deg the arrows are gray (moving sideways); the label reads
      "0.00 N", with no push/pull or motoring/braking word.
- [ ] Hover W1 at 45 deg: a tooltip shows "W1", "Mass: 50 kg" and
      "Force share: +5.5 kN (hurting)". Select it: the same readout stays
      next to the weight. With neither hover nor selection, no weight label
      is drawn.
- [ ] Hover W1 at 56 deg, a stroke reversal (the actuator speed passes zero
      between 56 and 57 deg): the share reads "Force share: - (hurting)".
      The other reversal, between 236 and 237 deg, shows no dash: no sample
      falls inside the 1 % speed band there, so W1's share jumps from about
      +45 kN to about -20 kN instead.
- [ ] The label's push/pull word follows the sign of the Actuator Force plot
      at the cursor: positive = push (extension), negative = pull
      (retraction). From 57 to 89 deg, for example, it reads pull.
- [ ] Drag W1 elsewhere on the rocker: its arrow is gold until the sweep is
      recomputed (a moment later), then green, red or gray again. Ctrl+Z
      puts it back at the tip.

### 4. Plots

The force passes through infinity at the stroke reversals (about 227 kN at
56 deg), so those spikes set the y range of the force plots. Scroll to zoom
in; double-click resets the view.

- [ ] **Actuator Power**: shaded bands cover exactly the angles where the
      red statics curve is below zero (91-269 deg). The legend entry
      "Braking" hides and shows them. The tab tooltip mentions the bands.
- [ ] **Actuator Force**: the same bands. The tab tooltip says "Positive =
      extension". Enter a **Rated Force**: the lines are labelled
      "Rated (push)" and "Rated (pull)".
- [ ] At 0, 180 and 360 deg the parallelogram's links are collinear (change
      points) and the velocity is not unique, so one sample there can glitch
      in every channel: the 0 and 180 deg samples dip, and at 360 deg the
      plots show a one-sample braking band, a pull force and mixed line
      colours. That is a solver artifact of this sample, not a payload bug.
      (The canvas at 360 deg reads the 0 deg sample.)
- [ ] **Weight Breakdown**, Force share: the legend lists "coupler (link)",
      "crank (link)", "rocker (link)", "Robot torso (W2)", "W1", "Other
      loads" and "Total". Other loads stays at 0 (gravity is the only load),
      and Total lies on the Actuator Force statics curve. Each weight's line
      is red where that weight rises, green where it falls, and gray where
      it moves sideways (90 and 270 deg).
- [ ] The weight lines and Other loads have a gap at 56-57 deg (force shares
      are blank near stroke reversal), while Total spikes there. At 236-237
      deg the weight lines spike and change sign without a gap (see
      section 3).
- [ ] Near 200 deg W1's line, and the rocker's own, is green (helping) while
      its force share is positive (W1 about +1.0 kN): the retracting
      actuator pushes harder to hold the load back (see "Key physics
      statement").
- [ ] **Power share**: each weight's line is below zero where it is green
      and above zero where it is red, with no gaps; Total lies on the
      Actuator Power statics curve.
- [ ] At three angles, hover the lines: the weight shares plus Other loads
      add up to Total.

### 5. Physical intuition

Read the shares at 45 deg in the Force share view.

- [ ] Set W1's Position to X 1000 mm, Y 0 mm, halfway from the tip (X 0) to
      the rocker pivot (X 2000 mm): its share halves, from about +5.5 kN to
      about +2.7 kN. A rocker point's speed is proportional to its distance
      from the pivot, in the same direction.
- [ ] Drag W2 anywhere on the coupler: its share does not change. The
      parallelogram coupler translates, so every coupler point has the same
      velocity.
- [ ] Set W1's mass to 100 kg: its shares double; the other weights' lines
      do not move.

### 6. Modes, driver and persistence

- [ ] In the property panel, set the actuator's **F** to 0 (sizing mode),
      then back to 50 N: the Actuator Force and Actuator Power plots, the
      Weight Breakdown Total and the actuator label do not change (since
      BL-026 they all show the required force).
- [ ] Open **Mounting Angle** in the input panel and set 30 deg: at 90 deg
      the weights are no longer gray, because gravity now has a component
      along their sideways motion; they turn gray near 60 and 240 deg
      instead. Set it back to 0.
- [ ] Turn on View > **Nathan Mode**: the green, red and gray lines and
      arrows stay distinguishable by brightness. Turn it off.
- [ ] Right-click the rocker's ground pivot joint (J4) and pick **Set as
      Driver**: the stock sample comes back without the weights (the known
      reset in the caution above). One Ctrl+Z restores W1 and W2 with their
      ids, names and masses, and J1 as the driver.
- [ ] Save and reopen: natively File > **Save As...**, then File > **Open
      JSON...**; in the browser File > **Download JSON...**, then File >
      **Recent Mechanisms**. Then use File > **Share via URL** and open the
      copied link in the browser. Each time, the weights keep their ids,
      names and masses, and each link's Mass in the Link Editor still reads
      1 kg (no double counting).
- [ ] Load **4-Bar Crank-Rocker**, which has no actuator: the Weight
      Breakdown shows driver torque shares ("Driver Torque Share (N·m)") for
      crank, coupler and rocker, and the Actuator Force, Actuator Speed and
      Actuator Power tabs are disabled.
