# Update Tracker

Meaningful changes to the project that future sessions should know about.
Reverse chronological (newest at top).

---

## 2026-04-23 — Ground-pivot discoverability + DXF geometry error fix

**What:**
- Right-click context menu on a joint or attachment point now shows an
  inline amber tip ("Needs a ground pivot — use the + Ground tool...")
  directly under the disabled "Set as Driver" button whenever the
  mechanism has zero grounded revolute joints. Previously the hint was
  only visible by hovering the disabled button, which new users missed.
  Implemented in `gui/canvas/context_menu.rs` for both `show_joint_menu`
  and `show_attachment_menu`.
- DXF "Add Geometry to Link" now emits an accurate error toast when the
  chosen target body doesn't exist in the blueprint (e.g. synthetic
  compound-expansion bodies like `force_0_cyl` / `force_0_rod` which
  live on the `Mechanism` but not in `state.blueprint.bodies`).
  Previously the blueprint write silently no-op'd and the status toast
  falsely reported success. Fix in
  `gui/dxf_import.rs::convert_selected_to_rigid_geometry`.

**Why:** A user building a press mechanism from scratch couldn't find
their way to adding a driver — the mechanism had no ground pivots yet,
and the only hint was a hover tooltip on a disabled menu item. The
inline tip directs the user at the + Ground tool immediately.

Separately, while diagnosing the same report, the DXF geometry flow was
found to claim success even when the target body wasn't user-editable.
Surfacing the real error keeps the user from chasing a ghost.

**Test results:** `cargo check --bin linkage-gui` clean. Context-menu
change is render-only; DXF change is a defensive early-return that
triggers only when the blueprint lacks the target body.

---

## 2026-04-23 — Suppress blank console window on Windows release builds

**What:** Added
`#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]`
to the top of `linkage-sim-rs/src/bin/linkage_gui.rs`. In release builds
the resulting `linkage-gui.exe` is linked with the Windows GUI subsystem,
so launching it from the LinkageSuite launcher (or from Explorer) no
longer spawns an empty black console alongside the egui window.

**Why:** Rust's default bin subsystem on Windows is CONSOLE. Without the
attribute, Windows allocates a console for every GUI-only binary, which
appears as a blank terminal next to the simulator. Debug builds
intentionally keep the console so `cargo run`, `env_logger::init()`, and
panic backtraces still print to stdout.

**Test results:** `cargo check --bin linkage-gui` clean. No behavior
change to the library or WASM bin (`linkage_web.rs` is web-only and
doesn't need the attribute).

---

## 2026-04-15 — Crank angle label + canvas indicator

**What:**
- Added a small grey subtitle above the Crank Angle slider reading
  `Driver: <driven> relative to <partner> (0° = <ref> +X)`. Pulls body
  names from `mech.driver_body_pair()`. Makes explicit that the slider
  value is the driver constraint's f(t) = θⱼ − θᵢ.
- Added `draw_crank_angle_indicator` in `canvas/rendering/mod.rs`:
  renders an orange arc at the driver's revolute pivot spanning from
  the partner body's +X reference to the driver body's current
  orientation, with a perpendicular tick at the zero mark and a
  midpoint label showing the angle in the current display units.
  Skips silently when the driver isn't revolute or the connecting
  joint can't be identified.

**Why:** The slider label "Crank Angle" didn't say which body rotated
relative to what, or where 0° pointed. Now both the input panel and
the canvas answer that question at a glance.

**Test results:** 568 lib tests pass (no new tests — overlay is
render-only); native + wasm32 `cargo check` clean.

---

## 2026-04-15 — Flip-branch button + slider crash fix

**What:**
- New `AppState::flip_assembly_branch()` method reflects non-ground
  non-driver body poses across the line joining the two farthest-apart
  ground pivots (fallback: world x-axis when < 2 pivots exist), then
  re-solves at the current driver angle. On solver convergence it
  updates `q`, `last_good_q`, marks the sweep dirty, and recomputes
  forces. Status toast reports whether the flip actually changed the
  assembly or landed on the same branch.
- New `Flip Branch` button in the Crank Angle section (next to the
  Loop/Once toggle). Hover text describes the debug use-case.
- Fixed panic in `input_panel.rs`: the sweep-range slider's
  `f64::clamp(slider_min, slider_max)` panicked when the user typed a
  max below the current min mid-edit. Slider bounds are now
  defensively sorted; sweep computation still reads the raw values
  (commit-time enforcement already guarantees `max >= min`).

**Why:** Adjusting the sweep range can occasionally kick the solver
onto the other assembly branch (common with parallelogram-style or
change-point mechanisms). The Flip Branch button gives the user a
one-click way to jump back. The slider crash was a regression from
removing the auto-swap in the previous commit.

**Test results:** 568 lib tests pass (two new:
`flip_assembly_branch_lands_on_alternate_config`,
`flip_assembly_branch_noops_without_mechanism`); native + wasm32
`cargo check` clean.

---

## 2026-04-15 — Sweep range can cross the 0/360 seam

**What:**
- Sweep range min/max DragValue clamps relaxed from `0..=360` to
  `0..=720`. Users can now sweep ranges that cross the cycle seam
  (e.g. `200° → 365°`) which previously was impossible.
- `compute_sweep_data` rewritten so range-enabled sweeps iterate
  literally `min..=max` in 1° steps (e.g. 200..=365 = 166 samples with
  display angles 200, 201, …, 365). Range-disabled sweeps still cover
  0..=360 (361 samples) unchanged.
- Solver receives the display angle in radians directly — it doesn't
  require `mod 2π`, so "angle 365°" produces the same physical q as
  "angle 5°" while preserving a contiguous, monotonic plot X-axis.
- `SweepData::active_range` is now always `None` (the sweep IS the
  range). Plots render the active curve solid with no faded-context
  overlay. The field is retained to keep the plot render path stable.
- `q_at_zero` only updates when the sweep visits angle 0 (start ≈ 0).
  Custom-range sweeps leave the caller-supplied `q_at_zero` untouched
  so subsequent full sweeps re-seed from the correct angle-0 q.
- Commit-time enforcement: if user leaves max < min, max snaps up to
  min. Cleared the blueprint_ops swap-if-inverted fallback.

**Why:** User's press workflow has a stroke that spans the mechanism's
0°/360° transition. Without wrap support the "working range" of many
press linkages can't be expressed as a single contiguous sweep.

**Test results:** 566 lib tests pass (two old toggle-stability tests
updated; one new test `sweep_range_can_wrap_past_360` added). Native +
wasm32 `cargo check` clean.

**Breaking changes:** The length-361 invariant on `SweepData` vectors
no longer holds for range-enabled sweeps. Channel-length alignment
(all channels share the same length within a sweep) IS still enforced.
`02-system.yaml invariants_to_protect` updated.

---

## 2026-04-15 — DXF Add Geometry uses a link-picker popup

**What:**
- Renamed the DXF sidebar button "→ Add Geometry to Selected Link" to
  "→ Add Geometry to Link". Clicking it no longer requires pre-selecting
  a body; instead it opens a modal popup listing every non-ground body.
- Clicking a body name in the popup applies immediately (no Apply button)
  and closes the dialog. Escape or the close button cancels.
- New state fields `show_dxf_geometry_target_dialog` +
  `dxf_geometry_pending_indices` in `AppState`. The DXF entity indices
  are snapshotted when the popup opens, so later overlay edits cannot
  desync the target.
- New `DxfAction::OpenRigidGeometryTargetDialog` variant replaces the
  previous `ConvertSelectedToRigidGeometry` direct dispatch.
- `convert_selected_to_rigid_geometry` signature changed to take an
  explicit `target_body: String` (auto-resolution from `link_editor_body`
  / `state.selected` removed; the popup supplies it instead).
- New `draw_geometry_target_dialog(ctx, state)` rendered from
  `LinkageApp::update` beside the other dialog windows.

**Why:** Users had to remember to click a link on the canvas before
clicking the button; otherwise they got a toast asking them to do so.
The popup makes the flow explicit — every click of the button opens a
picker — eliminating the "nothing selected" error path entirely.

**Test results:** 565 lib tests pass; native + wasm32 `cargo check` clean.

**Breaking changes:** None at the API level. The button label changed
from "Add Geometry to Selected Link" to "Add Geometry to Link".

---

## 2026-04-12 — Modularize gui/mod.rs and gui/sweep.rs

**What:** Four refactorings, zero behavior change:

1. **Menu bar extraction:** `gui/mod.rs` (1925 lines) → extracted File/Edit/Help/View/Image
   menus + sample gallery into `gui/menu_bar.rs` (703 lines). `mod.rs` dropped to 1147 lines.

2. **Sweep submodule:** `gui/sweep.rs` (1399 lines) → `gui/sweep/` directory:
   - `mod.rs` (1003 lines): SweepData, compute_sweep_data, push_nan_row helper
   - `motion_profile.rs` (317 lines): trapezoidal velocity profile + tests
   - `fourbar.rs` (110 lines): 4-bar linkage detection

3. **Theme DRY fix:** Extracted `gui/theme.rs` (83 lines) with `cad_dark_visuals()`,
   `apply_nathan_mode()`, `restore_normal_visuals()`. Eliminates duplicated color
   constants between `LinkageApp::new()` and `restore_normal_visuals()`.

4. **push_nan_row helper:** Replaces 55 lines of channel-by-channel NaN pushes in
   the sweep solver-failure branch with a single function call.

**Test results:** 565 lib/integration tests pass. Pre-existing linear_driver doctest
still fails (not related).

**Breaking changes:** None. All public APIs re-exported from new `mod.rs` files.

---

## 2026-04-10 — Code quality: mutate_and_rebuild helper + Plotly extraction

**What:**
- Added `AppState::mutate_and_rebuild(|s| ...)` in `gui/state/undo_ops.rs`.
  Encodes the "push_undo + mutate + rebuild" invariant documented in
  `04-memory.yaml` as a single helper method. Skipping the helper is still
  possible but now there's a canonical way to get the pattern right.
- Converted all 12 mutation helpers in `gui/state/entity_crud.rs` to use it
  (update_ground_pivot_position, nudge_body, nudge_joint,
  add_attachment_point_to_body, remove_attachment_point, add_ground_pivot,
  add_body_with_points, remove_body, add_revolute_joint, add_prismatic_joint,
  add_fixed_joint, remove_joint). `blueprint_ops.rs` (10 sites) and
  `driver_ops.rs` (5 sites) still use the raw pattern — migrate later.
- Added `add_plotly_line_chart()` and `add_plotly_multi_chart()` helpers in
  `gui/export/report.rs`. Converted the 3 simple single-series chart sites
  (torque, actuator force, transmission angle with 40°/90° shapes) to use
  the helper. The multi-series plots (joint reactions, energy, coupler
  traces) still have inline loops — migrate later if patterns converge.

**Why:** Two of the "top 5 code quality issues" from the 2026-04-10 audit.
The undo helper enforces an invariant that future AI sessions are explicitly
told to protect; the Plotly helper removes ~60 lines of duplicated boilerplate.

Context: the audit also flagged "19 panics in force element setters" and a
"1475-line impl block in forces/elements/mod.rs" — both were false alarms.
The panics were all in `#[cfg(test)]` assertion code, and the 1620-line
`forces/elements/mod.rs` is almost entirely tests (production code is in
sub-modules totaling ~1087 lines). See 04-memory.yaml lessons_learned.

---

## 2026-04-10 — Modularize rendering and plot_panel

**What:** Split two of the largest GUI files:

- `src/gui/canvas/rendering.rs` (2233 lines) → `src/gui/canvas/rendering/` directory
  - `mod.rs` (1236 lines): render_mechanism, render_overlays, tooltips, grid, background image
  - `primitives.rs` (508 lines): spring/damper/arrow/arc/marker/alignment-guide drawing helpers
  - `force_render.rs` (526 lines): force element visualization + load path heat map

- `src/gui/plot_panel.rs` (1770 lines) → `src/gui/plot_panel/` directory
  - `mod.rs` (539 lines): PlotTab enum, dispatcher, shared helpers (colors, outlier filter, toggle markers, series-with-range)
  - `mechanics.rs` (324 lines): body angles, transmission, mechanical advantage, joint reactions
  - `dynamics.rs` (275 lines): driver torque, inverse dynamics, energy
  - `actuator.rs` (390 lines): force, speed, power
  - `coupler.rs` (289 lines): trace, velocity, acceleration, output force

**Why:** Both files mixed many distinct responsibilities. `rendering.rs` conflated
drawing primitives, force visualization, and mechanism render pass into one file.
`plot_panel.rs` had 13 independent plot functions that shared nothing except
helpers. The splits preserve behavior exactly (zero logic changes) but make the
files easier to navigate and reason about.

**Test results:** 644 tests pass, WASM + native builds clean, pre-existing
linear_driver doctest still fails (not related).

**Breaking changes:** None. All public APIs re-exported from the new `mod.rs`
files so external callers are unchanged.

---

## 2026-04-10 — Non-Grashof sweep support + two-pass statics

**What:**
- Sweep no longer breaks at unreachable angles. Pushes NaN across all data
  channels and continues, so non-Grashof mechanisms that only oscillate
  produce usable plots for the reachable range.
- Two-pass statics solve in sweep for actuator-driven mechanisms:
  pass 1 with force=0 → read driver torque → compute F_actuator via power
  balance → pass 2 with F_actuator injected into Q → reactions reflect the
  actuator load path.

**Why:** User's Custom 6-Bar press has a non-Grashof linkage (only oscillates)
and is actuator-driven. The previous sweep stopped at the first unreachable
angle, and joint reactions didn't show the actuator's load path.

---

## 2026-04-10 — DXF import overhaul

**What:**
- Drag-and-drop DXF import (works on web + native)
- Interactive DXF entity selection
- Conversion buttons: → Links, → Multi-joint Body, → Add Geometry to Selected
  Link, → Ground Pivots, → Linear Actuator, → Delete Selected
- Auto-create ground pivots when LinearActuator endpoint has no nearby body point
- All conversions are additive (preserve existing mechanism, push undo, rebuild)
- "Add Geometry to Selected Link" uses the Link Editor's pattern: sets body.geometry
  directly instead of creating a new body

**Why:** Imports SolidWorks sketches into the simulator with fast selection-based
workflow. Rigid geometry attachment matches the standard Link Editor behavior.

---

## 2026-04-10 — UX polish

**What:**
- `+Ground` tool snaps to unconnected link endpoints and creates revolute joint
- Joint creation handles coincident pivots correctly (excludes first-clicked point)
- Canvas joint labels show actual joint IDs (J1, J2, ...) instead of R1/R2 auto
- Driver speed editor added to input panel (RPM display, rad/s readout)
- Plot legends moved from RightTop to LeftTop to avoid blocking data
- Computed actuator force displayed on canvas when stored force=0 (sizing mode)
- Force zone creation no longer auto-creates BodyGeometry
- Sweep range editor defers recompute until drag_stopped/lost_focus to avoid
  crashes from transient inverted ranges
- `(full)` legend tooltip explains faded dashed vs solid curve when sweep range
  is enabled

**Why:** Various discoverability, correctness, and robustness fixes from user
feedback during the DXF press-mechanism workflow.

---

## Earlier history

See `docs/FEATURES.md` and `docs/history/` for the full project history including
the Python→Rust port (411→644 tests), Phase 5 GUI, sample mechanisms, export
formats, and Phase 6.8 UX overhaul.
