# Update Tracker

Meaningful changes to the project that future sessions should know about.
Reverse chronological (newest at top).

---

## 2026-04-24 — Sweep range stored in display frame (follow-up)

**What:** The initial crank-angle-offset patch shifted the Crank Angle
slider to display frame but kept `sweep_angle_min_deg` /
`sweep_angle_max_deg` as body-frame values, then added/subtracted the
offset at the UI edges. That produced a jarring slider range (e.g.
`41°..61°` for a DXF-imported crank with α=41°) and sweep-range
DragValues that couldn't accept `0` as min.

Semantic cutover: `sweep_angle_min_deg` and `sweep_angle_max_deg` are
now stored in DISPLAY frame directly.

- Slider range: clean `[0, 360]` when sweep range is off; exactly the
  user's `[min, max]` when on. No more offset-shifted ranges.
- Sweep-range DragValues accept `0..=720` display degrees, matching
  the Crank Angle slider they sit below.
- `compute_sweep` subtracts the offset to produce body-frame
  `(min, max)` for `compute_sweep_data`.
- `step_animation_revolute` subtracts the offset for the ping-pong
  bounds when sweep range is on.
- `has_toggles_in_range` (health panel) subtracts the offset for the
  body-frame comparison.

**Schema note:** `_sweep_angle_min` / `_sweep_angle_max` in share URLs
are now display-frame (matches the `_animation_speed_deg_per_sec`
semantic). Old URLs load with body-frame values that get interpreted
as display — the sweep range visually shifts by α on first load. This
is a one-time migration side-effect; subsequent saves are clean.

**Test results:** 575 lib tests pass unchanged.

---

## 2026-04-24 — Crank angle display matches visible bar orientation

**What:**
- New `AppState::driver_display_offset` (rad), recomputed on every
  `rebuild` / `load_sample` / `load_from_json_str` / `reassign_driver`.
  It equals the angle from the driver body's grounded-pivot local
  coord to its farthest other attachment point, i.e. the local A→B
  direction. Zero for sample-built mechanisms (A at (0,0), B at
  (len, 0)); nonzero for DXF imports whose sketch wasn't drawn along
  +X. (`src/gui/state/mod.rs` + rebuild / load / driver paths.)
- Every user-visible angle surface now shows `body_frame_θ + offset`:
  Crank Angle slider (`src/gui/input_panel.rs`), sweep-range min/max
  DragValues (same file), plot X-axes + current-angle cursor +
  click-to-scrub (`src/gui/plot_panel/mod.rs`), and the canvas
  crank-angle indicator arc (`src/gui/canvas/rendering/mod.rs`). The
  solver, sweep data, JSON schema, and share URLs keep body-frame θ
  as the source of truth — the offset is applied only at the display
  boundary.
- Plot dispatcher clones `SweepData` per frame with shifted
  `angles_deg` and `toggle_angles`; every individual plot function
  remained unchanged. The clone is cheap relative to the redraw and
  keeps the 20+ plot call sites consistent without modification.

**Why:** User reported the Crank Angle slider showed near 0° while the
visible link sat ~15° above horizontal (and 53° looked almost vertical
instead of at 90°). Diagnosis: the slider showed the driver body's
internal θ. For sample-built bodies that equals the visible bar
direction (because local A→B lies on +X), but DXF imports preserve
world coords as local coords so the A→B direction has an arbitrary
offset α. Display now reads `θ + α` so the slider tracks the
orientation the user actually sees on canvas.

**Test results:** 575 lib tests pass (+2 new:
`driver_display_offset_zero_for_sample_builders`,
`driver_display_offset_picks_up_rotated_crank`).

---

## 2026-04-24 — Force zone application point: visualize + draggable override

**What:**
- `ForceZoneElement` gets a new optional field
  `body_local_app_point: Option<[f64; 2]>`. When `None`, the solver
  uses the overlap centroid as before. When `Some`, the force applies
  at the pinned body-local point, letting the user model contact at a
  specific location rather than an area-averaged centroid.
  (`src/forces/elements/element_types.rs`,
  `src/forces/elements/evaluation.rs`.)
- Canvas now renders a crosshair + "F" label at the active application
  point — faint yellow for the auto-centroid, brighter orange for a
  pinned override (labelled "F (locked)"). Uses a shared helper
  `force_zone_app_point_world` in
  `src/gui/canvas/rendering/force_render.rs`.
- Left-drag on the marker in Select mode enters an app-point drag. A
  ghost crosshair follows the pointer; on release the pointer world
  position is converted to the target body's local frame and written
  to the override, committing via `update_force_element`. Escape
  cancels the drag mid-gesture.
- Force editor for `ForceZone` grows an "Application point" section
  with a `Lock to body-local point` checkbox, Local X/Y DragValues
  (mm), and a `Reset to auto (overlap centroid)` button.

**Why:** User wanted to see where the force is actually applied and
pin it to the real contact point. Overlap centroid is correct for
distributed contact but wrong for a specific contact pad — the
override lets the user enforce the physical contact location without
abandoning the zone's "trigger on contact" semantics.

**Schema compatibility:** `serde(default, skip_serializing_if =
"Option::is_none")` — old files load cleanly with
`body_local_app_point = None`, and files saved without an override
don't grow the field.

**Test results:** 573 lib tests pass unchanged. JSON round-trip and
sample-builders updated to include the new field explicitly.

---

## 2026-04-24 — Delete body also strips referencing forces + diagnostics relocated

**What:**
- `AppState::remove_body` (`src/gui/state/entity_crud.rs`) now also
  retains-out force elements whose `attached_body_ids()` reference the
  deleted body. Previously a LinearActuator (or any spring/damper/etc.)
  attached to the deleted link stayed in the blueprint, and the next
  rebuild tried to resolve its attachment points on a missing body,
  freezing the GUI. New regression test
  `remove_body_cascades_to_force_elements` loads the
  ParallelogramActuator sample (which has a crank-attached linear
  actuator), removes the crank, and asserts no force still references
  it.
- Moved the debug `dt / fps / step` readout from the top toolbar
  (where it wrapped off-screen on narrow viewports) to the bottom
  status bar. Still gated on the View → Debug Overlay toggle, but
  always visible next to the angle / torque / DOF readouts.

**Why:** User reported "Cannot delete link 1, it's the last link but
has an actuator attached at the end joint, tool freezes, delete
actuator with it if needed" — which is the cascade gap described
above. Also reported they couldn't see the diagnostics line with
Debug Overlay on, so it's been relocated to a guaranteed-visible
spot.

**Test results:** 573 lib tests pass (+1 new cascade test).

---

## 2026-04-24 — Context menu Set-Driver submenu + frame-time diagnostics

**What:**
- Right-click context menus on joints and attachment points now offer
  a `Set Driver to \u{2026}` submenu whenever *any* grounded revolute
  exists in the mechanism, not only when the clicked element is itself
  the grounded one. Previously users right-clicked a non-grounded
  joint, saw `Set as Driver` disabled with only a hover tooltip, and
  reported it as "grayed out, can't add a driver". The submenu lists
  every grounded revolute with the current driver labeled `(current)`,
  so the action is always reachable from any joint.
  (`src/gui/canvas/context_menu.rs`.)
- Top toolbar shows a compact `dt=16.7ms  fps=60  step=0.083°/frame`
  diagnostics line next to the speed slider when the Debug Overlay is
  enabled (View menu). Intended to narrow down the user's "no motion
  under 15 °/s" report on the WASM build — if the reported dt or step
  is out of whack, the issue is in the browser/egui frame pipeline, not
  the animation math.

**Why:** User said "I couldn't add a driver because the selection was
grayed out before". The previous UX required right-clicking the right
*specific* element; the submenu decouples discovery from precision.
Frame-time readout gives us an observable signal to diagnose the
low-speed animation-freeze complaint without guesswork.

**Test results:** 572 lib tests pass unchanged.

---

## 2026-04-24 — Link Editor delete button + Driver panel picker

**What:**
- `Delete` button in the Link Editor header row
  (`src/gui/property_panel/mod.rs`). Shown next to the body-label edit
  field for any non-ground body. Red text, hover text
  "Delete this link and all joints connected to it". Wired through a
  new `PendingPropertyEdit::DeleteBody` variant that calls
  `state.remove_body` and clears `selected` + `link_editor_body`.
- Driver section now shows a `ComboBox` picker when the mechanism has
  no driver but at least one grounded revolute joint exists
  (`src/gui/input_panel.rs` — new `draw_no_driver_picker`). Selecting
  a joint writes to `pending_driver_reassignment`, which the frame
  pipeline already consumes to call `reassign_driver`.
- When NO grounded revolute joint exists, the Driver section shows the
  same amber "Needs a ground pivot…" hint the right-click context
  menu uses, so the discoverability path is consistent across
  surfaces.

**Why:** User reported "We have no way to delete links…also can't add
driver even with ground points". Decoding their shared URL confirmed
two grounded revolute joints (J5, J6) and an empty drivers map — the
data was correct, but the only path to set a driver was the canvas
right-click menu, which the user wasn't discovering. A dedicated
picker in the Driver panel makes the action visible where the user
expects it. Same reasoning for Delete: the canvas right-click menu
had it, but the Link Editor (where users already look to edit a link)
didn't.

**Test results:** 572 lib tests pass unchanged.

---

## 2026-04-23 — Driver omega floor + disabled + Body ribbon button

**What:**
- New constant `MIN_DRIVER_OMEGA_ABS = 0.01` (rad/s ≈ 0.1 RPM) and helper
  `clamp_driver_omega` in `src/core/driver.rs`. Applied at three
  boundaries:
  1. `set_constant_speed_driver` (GUI Speed DragValue) — typing 0 RPM
     now snaps to 0.1 RPM instead of freezing the mechanism.
  2. `load_mechanism_unbuilt_from_json` — legacy files written with
     omega=0 are rescued at load time.
  3. `solve_at_angle` — belt-and-braces zero guard mirroring the other
     callsites (blueprint_ops, file_io, undo_ops).
- Top-ribbon "+ Body" button is now disabled with hover text redirecting
  users to the Link Editor ("To add a rigid body, use the Link Editor
  in the property panel..."). The Link Editor exposes mass, inertia,
  mount/coupler points, and geometry together, which is the desired
  canonical flow.

**Why:** User reported the animation would hang at low speeds. Root
cause was the driver closure capturing omega=0: `f(t) = theta_0 + 0*t`
freezes the constraint at `theta_0`, and `solve_at_angle` separately
divides by `driver_omega`. Every entry point that writes omega now
clamps above the floor. 0.01 rad/s ≈ 1 revolution per 10 minutes, so
anything slower is kinematically indistinguishable from a static
mechanism; users who truly want "paused" should use Play/Pause.

Separately, "+ Body" in the top ribbon was a shortcut whose flow
diverged from the Link Editor's full-featured body creation. Disabling
it pushes users to the canonical path.

**Test results:** 572 lib tests pass (+2 new:
`step_animation_advances_at_slow_speed` and
`set_constant_speed_driver_rejects_tiny_omega`).

---

## 2026-04-23 — Slower animation + share URL embeds speed and crank limits

**What:**
- Top-toolbar animation speed slider lower bound dropped from 10.0 to
  0.5 °/s (`src/gui/mod.rs:483`). The log scale previously compressed
  10–30 °/s into a sliver of screen, making the slider feel like it
  refused to slow the animation past ~15 °/s. Users can now genuinely
  crawl the kinematics for debugging.
- Share URLs now embed `_animation_speed_deg_per_sec` and the sweep
  range (`_sweep_range_enabled`, `_sweep_angle_min/_max`)
  unconditionally — previously the sweep range was only written when
  the toggle was on, so a recipient would land in 0..360 animation
  bounds even when the sender had a narrower working range configured.
  `driver_omega` rides along in the standard `drivers` JSON map (no
  dedicated override needed).
- `load_from_json_str` restores all four fields on the receiving side
  (`src/gui/state/file_io.rs`).

**Why:** User reported "anything less than 15 deg/s doesn't move" and
"URL sharing doesn't embed speed or crank-angle limits". First was a
slider floor, not an animation-math bug; second was a real
serialization gap.

**Test results:** Two new unit tests
(`share_url_round_trips_speed_and_crank_limits`,
`share_url_embeds_crank_limits_even_when_range_disabled`) verify the
round-trip including the "toggle off, limits still embedded" case. 568
lib tests pass.

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
