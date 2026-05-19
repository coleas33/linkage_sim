# Update Tracker

Meaningful changes to the project that future sessions should know about.
Reverse chronological (newest at top).

---

## 2026-05-18 — FBD validation test for 4-bar + LinearActuator at θ_2=π/3

**What:**
- New regression test `solver::reactions::tests::fbd_validates_pass2_reactions_at_60deg`.
  Builds the 9×9 Cartesian equilibrium system by hand (force balance +
  moment balance per body) for the crank-rocker + sizing-mode
  LinearActuator config, solves it with `nalgebra::full_piv_lu`, and
  asserts the simulator's pass-2 reactions and back-solved F_act match
  the independent FBD solution within 1e-4 N at pose θ_2 = π/3. Also
  checks the pass-2 driver torque lambda is ~0 (the point of pass-2).
- Caught and corrected a sign-convention bug in the FBD derivation
  during the write-up: `evaluate_linear_actuator` applies
  `force_on_a = −F_act·u` (with `u = (p_b − p_a)/|·|`), so positive
  F_act = extension. My first cut had `+F_act·u` on body_a, which is
  the compression convention. Same physical force; flipped sign on
  F_act in the answer. Now matches.

**Why this matters:**
- The J3-mismatch bug (commit `492fbb7`) shipped because no test
  asserted absolute correctness — only solver self-consistency
  (residual small, lambdas finite). This is the first absolute-truth
  anchor: the simulator's `Φ_qᵀ λ = −Q` matches an independently
  written Cartesian equilibrium at a verified pose. Catches future
  sign-flipped Jacobians, pass-2 double-counts, swapped actuator
  endpoints, mass-to-Q mapping errors, etc.

**Counts:** 699 → 700 lib tests passing.

**Spec status:**
`docs/superpowers/specs/2026-05-18-reaction-force-validation-design.md`
option A is now partially in (1 of 4–6 poses). Extending to additional
poses (top-dead-center, near-singular, etc.) is incremental; the
infrastructure is in place.

---

## 2026-05-18 — focus shift + validation-design spec

**What:**
- Updated `docs/ai/01-meta.yaml` `active_focus` to reaction-force
  validation for 4-bars with a LinearActuator (was: Custom 6-Bar press
  workflow). The 6-bar workflow is no longer the user's stated focus.
- New `docs/superpowers/specs/2026-05-18-reaction-force-validation-design.md`
  documenting three approaches — closed-form FBD (A), internal cross-
  checks (B), external tool comparison (C) — plus a Hybrid path
  (A+B). Recommendation: A first, then B.
- New entry in `docs/ai/04-memory.yaml` `open_questions` tracking the
  reference-choice decision that blocks the implementation plan.

**Why:**
- The J3-mismatch bug (commit `492fbb7`) shipped because existing
  reaction tests only assert solver self-consistency, not absolute
  correctness. Validation work is the natural next focus to lock in
  trust before further hardware-deployment work.

---

## 2026-05-12 — repo hygiene: gitignore browser-automation scratch

**What:**
- Added `.playwright-mcp/`, `.superpowers/`, and `/linkage-*.png` to the
  repo-root `.gitignore`. These artifacts are produced by Playwright MCP
  sessions and ad-hoc Claude Code work poking at the deployed web app;
  they should never have been tracked but had been showing as `??`
  untracked entries across sessions.

**Why:**
- Lowers the noise floor on `git status` so real working-tree changes
  stand out. No behavioural impact.

**Known noise still in the tree (not addressed here):** the
`docs/chebyshev_lambda/*.png` files drift by 1–2 bytes when the raster
GIF tests run — committed PNG metadata is non-deterministic across
runs. Worth investigating which test is regenerating them and either
making it deterministic or routing it to a scratch path.

---

## 2026-05-01 — docs + UX: trajectory mode walkthrough

**What:**
- New `docs/guides/TRAJECTORY_MODE.md` walkthrough — concept, when-to-use,
  5-minute quick-start on the canonical 4-Bar Crank-Rocker, plot reading,
  failure modes (R/S/B/N glyphs), profile selection, correctness
  verification recipe, advanced features (comparison overlay, motion
  ribbon, playback, exports), and a troubleshooting table.
- Linked from `README.md` doc index.
- One-line UX fix in `gui/mod.rs`: switching `Sweep mode → Trajectory`
  now calls `mark_sweep_dirty()` so the trajectory plot populates
  immediately instead of leaving the user staring at "No trajectory
  data yet" until they hunt for the Compute button.
- New regression test `compute_trajectory_world_x_target_tracks_target`:
  drives WorldX of crank.B along a constant-speed profile from 0.005
  to 0.009 m, asserts achieved tracks target within 1e-6 and the per-
  sample pose snapshot's θ_crank satisfies the inverse identity
  `x = 0.01·cos(θ)`. Complements the existing trivial-Angle test by
  exercising the Newton outer loop on a non-linear u→h relationship.

**Why:** User feedback was "not obvious how to use it." Three things
needed to land together: (1) a real walkthrough that explains *why*
trajectory mode exists vs. forward sweep, (2) an in-app cue that
trajectory data appears immediately on mode switch, and (3) a
regression test that pins correctness on a non-trivial target so
future refactors don't silently break the inverse solve.

**Test results:** 669 lib tests pass (was 668 + 1 new). Native + WASM
both compile clean.

---

## 2026-05-01 — fix(build): exhaustive cfg gating for `--no-default-features`

**What:** Inline audit of the recent web-pass commits caught two
configurations where cfg gates were not exhaustive across
`(feature = "native") × (target_arch = "wasm32")`:

1. `gui/export/download.rs::download_bytes_impl` had only the
   `feature = "native"` and `target_arch = "wasm32"` arms; a
   `--no-default-features` native build hit an unresolved symbol from
   the public caller. Added a third fallback arm returning
   `DownloadOutcome::Failed`.
2. `gui/state/templates.rs` gated the `dirs::home_dir()`-using
   functions (templates_dir, persist_*, remove_persisted_*,
   load_*_native) by `not(target_arch = "wasm32")` but `dirs` is in
   the `native` feature flag. Same shape of bug — `--no-default-features`
   native build had unresolved `dirs` crate. Switched to
   `feature = "native"` and added no-op fallbacks for the persist /
   remove methods so the ungated entry points (save_as_template,
   delete_template, save_as_custom_sample, delete_custom_sample) still
   link. The `load_saved_*` dispatchers got tri-state cfgs returning
   empty `Vec` for the unsupported configuration.

**Why:** The `chrono_now` panic on wasm32 (e7006ec) was the same shape
of bug — un-gated APIs that don't compile in every supported
configuration. Audit widened the verification to all four build
configurations: native default, native --no-default-features, wasm
default, wasm + raster. All now compile clean.

**Test results:** 668 lib tests pass.

---

## 2026-04-30 — feat(web): recent mechanisms list (Pass 4)

**What:** Added a 5-entry localStorage-backed ring buffer that captures
mechanism JSON snapshots when the user drag-and-drops a JSON or clicks
"Download JSON…". Surfaced as a "Recent Mechanisms" submenu under
File (web only) — clicking an entry restores it without re-dragging.

Storage key: `linkage_recent_mechanisms` (single localStorage entry,
JSON array of `(name, json_text, unix_secs)` tuples). New helpers
`AppState::wasm_push_recent_mechanism` and
`AppState::wasm_load_recent_mechanisms`. `format_relative_time` in
menu_bar.rs renders the timestamp as "5 min ago" / "2 hr ago" /
"3 days ago".

**Why:** Native has a Recent Files list; web didn't, so a user closing
the tab and reopening lost their session even with autosave (autosave
holds only the most recent state, not history). The ring buffer is the
web-equivalent of native's recent paths — paths don't translate to web,
so we store full JSON snapshots instead. Bounded to 5 entries to stay
under typical localStorage quotas.

**Discovery during work:** Most other "missing" persistence features
(Saved Templates, Custom Samples, Autosave) already had full
localStorage backings on WASM and just weren't documented as such.
Verified Save as Template / Manage Templates menu items show on web
and autosave-recovery prompt fires on startup. The web persistence
story is now complete.

**Test results:** 668 lib tests pass (no new tests; `js_sys::Date` is
WASM-only so the helpers can't be exercised under `cargo test`).
Native + WASM both compile clean.

---

## 2026-04-30 — feat(web): drag-and-drop imports for JSON + CSV (Pass 3)

**What:** Extended the existing drag-and-drop handler in
`gui/mod.rs::update()` to dispatch on file extension:
- `.json` → `state.load_from_json_str` (mechanism)
- `.csv` → `parse_keyframes_csv_str` (keyframes; only in Trajectory mode)
- `.dxf` → DXF overlay (existing)
- everything else → background image (existing)

Refactored `parse_keyframes_csv` (path-based, native-only) to layer over
`parse_keyframes_csv_str` (cross-platform). +3 tests covering simple
parse, comment/blank skipping, and empty-input rejection.

Updated the trajectory panel: hide the native-only Import CSV button on
web and show a "Drag a .csv onto the canvas" hint instead. Updated the
File menu's web-only drag-and-drop hint to enumerate all four supported
formats.

**Why:** After Passes 1 and 2 made every export work on web, the
export-import asymmetry was the next gap. Drag-and-drop already
worked for DXF and images on web (egui `RawInput::dropped_files`
delivers bytes on both targets), so extending it to JSON and CSV was
~30 lines of dispatcher code rather than building a full `<input
type=file>` + FileReader integration. An explicit "Open JSON…" menu
button on web is deferred — the channel-based async file-picker
architecture is documented but not built.

**Test results:** 668 lib tests pass (was 665 + 3 new). Native + WASM
both compile clean.

**Limitations:** No explicit menu file picker on web — drag-and-drop
only. Recent files / saved templates / autosave on web all still
require localStorage wiring (separate effort).

---

## 2026-04-30 — feat(web): browser-side raster + CSV downloads (Pass 2A + 2B)

**Pass 2A — CSV:** Refactored `gui/export/csv.rs` to factor out
cross-platform `write_sweep_csv` and `write_coupler_csv` writers
taking `&mut impl Write`. Public surface gained
`generate_sweep_csv_string` / `generate_coupler_csv_string` companions
to the existing path-writing functions. Both Sweep CSV and Coupler CSV
menu_bar buttons now route through `download::download_text` and work
on web. +4 tests pin file/String parity. 665 lib tests pass.

**Pass 2B — PNG/GIF:** Added a new `raster` feature flag in
`Cargo.toml` (the `native` feature now depends on it). Refactored
`gui/export/raster.rs` from `cfg(feature = "native")` to
`cfg(feature = "raster")` and added `generate_mechanism_png_bytes` /
`generate_mechanism_gif_bytes` companions. Extended `download.rs`
with a `download_bytes` helper for binary content. Menu_bar PNG/GIF
buttons gated by `raster` (so they appear on any build with the
feature, including web).

**Deployment:** Updated `scripts/build_web.sh` and
`.github/workflows/deploy-web.yml` to add `--features raster` to the
WASM build. Bundle cost is ~+2.5 MB pre-wasm-bindgen
(9.25 MB → 11.73 MB; post-processing brings the final shipped
`_bg.wasm` to a smaller fraction).

**Why:** After Pass 1 lit up text exports on web, raster was the last
gap blocking parity with the native export menu. Resvg + gif compile
to wasm32 cleanly — verified via `cargo check --no-default-features
--features raster --target wasm32-unknown-unknown` — so the only
question was bundle size. 27% growth is acceptable given the user
gets PNG and GIF in exchange.

---

## 2026-04-30 — feat(web): browser-side downloads for text exports (Pass 1)

**What:** Web build can now save files. Added `gui/export/download.rs` with
`download_text(filename, mime, contents, filter)` that branches by target:
native opens an `rfd` save dialog and writes bytes; web wraps the contents
in a `Blob`, mints an `ObjectURL`, programmatically clicks a transient
`<a download>` anchor, then revokes the URL. Same call site, both
platforms. Returns a `DownloadOutcome { Saved | Cancelled | Failed }`
that maps to the AppState status / error-panel UX via `apply_download_outcome`.

Refactored five export buttons in `gui/menu_bar.rs` to route through the
helper and dropped their `#[cfg(feature = "native")]` gate:
- Export firmware (JSON) — uses new `build_firmware_json_string`
- Export SVG — `generate_svg_string`
- Export labeled schematic SVG — `generate_schematic_svg`
- Export DXF — `generate_dxf_string`
- Generate Report (HTML) — `generate_html_report`

Added a web-only "Download JSON…" button as the Save analog (the web
build has no concept of a "current open file" so Save / Save As stay
native-only). The associated `serialize_to_json_string` was promoted to
`pub(crate)`. Made the `*_string` generators platform-independent by
dropping the over-broad `#[cfg(feature = "native")]` from `report.rs`,
`schematic.rs`, `svg.rs`, and `dxf.rs` — the gating was historical, the
function bodies have always been pure-Rust.

Pass 2 (Sweep CSV / Coupler CSV / PNG / GIF / DXF import / Image import
/ keyframe CSV import) remains native-only. The CSV generators write
directly to a `Path`; PNG/GIF use native-only crates (resvg, gif).

**Why:** A full month of trajectory work landed exports for the native
build only. On the deployed web app, every export menu item was either
absent or non-functional — so the user could compute a beautiful
trajectory plus pose snapshots and have no way to extract the data.
Pass 1 lights up the highest-value text exports without touching the
raster-export code path.

**Test results:** 661 lib tests pass (no new tests; the helper is
JS-side glue). WASM target compiles clean. Native dev build compiles
clean.

**Cargo.toml:** Expanded `web-sys` features with `Blob`,
`BlobPropertyBag`, `HtmlAnchorElement`, `HtmlElement`, `Element`,
`Document` for the blob-download path.

---

## 2026-04-29 — feat(traj): trajectory diff overlay (T3)

**What:** Added `AppState::trajectory_comparison: Option<SweepData>` plus
"Save as comparison" / "Clear" buttons in the trajectory panel's
Visualization section. When a comparison is saved, the trajectory plot
overlays its target/achieved/u traces in faded dotted style alongside the
live trajectory so the user can A/B two profiles, two targets, or two
severity settings on the same axes. The overlay carries the snapshot's own
time axis (not the live one) so different durations render honestly.
Implemented in `gui/plot_panel/trajectory.rs::comparison_traces_from`.

**Why:** Users iterating on trajectory design need a quick A/B without
exporting CSVs and replotting externally. Trapezoidal vs S-curve at the
same target, or the same profile pointed at two different targets, are
the canonical "did I improve it?" workflows.

**Test results:** 661 lib tests pass (657 baseline + 4 new). New tests
cover snapshot duration vs live-duration time axis, non-trajectory mode
rejection, partial-snapshot rejection, and degenerate sample-count
rejection.

---

## 2026-04-29 — feat(traj): per-sample pose snapshot in CSV/JSON exports (T4)

**What:** Added `pose_body_order: Option<Vec<String>>` and
`pose_snapshots: Option<Vec<Vec<[f64; 3]>>>` to `SweepData`.
`compute_trajectory` populates them with each non-ground body's
`(x, y, θ_rad)` at every sample. Trajectory CSV appends
`q_x_<body>_m, q_y_<body>_m, q_theta_<body>_rad` columns per body after
the v1 columns; firmware JSON adds an optional `pose: { body_id: [x, y, θ] }`
map per sample. Both are skipped when the snapshot is absent so the v1
schema stays valid for older / non-trajectory data.

**Why:** Downstream tools (firmware test harnesses, motion previewers,
report generators) need to replay the full pose without re-running the
inverse solve. Per-body columns are additive, so existing v1 consumers
keep working.

**Test results:** 657 lib tests pass (653 baseline + 4 new). New tests
cover CSV present/absent paths, JSON present/absent paths, plus an
extension to `compute_trajectory_populates_all_trajectory_fields`
asserting θ_crank in the pose snapshot tracks the Angle target on the
directly-driven body.

---

## 2026-04-29 — feat(eq): auto-generated labeled mechanism schematic (SVG)

**What:** Added `gui/export/schematic.rs::generate_schematic_svg(mech, q)` that
produces a self-contained SVG of the loaded mechanism with constraint labels,
matching the visual style of figures 2 & 3 in
`docs/superpowers/specs/2026-04-29-linkage-equations-reference.md`. The figure
includes a title strip with DOF, ground hatching, bars (one per non-ground
body), joint clusters (coincident joints render as one circle with merged
labels), italic body labels at each body's centroid, a red driver indicator
(curved arrow for `RevoluteDriver`, cylinder/piston line for `LinearDriver`),
and a sidebar legend listing constraint rows grouped by joint kind plus the
driver row's symbolic form (delegated to `gui::eq_rendering::symbolic_form`).
Wired into `gui/menu_bar.rs` as File → "Export labeled schematic (SVG)...".

**Why:** Part of the "linkage equations reference" body of work (E-series). The
goal is to give users a publication-quality labeled figure of their own
mechanism for design docs and lab reports — generated programmatically from
the loaded mechanism rather than hand-coded per-figure as in the spec.

**Test results:** 647 lib tests pass (642 baseline + 5 new). New tests cover
the 4-bar revolute-driver path, the `ParallelogramActuator` sample, an inline
`LinearDriver`-driven 4-bar (linear-driver indicator branch), an empty
mechanism (returns Err), and structural well-formedness of the emitted SVG.

---

## 2026-04-30 — feat(traj): analytic acceleration inverse via per-variant Hessian

**What:** Replaced the placeholder zero-Hessian arms in `ControlTarget::hessian`
with closed-form Hessians for `WorldX`, `WorldY`, `Projection`, and `Distance`
(Angle is linear in q, retains zero Hessian). Added
`solver/inverse_kinematics/derivatives.rs::inverse_acceleration_analytic` —
computes `r''(u) = ∇²g(dq/du, dq/du) + ∇g·(d²q/du²)` directly via the
analytic Hessian plus a "kinematic-only γ" assembly (existing `assemble_gamma`
with `dq/du` in place of `q̇` and the `Φ_tt` driver-row term zeroed). Eliminates
the two extra forward solves per sample that `inverse_acceleration_fd` requires.

**Why:** Resolves the `04-memory.yaml` open question on analytic acceleration.
While FD was sufficient for v1 trajectory analysis, the analytic path is more
accurate (no FD truncation error) and faster (one linear solve vs two forward
solves). Available for callers to opt into; `compute_trajectory` continues to
use FD by default per spec §8.3 recommendation.

**Test results:** 635 lib tests pass (628 baseline + 4 Hessian FD-check tests +
2 analytic-vs-FD agreement tests + 1 constant-velocity test). Analytic and FD
agree to ~1e-2 (FD's 1e-3 truncation error dominates). Constant-velocity test
hits machine-zero with analytic vs `1e-3` with FD — confirming the accuracy gain.

---

## 2026-04-29 — feat(traj): JSON firmware export adapter

**What:** Added a `FirmwareAdapter` trait and a concrete `JsonAdapter` under
`linkage-sim-rs/src/gui/export/firmware/`. The adapter consumes a
post-`compute_trajectory` `SweepData` plus the active `ControlTarget`,
`Trajectory`, and input-parameter unit label and emits a versioned JSON
envelope (`schema_version = "linkage-traj-firmware-v1"`) with metadata
(`target_kind`, `target_units`, `input_units`, `n_samples`,
`duration_seconds`) and an array of per-sample objects (`t`, `target`,
`achieved`, `residual`, `u`, `u_dot`, `u_ddot`, `f_actuator_n`, `status`).

NaN/non-finite actuator forces serialize as JSON `null` via
`Option::filter(|x| x.is_finite())` so consumers see a clean sentinel
instead of an invalid `NaN` token. Solver failure modes carry diagnostic
detail in the `status` string (`"Reachability: target=… clamped=…"`,
`"Singularity: dg_du=…"`, `"BranchJump: norm=…"`,
`"NonConvergent: iter=… residual=…"`).

A new "Export firmware (JSON)..." entry appears in the File menu only when
the cached sweep is in `SweepMode::Trajectory`. The wrapper picks
`input_units` from `state.driver_kind` (revolute → `"rad"`, linear →
`"m"`, none → `"rad"` fallback) and surfaces failures via `error_log` /
`show_error_panel`; success sets a transient status message.

**Why:** Resolves the v3 firmware adapter open question in
`04-memory.yaml`. JSON is the v1 universal target — easy to consume from
any language, schema-versioned for forward compatibility. G-code, Aerotech
AeroBasic, Beckhoff TwinCAT NC PTP, and Galil DMC are planned ~150-LoC
follow-ups behind the same `FirmwareAdapter` trait, implemented on demand
once a hardware target is selected.

**Touched files:**
- `linkage-sim-rs/src/gui/export/firmware/mod.rs` — new: `FirmwareAdapter` trait + re-exports
- `linkage-sim-rs/src/gui/export/firmware/json.rs` — new: `JsonAdapter`, `FirmwareJsonEnvelope`, `FirmwareJsonSample`, schema constant + 3 unit tests
- `linkage-sim-rs/src/gui/export/mod.rs` — `pub mod firmware;`
- `linkage-sim-rs/src/gui/menu_bar.rs` — "Export firmware (JSON)..." menu entry + `export_firmware_json` helper
- `docs/ai/04-memory.yaml` — closed v3 firmware adapter open question
- `docs/ai/05-update-tracker.md` — this entry

**Test results:** 628 lib tests pass (625 baseline + 3 firmware adapter
tests: round-trip parse, NaN-actuator-force filter, empty-data rejection).
Build clean for `--features native`. Pre-existing `dirs`-crate errors in
non-native builds are unrelated to this change.

---

## 2026-04-29 — feat(traj): keyframe trajectory input + CSV import

**What:** Added a `Trajectory` enum with two variants — `Profile` (existing
analytic motion profile: ConstantSpeed / Trapezoidal / SCurve) and
`KeyframeTable` (user-defined `(t, h)` waypoints with linear interpolation) —
and migrated `SweepMode::Trajectory.profile: TrajectoryProfile` to
`SweepMode::Trajectory.trajectory: Trajectory`. Added a CSV import path so
`(t, h)` series authored externally (e.g. from a hardware capture or
spreadsheet) can be loaded as keyframes.

`KeyframeTrajectory::evaluate(t)` returns `(h, ḣ, ḧ)` where:
  - `h` is the linear interpolation between the bracketing waypoints.
  - `ḣ` is the segment slope `(h₁ - h₀) / (t₁ - t₀)`.
  - `ḧ` is `0` everywhere (piecewise-linear has zero curvature on each
    segment; the velocity step at waypoint boundaries is small at typical
    trajectory sample rates and the inverse-dynamics `ü` solve falls back to
    its FD path anyway).
Outside the table (`t < first` or `t > last`), evaluate clamps to the edge
value with `ḣ = 0`.

The trajectory_panel UI gains a top-level "Trajectory kind" dropdown
(Profile / Keyframes); switching kinds substitutes a sensible default for
the new variant. The Keyframes editor shows a row-per-waypoint table with
inline `t`/`h` drag-edits, an X button to remove rows (minimum 2 retained),
"+ Add waypoint", "Sort by t", and (native only) "Import CSV...". The
inline preview plot renders the current `h(t)` curve and overlays
waypoint markers when in Keyframes mode.

CSV format: 2 columns `t_seconds, target_value`. The first row is treated
as a header if its first cell fails to parse as a float. Blank lines and
`#`-prefixed lines are skipped. Waypoints are sorted by `t` ascending.

**Why:** Closes the long-standing "Trajectory v2: keyframe / waypoint
trajectory input" and "Trajectory v2: CSV-table trajectory import" open
questions in `04-memory.yaml`. Enables non-canonical trajectory shapes
(replays of recorded motion, hand-authored bring-up sequences, profiles
that don't fit ConstantSpeed / Trapezoidal / SCurve).

**Touched files:**
- `linkage-sim-rs/src/gui/state/types.rs` — `Trajectory` enum, `KeyframeTrajectory` struct + 3 unit tests
- `linkage-sim-rs/src/gui/state/mod.rs` — re-export `Trajectory`, `KeyframeTrajectory`
- `linkage-sim-rs/src/gui/sweep/mod.rs` — `SweepMode::Trajectory.profile` → `.trajectory`; `compute_trajectory` parameter type → `&Trajectory`; integration test updated
- `linkage-sim-rs/src/gui/state/blueprint_ops.rs` — destructure / call site updates
- `linkage-sim-rs/src/gui/mod.rs` — toolbar default constructor wraps in `Trajectory::Profile(...)`
- `linkage-sim-rs/src/gui/plot_panel/trajectory.rs` — duration / evaluate via enum
- `linkage-sim-rs/src/gui/export/csv.rs` — duration via enum + test fixture
- `linkage-sim-rs/src/gui/trajectory_panel/profile_input.rs` — full rewrite: two-level kind dropdown + analytic editor + keyframe editor + CSV import + parse_keyframes_csv test

**Test results:** 625 lib tests pass (621 baseline + 3 keyframe interp tests
+ 1 CSV parser test). Build clean for both `--features native` and the
default (WASM-friendly) profile; no new warnings beyond the pre-existing
baseline.

**Concerns / known limitations:**
- WASM CSV import is not wired up (rfd's WASM async API would need a
  separate code path; defer until a browser user actually asks for it).
  The button is `#[cfg(feature = "native")]`-gated, so the WASM build
  simply doesn't render it.
- `ḧ` reports `0` for keyframe trajectories. The inverse-dynamics path
  uses FD anyway for non-Angle targets, so this is a non-issue in
  practice; the `h_ddot` value flows into `compute_trajectory` only as a
  hint for the FD step direction (which the solver then refines).
- Switching trajectory kinds in the UI discards the prior variant's
  parameters (analytic ↔ keyframes). This is intentional — there's no
  obvious mapping between "ConstantSpeed start=0 end=1 dur=1" and a
  3-waypoint keyframe table — and matches the established switch-resets
  pattern of the shape dropdown.

---

## 2026-04-29 — feat(traj): add SCurve (jerk-limited) motion profile

**What:** Added a third variant `MotionProfile::SCurve { jerk_fraction: f64 }`
to the `MotionProfile` enum. v1 implementation uses the pure quintic
ease-in-out polynomial `s(τ) = τ³(10 − 15τ + 6τ²)` for trajectory-mode
position, so velocity and acceleration are both zero at `t=0` and
`t=duration` (the defining property of a jerk-limited profile). The
`jerk_fraction` field is reserved for a future full 7-segment formal
version and is currently unused.

In **trajectory mode**, `TrajectoryProfile::evaluate` dispatches to a new
`scurve_value(t, duration, start, end)` helper next to the existing
`trapezoidal_value` in `gui/state/types.rs`. The profile-editor UI in
`gui/trajectory_panel/profile_input.rs` exposes "SCurve" alongside
"ConstantSpeed" and "Trapezoidal" in the shape dropdown.

In **driver-side sweep mode**, SCurve falls back to `ConstantSpeed`
behaviour (sets `profile_torques / profile_omega / profile_alpha` to
`None`). The jerk-limited evaluation is meaningful only for trajectory
mode where the `(h, ḣ, ḧ)` tuple feeds the inverse-kinematics solve.
The driver-side selector in `gui/input_panel.rs` does not surface
SCurve as a selectable option but maps it to the "Constant Speed" label
when the trajectory-mode UI has switched it on.

**Why:** Closes the long-standing "Trajectory v2: S-curve" open question
in `04-memory.yaml`. Real actuator hardware uses jerk-limited profiles
to avoid mechanical shocks at start/stop; v1 quintic captures the
endpoint property without the implementation cost of the full
7-segment piecewise jerk profile.

**Touched files:**
- `linkage-sim-rs/src/gui/state/mod.rs` — new `SCurve { jerk_fraction }` variant
- `linkage-sim-rs/src/gui/state/types.rs` — `scurve_value` + dispatch + 2 tests
- `linkage-sim-rs/src/gui/trajectory_panel/profile_input.rs` — dropdown + ctor
- `linkage-sim-rs/src/gui/sweep/motion_profile.rs` — driver-side fallback to const-speed
- `linkage-sim-rs/src/gui/input_panel.rs` — exhaustive match (SCurve labelled "Constant Speed")

**Test results:** 621 lib tests pass (619 baseline + 2 new SCurve tests:
`trajectory_profile_scurve_endpoints_match_target`,
`trajectory_profile_scurve_integrates_to_span`). Build clean (no new
warnings beyond the pre-existing baseline).

---

## 2026-04-29 — R1b refactor: collapse driver scalars into DriverKind enum payload

**What:** Removed the dual-purpose `driver_omega`, `driver_theta_0`, and
`driver_stroke` scalar fields from `AppState`. Their values now live as
payload on the `DriverKind` enum:

- `DriverKind::Revolute { angle, omega, theta_0 }`
- `DriverKind::Linear { stroke, velocity, length_0 }`
- `DriverKind::None` (unit, unchanged)

Added `driver_omega() / driver_theta_0() / driver_stroke()` getters and
`set_driver_omega() / set_driver_theta_0() / set_driver_stroke()` setters
on `AppState` so flat call sites stay readable; dispatch points in
`step_animation`, `compute_sweep`, and trajectory mode pattern-match the
variant directly. Setters on the wrong variant (e.g. setting stroke on
Revolute) are silent no-ops.

Equality / pattern-match call sites updated from
`state.driver_kind == DriverKind::Linear` to
`matches!(state.driver_kind, DriverKind::Linear { .. })`.
`Eq` derive removed from `DriverKind` (f64 payload has no `Eq`); `Copy`
preserved (all-f64 payload is Copy).

**Why:** The dual-purpose scalars were the implicit-overloading remnant
flagged as R1b in `04-memory.yaml/open_questions`. Collapsing onto the
enum makes per-driver-kind semantics type-checked rather than
documented-by-convention.

**Touched call sites:** 14 files, +283 / -155 lines.

**Test results:** 619 lib tests pass (no count change vs.
`c905f28`/baseline). Release lib + `linkage-gui` binary both compile
clean.

**Subtleties handled:**
- `load_sample()`, `reassign_driver()`, `set_constant_speed_driver()`,
  `convert_actuator_to_linear_driver()`, `restore_snapshot()`, and
  blueprint `rebuild()` reorder writes to construct the kind variant
  fully before any setter call — setters are no-ops on the wrong
  variant, so write-then-transition was the failure mode to guard
  against.
- The `MechanismSnapshot` DTO in `gui/undo.rs` retains its dual-purpose
  scalar shape (it's a serialization format, not state) — `take_snapshot`
  / `restore_snapshot` translate between the DTO and the variant
  payload via the new accessors and a fresh blueprint-driven kind
  reconstruction on restore.
- Linear-driver stroke preservation across rebuilds: the old code
  preserved a finite `driver_stroke` field across `rebuild()`. The new
  blueprint_ops pattern-matches the prior `DriverKind::Linear { stroke,
  .. }` and reuses it; resets to `length_0` on Linear→Linear with NaN
  stroke or any non-Linear→Linear transition.

---

## 2026-04-30 — Trajectory-mode position control: Stage 4 (polish + CSV + docs) — feature shipped

**What:**
- CSV export wired for trajectory mode in `gui/export/csv.rs::write_trajectory_csv`. v1 superset column layout: `t_seconds, target_value, achieved_value, residual, u, u_dot, u_ddot, F_actuator_N, driver_torque_Nm, status`. Status text format like `Reachability:0.0920->0.0870`, `Singularity:1.5e-7`, etc.
- "From joint" helper in target picker (`gui/trajectory_panel/target_picker.rs::draw_point_input_with_joint_helper`) — populates body-local point coords from existing joints on the chosen body.
- Failure-band hover tooltips: rendered as a `ui.collapsing("⚠ N failure(s)")` summary below the trajectory plot, with one line per failed sample showing `t={:.3}s: <status payload>`. Cleaner than per-band tooltips per egui_plot 0.33 limitations.
- `docs/FEATURES.md` — user-facing entry under "Trajectory-mode position control" with use case, how-to, failure handling, math reference cross-link.
- `docs/ai/04-memory.yaml` — 9 deferred items added to `open_questions`: SCurve, click-to-set canvas, keyframe input, CSV-table import, firmware adapter, analytic acceleration, sweep_mode persistence, linear-driver dispatch, u_range frame conversion.

**Why:** Closes the trajectory-mode feature. v1 ships with full forward + inverse pipeline, GUI activation, CSV export, plot rendering, and click-to-scrub. Polish items deferred to follow-up tasks per the spec.

**Test results:** 619 lib tests pass (+2 from Stage 3 baseline; +9 cumulative across Stage 4: 2 CSV + 7 from earlier Stage 4 substages).

**Commits in Stage 4:**
- 0ce97af — CSV export
- 28028eb — From-joint helper
- bab31f0 — Failure-band tooltip summary
- f325484 — User-facing FEATURES.md + deferred items in 04-memory.yaml
- (plus this final wrap entry)

**Known limitations / follow-ups (tracked in 04-memory.yaml):**
- sweep_mode/trajectory_severity not persisted through save/load
- Linear-driver trajectory dispatch is TODO
- u_range display-frame vs body-frame conversion (subtle bug; needs verification on a DXF-imported mechanism)
- F_actuator_N column always NaN until compute_trajectory wires actuator force computation
- v2 features (SCurve, keyframes, CSV import) deferred until users ask

---

## 2026-04-30 — Trajectory-mode position control: Stage 3 (UI surface) shipped

**What:** Stage 3 surfaces Stage 2's `compute_trajectory` backend through the
GUI. New `Trajectory` option in the SweepMode dropdown switches the input
panel to a trajectory-specific UI (target picker + profile editor + severity
toggle), and the plot panel renders a time-axis view with click-to-scrub
that drives the canvas mechanism through the back-solved trajectory.

- **`gui/state/trajectory_ops.rs`** — new `AppState::solve_for_trajectory_target(target, h)`
  sibling to `solve_at_angle` / `solve_at_stroke`. Calls `solve_for_target`
  in `Severity::Analysis` and updates `self.q` on success, leaving state
  unchanged on failure. Used by click-to-scrub.
- **`gui/trajectory_panel/`** (3 files):
  - `mod.rs` — top-level dispatcher. Three `CollapsingHeader` sections:
    Target observable → Profile → Solve options (severity radio toggle:
    Analysis vs Strict).
  - `target_picker.rs` — `ControlTarget` variant picker (5 variants:
    `Angle`, `WorldX`, `WorldY`, `Distance`, `Projection`) with the
    relevant body / point / axis fields per variant + live `g(q)` readout
    on the current pose so the user can see where they are before
    choosing the target value.
  - `profile_input.rs` — `TrajectoryProfile` editor: shape selector
    (currently trapezoidal), `start` / `end` / `duration` /
    `accel_frac` / `decel_frac` fields, `n_samples` slider, and an
    inline `h(t)` preview plot.
- **`SweepMode` dropdown extended** in `gui/input_panel.rs` — `Angle` /
  `Stroke` / `Trajectory` options. Selecting `Trajectory` constructs a
  default `SweepMode::Trajectory { target: Angle{driver}, profile:
  default, severity: Analysis, n_samples: 64 }` and routes the input
  panel to `trajectory_panel::draw` instead of the angle/stroke
  controls.
- **`AppState` new fields** (`gui/state/mod.rs`): `sweep_mode:
  SweepMode` (replaces ad-hoc `is_stroke_sweep` checks), `trajectory_severity:
  Severity`. Both default to `Angle` / `Analysis`.
- **`gui/state/blueprint_ops.rs::compute_sweep`** — branches on
  `state.sweep_mode`. `SweepMode::Trajectory` dispatches to the
  Stage 2 `compute_trajectory` for revolute drivers (the integration
  deferred from Stage 2). Linear-driver path is TODO and falls back
  to the existing `compute_sweep_data` so users still get a sweep
  curve.
- **`gui/plot_panel/trajectory.rs`** (new) — time-axis plot rendering.
  Main plot stacks target(t) (orange) + achieved(t) (cyan) on the
  primary axis; residual(t) and u(t) are stacked in separate panels.
  Failure-band overlays paint regions where
  `inverse_solve_status != Converged`. Click-to-scrub on the main
  plot calls `state.solve_for_trajectory_target(&target, h)` to
  drive the canvas mechanism to the clicked sample's pose.
- **`gui/plot_panel/mod.rs`** — early-return dispatch when
  `sweep_mode.is_trajectory()`: `trajectory::render(state, ui)`
  bypasses the forward-sweep tabs (the trajectory X-axis is time, not
  driver angle, so the existing tabs don't apply).
- **Minor refactor:** `empty_trajectory_sweep_data` promoted to
  `pub(crate)` so `compute_sweep` in blueprint_ops can construct an
  empty `SweepData` shell before calling `compute_trajectory` to fill
  it.

**Why:** Stages 1 and 2 shipped the math and the per-sample analysis
loop, but the trajectory-mode pipeline was unreachable from the UI.
Stage 3 wires it through: dropdown → input panel → compute → plot →
click-to-scrub. The user can now define a desired output trace, see
the back-solved actuator input alongside it, and scrub the canvas
mechanism through the trajectory by clicking the time-axis plot.

**Test count:** 617 lib tests pass (unchanged from Stage 2). UI is
integration-only — existing solver tests and `TrajectoryProfile` /
`SweepData` unit tests cover the math; manual GUI verification is
deferred to user.

**Open follow-ups:**
- Persist `sweep_mode` and `trajectory_severity` through save/load.
  Both fields are in-memory only at the moment, so reopening a
  document drops the trajectory configuration.
- Linear-driver dispatch in `compute_sweep`. Currently the
  `SweepMode::Trajectory` branch checks `driver_kind == Revolute`
  and falls through to the angle/stroke `compute_sweep_data` path
  for linear-driver mechanisms.
- Display-frame vs body-frame conversion of `u_range` in the
  trajectory dispatch. The forward-sweep path subtracts
  `driver_display_offset` to get body-frame θ; the trajectory
  dispatch currently does not. Confirm correct frame.
- Per-frame body velocity / acceleration readouts. Stage 1 noted
  that `solve_velocity` / `solve_acceleration` are not currently
  called per-frame; trajectory mode might benefit from them for
  diagnostics.

---

## 2026-04-30 — Trajectory-mode position control: Stage 2 (sweep extension) shipped

**What:** Stage 2 wires the Stage 1 inverse-kinematics solver into the
sweep pipeline. Adds a `compute_trajectory` sibling to `compute_sweep_data`
that, given a `TrajectoryProfile` describing the desired output trace
`h(t)`, back-solves the actuator input `u(t)` per sample and runs the
existing per-sample force/energy/reaction analyses unchanged.

- **`TrajectoryProfile` struct** (`gui/state/types.rs`) — composes the
  existing `MotionProfile` shape (trapezoidal velocity ramp) with absolute
  units (`start`, `end`, `duration`). Implements
  `evaluate(t) -> (h, h_dot, h_ddot)` and `sample_times(n)` for uniform
  sampling across `[0, duration]`.
- **`SweepData` trajectory time-series fields** (`gui/sweep/mod.rs`) —
  6 new `Option<Vec<f64>>` fields (`target_values`, `achieved_values`,
  `tracking_residual`, `u_values`, `u_dot_values`, `u_ddot_values`)
  plus `Option<Vec<InverseSolveStatus>>` for diagnostics. All carry
  `#[serde(default, skip_serializing_if = "Option::is_none")]` so existing
  saved snapshots load cleanly. SweepData / SweepMode / InverseSolveStatus
  gain Serialize/Deserialize derives.
- **`SweepMode::Trajectory { target, profile, severity, n_samples }`
  variant** joining `Angle` and `Stroke`. `is_trajectory()` accessor
  added; existing `is_stroke()` consumers in plot panels unchanged.
- **`compute_trajectory` per-sample loop** (`gui/sweep/mod.rs`) — for each
  sample time `t_k`: evaluates the profile, calls `solve_for_target` to
  back-solve `u_k`, computes `u_dot_k` / `u_ddot_k` via
  `inverse_velocity` / `inverse_acceleration_fd`, then invokes the same
  `solve_statics` / `solve_inverse_dynamics` /
  `compute_energy_state_mech` calls used by `compute_sweep_data`. The
  driver row of `Φ_t` and `γ` is overridden with the back-solved
  `u_dot_k` / `u_ddot_k` (two lines per sample) so the constant-speed
  constraint code in `core/` remains untouched. On
  `Severity::Analysis` failures the output is re-evaluated from the
  partial `q` so time-series channels remain length-matched.
- **Refactor fold-in:** `Mechanism::driver_row()` helper centralizes the
  `n_constraints() - 1` driver-last-row assumption (Task 1.10 review
  item, applied at the Φ_t / γ override site).
- **Test count:** +1 integration test
  (`compute_trajectory_populates_all_trajectory_fields`) verifying field
  population, target tracking, and `Converged` status across all samples.
  Total lib tests: 617 pass (was 614 after Stage 1; +3 across Stage 2
  tasks).
- **Stage 2 GUI activation: NONE.** `compute_trajectory` is reachable
  from code and tests but not from the UI. `compute_sweep_data` does
  not yet dispatch to it. Stage 3 introduces `state.sweep_mode` and
  wires `AppState::compute_sweep` to branch on the variant.

**Why:** Stage 1 shipped the inverse solver in isolation. Stage 2
threads it through the per-sample analysis pipeline so a full trajectory
produces the same plottable channels as a forward sweep (joint
reactions, driver torque, energies, etc.) — without ripping up the
constant-speed-parameterized constraint code. The Φ_t / γ driver-row
override is the minimal seam.

**Migration note:** No GUI changes shipped in Stage 2; users see no
difference yet. Stage 3 adds the trajectory-mode UI surface (target
picker, profile editor, severity toggle, mode switcher) and dispatches
`AppState::compute_sweep` to the right backend.

**Test results:** 617 lib tests pass (+1 new
`compute_trajectory_populates_all_trajectory_fields`).

---

## 2026-04-30 — Trajectory-mode position control: Stage 1 (solver layer) shipped

**What:** New `solver/inverse_kinematics/` subtree (6 files) implements the
inverse-kinematics solver for trajectory-mode position control. Given a
desired output observable `g(q)` and target value `h`, back-solves the
actuator input parameter `u` such that `g(q(u)) = h`, where `q(u)` is the
forward solution from the existing kinematics solvers.

- **Module layout** (`linkage-sim-rs/src/solver/inverse_kinematics/`):
  - `mod.rs` — public surface (re-exports of types + functions).
  - `severity.rs` — `Severity::{Strict, Analysis}` + `InverseSolveStatus`
    (5 variants: `Converged`, `Reachability`, `Singularity`, `BranchJump`,
    `NonConvergent`).
  - `control_target.rs` — `ControlTarget` enum with 5 variants
    (`Angle`, `WorldX`, `WorldY`, `Distance`, `Projection`); constructors
    validate against ground.
  - `solver.rs` — `workspace_probe` + `solve_for_target` outer Newton loop;
    full failure-mode detection (reachability, singularity, branch-jump,
    non-convergence).
  - `derivatives.rs` — `inverse_velocity` (closed-form) +
    `inverse_acceleration_fd` (finite difference); shared `compute_dq_du`
    helper.
  - `test_helpers.rs` — `#[cfg(test)]` 4-bar fixture + warm-start q.
- **Severity model** — `Severity::Strict` returns `Err(LinkageError::...)`
  on any failure; `Severity::Analysis` returns
  `Ok(InverseSolveResult { status: <failure variant>, ... })` so the GUI
  can render failed samples in red rather than aborting the trajectory.
  Same Newton/FD math runs in both modes; only the return type differs
  (`classify_or_fail` helper centralizes the dispatch).
- **New `LinkageError` trajectory variants** (`error.rs`):
  `TrajectoryUnreachable`, `TrajectorySingular`, `TrajectoryBranchJump`,
  `TrajectoryNonConvergent`. `From<InverseSolveStatus> for LinkageError`
  bridges the two enums.
- **Refactor in `core/`** — new `Mechanism::driver_row()` helper centralizes
  the `n_constraints() - 1` assumption that the driver is the last
  constraint row.
- **Test count:** ~32 new tests in inverse_kinematics module + 1 new error
  test. Total lib tests: 614 pass (was 613, +1 singularity-detection test
  added by Task 1.14).
- **Stage 1 GUI integration:** NONE. Solver works in isolation. Stage 2
  starts with sweep extension (`gui/sweep/mod.rs::compute_trajectory`),
  which will override Φ_t / γ on the driver row using back-solved
  `u_dot` / `u_ddot`.

**Why:** User wants position control via trajectory — specify a desired
output (e.g. coupler-point world Y vs time) and back-solve the required
crank/stroke trajectory. Existing solvers go forward (driver → output);
this layer goes inverse (target output → required driver input).

**Migration note:** No GUI changes shipped in Stage 1; users see no
difference yet. Stage 2 will wire the solver into `compute_trajectory`,
and Stage 3 will add the trajectory-mode UI.

**Test results:** 614 lib tests pass (+1 new
`singularity_detection_near_extremum`). Pre-existing linear_driver
doctest still fails (not related). BranchJump test deferred to
stage-1.5 — engineering the failure mode requires constructing a 4-bar
near a toggle, which is a separate spec; TODO comment in place.

---

## 2026-04-24 — LinearDriver GUI feature: stroke-driven sweeps

**What:** Wired the dormant LinearDriver pipeline end-to-end so a user
can convert a `LinearActuator` force element into a kinematic
`LinearDriver` constraint and run stroke-driven analysis. Six
preparatory refactors plus the conversion UI:

- **R1 — `DriverKind` discriminant.** New enum on `AppState`
  (`None` / `Revolute` / `Linear`). Documents the per-variant meaning
  of the existing `driver_omega` / `driver_theta_0` / `driver_stroke`
  scalars (rad+rad/s vs m+m/s) and unblocks dispatch without ripping
  out 150+ usages of the four scalars. Follow-up R1b (deferred) can
  collapse them into payload on the variants.
- **R2 — `rebuild()` reads `linear_drivers`.** Sets `driver_kind =
  Linear` when the blueprint has any linear driver, populating
  velocity/length_0 and seeding `driver_stroke`. Falls back to
  Revolute if `bp.drivers` is non-empty, otherwise None.
  `load_from_json_str` applies the same detection.
- **R3 — `step_animation` dispatch.** Matches `driver_kind` and
  delegates to `step_animation_revolute` (existing) or new
  `step_animation_linear` (mirrors revolute but moves stroke in
  metres). The animation slider's "deg/s" doubles as "mm/s" in
  Linear mode for a consistent feel.
- **R4 — `SweepMode::Stroke`.** `compute_sweep_data` detects via
  `mech.n_linear_drivers()` and iterates 1 mm steps; default range
  is `length_0 ± 100 mm`. `compute_sweep` guard counts linear
  drivers; sweep-range field is interpreted as mm in Linear mode and
  converted to metres for the solver. Plot dispatcher's offset shift
  is a no-op for stroke sweeps; click-to-scrub branches between
  `solve_at_angle` and `solve_at_stroke`.
- **R5 — Driver-section UI dispatch.** `draw_input_panel` shows the
  Crank Angle section for Revolute/None, the new Actuator Stroke
  section for Linear (mm slider + mm sweep-range DragValues). The
  `sweep_angle_min/max_deg` field is repurposed for stroke (mm) when
  in Linear mode — only one driver mode is active at a time.
- **R6 — Display-offset gate.** `compute_driver_display_offset`
  returns 0 when `driver_kind != Revolute`; the angle-frame offset
  has no meaning for stroke values. Plot dispatcher's
  `sweep_in_display_frame` is also a no-op for stroke sweeps.

- **Conversion UI.** `LinearActuator` force editor gets a `Set as
  Linear Driver` button. `AppState::convert_actuator_to_linear_driver`
  removes any revolute drivers, strips the actuator force element,
  and adds a `LinearDriverJson` with `length_0 = current stroke` so
  the pose doesn't jump on takeover. Default velocity 10 mm/s.

**Why:** User wanted stroke-driven analysis for press mechanisms.
`LinearActuator` was a force element (apply force, find equilibrium),
which is the wrong abstraction for "sweep stroke and read out
required force". `LinearDriver` is the right one but had no GUI entry
point and dormant solver wiring.

**Test results:** 578 lib tests pass (+3 new:
`sweep_stroke_mode_for_linear_driver`,
`sweep_stroke_mode_with_explicit_range`,
`convert_actuator_to_linear_driver_switches_mode`).

**Migration note:** Existing files with revolute drivers continue to
work unchanged. Sample mechanisms (FourBar, ChebyshevLambdaActuator
with its LinearActuator force, etc.) all load in Revolute mode by
default; user opts into Linear mode via the Set as Linear Driver
button.

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
