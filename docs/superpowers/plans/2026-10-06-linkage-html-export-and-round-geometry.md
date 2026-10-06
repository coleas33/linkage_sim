# Linkage website: round shapes, the contact point, and an animated HTML export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Part A gives a link's visual geometry a circle shape and a force zone a third application mode, the shape's contact point, so a wheel's load acts (and is drawn) at the wheel's bottom at every pose. Part B adds File -> "Export animation (HTML)...", a self-contained animated page of the current mechanism like the user's press animation.

**Architecture:** Part A: `BodyGeometry` gains a `shape` (rectangle, the default and left out of files, or circle, its diameter in `width` with `height` equal) and three methods (`centre_world`, `outline_world`, `extreme_point_world`); every place that today builds the rectangle's world corners for a force zone (the force evaluation, the overlap ratio, the independent equilibrium check, the canvas marker) calls one new helper, `force_zone_application`, which returns the overlap and the application point by the zone's mode (overlap centroid, locked body point, or contact point = the shape's extreme point against the zone's force). The property panel switches shapes and adds them; the force editor picks the mode. Part B (design here, code written after Part A merges): the export precomputes every sweep sample in Rust (poses, outlines, force visuals, reactions, readouts) and embeds them as JSON in an HTML template with a small JS player.

**Tech Stack:** Rust 2024, egui 0.32 (headless tests), nalgebra, serde/serde_json; no new dependency. Playwright for the browser checks.

**Spec:** the user's requests: 2026-10-05 "Can the square instead be a circle whose bottom center point is located where the F is located at full extension of actuator. 5.2" diameter, centered as is"; "In the website why is the F locked shown as the centroid now?"; asked whether to add round shapes and a lowest-point contact to the website: "Yes add to 10:45 task"; and "Schedule task at 10:45am to update website to be able to export html animation that is this but for the current layout if possible" (the reference: `Lift_Linkage_Animation_2026-10-06.html`, the press animation checked against the website solver). Physics the user confirmed the same day: the wheel and the mass move rigidly with the link; the ground's vertical push acts at the wheel's lowest point, which slides round the rim relative to the link ("It still acts vertically, just on a different part of wheel relative to link").

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-shapes`, branch `linkage/round-geometry`, created by Task 0 from `main` at `f9df219` with LF line endings and a worktree-scoped `core.autocrlf=false`. Every command uses absolute paths into it. Nothing is pushed without the user's go.

**Process (the user's lean process, memory `project_magcoupling_lean_pacing`):** Part A is Tasks 0 to 4. Tasks 1 to 3 give the exact code, so implementers transcribe on `sonnet` with a `sonnet` review per task; Task 4 (docs and the browser check) also on `sonnet`; the whole-branch review after Task 4 on the session model. No pre-flight scan: every old block quotes `main` at `f9df219` (or the file as earlier tasks leave it). Part B's tasks get their exact code in this file after Part A merges (decision H-8), then run the same way.

## Decisions to confirm

**Confirmed (user, 2026-10-06): the recommended option on all of R-1 to R-7 and H-1 to H-8.** The plan implements the recommended option of each (code and docs cite the ids).

| Id | Question | Recommended (implemented here) | Alternatives | Tasks |
|---|---|---|---|---|
| R-1 | How a circle is stored | **A `shape` field on `BodyGeometry`:** `rectangle` (the default, left out of files, so every existing file keeps its bytes) or `circle`, whose diameter is `width`, with `height` written equal to it (a build before schema 1.2.0 ignores `shape` and draws the circle's bounding square) | (b) `BodyGeometry` as an enum of shapes (a cleaner type; a new file format and every use rewritten); (c) a separate optional circle field on the body | 1 |
| R-2 | How the user adds and switches shapes | **The property panel:** "Add rectangle" and "Add circle" buttons on a link without geometry (sized from the link: a rectangle as long as the link and a quarter as deep, a circle half the link's length, both centred on the link), and a Rectangle / Circle switch plus a Diameter slider on a link with geometry. The "Draw Geometry" and "Redraw" buttons go: their canvas tool is an unfinished stub that does nothing (BL-044) | (b) also finish the canvas draw tool (drag out a rectangle or a circle): a separate task; (c) circles from files only | 3 |
| R-3 | What "contact point" means | **The shape's extreme point against the zone's force:** where a surface pushing in the force's direction first meets the shape; for an upward force, the shape's lowest point (a wheel's bottom, straight below its hub at every pose); for a downward force, its top; a tilted force follows. A level rectangle edge gives the edge's midpoint | (b) the lowest point in the world frame whatever the force; (c) the lowest point along gravity (follows the mounting angle) | 2 |
| R-4 | How the force editor shows it | **A three-way choice, "Apply at: Overlap centre / Locked point / Contact point",** replacing the "Lock to body-local point" checkbox and the "Reset to auto" button. Choosing Locked keeps an earlier locked point, else seeds the shape's centre (today it seeds the body origin, which for a DXF-imported body is the world origin, far from the link). Choosing Contact keeps the locked point, so switching back restores it. Dragging the F marker switches to Locked at the drop point | (b) keep the checkbox and add a separate "at contact point" checkbox | 2, 3 |
| R-5 | The schema version | **1.2.0:** the loader checks only the major number, so files written now still open in older builds (which draw circles as squares and ignore the contact mode) | (b) stay 1.1.0 (the format change goes unmarked) | 1 |
| R-6 | Circle maths | **A 64-sided polygon for the overlap test and the overlap centroid; a true circle on the canvas; the exact centre plus radius for the contact point** | (b) an exact circle-rectangle intersection (more code for a binary on/off test) | 1, 2 |
| R-7 | What proves the contact point right | **A sample linkage, not the user's press:** the Parallelogram Press with a wheel on its coupler gives the same driver torque, sample for sample, with the force at the wheel's contact point as with it locked at the hub (a vertical force on the hub's vertical line). The repo is public, so the user's press model is not committed; its numbers (21,506 N at crank 28 deg, 5,776 N at 65 deg) are checked locally in Task 4 and live after the push | (b) commit the user's press model as a test fixture | 2, 4 |
| H-1 | Where the export lives | **File -> "Export animation (HTML)...",** after "Generate Report (HTML)..." | (b) also a button on the plot panel | B3 |
| H-2 | A mechanism without an actuator | **The driver torque takes the actuator force's place** in the readout and the chart | (b) no chart without an actuator | B1, B2 |
| H-3 | Samples and file size | **The sweep's own samples** (1 deg over the sweep range, 1 mm for a stroke drive; about 0.3 MB for the press over 28-65 deg) | (b) 0.25 deg (smoother, four times the file) | B1 |
| H-4 | Units | **The app's units (N, N m, and mm or m from Display Units) with lbf beside every force** (the app itself has no lbf; the user works in lbf) | (b) the app's units only; (c) a units switch inside the page | B1, B2 |
| H-5 | Samples with no solution | **Left out:** the slider covers solved samples only; the chart breaks there | (b) shown as a "no solution" frame | B1, B2 |
| H-6 | Self-contained | **One file, no network:** inline SVG and JS, no CDN (the existing HTML report loads Plotly and KaTeX from CDNs) | (b) reuse Plotly from its CDN for the chart | B2 |
| H-7 | Desktop app too | **Yes:** the web download and the desktop save dialog, through `export::download::download_text` like every export | (b) web only | B3 |
| H-8 | Shipping order | **Part A first:** built, reviewed, merged and (with the user's go) pushed; then Part B's code is written here against that `main`, built and pushed | (b) both parts on one branch, one push | all |

## Global Constraints

- Base: `main` at `f9df219` ("merge: BL-042 weights draw over force zones"). Worktree `C:/Users/Cole/source/repos/lsim-shapes`, branch `linkage/round-geometry`. Never modify `C:/Users/Cole/source/repos/linkage_simulation` itself beyond the `worktree add` of Task 0 Step 1.
- Existing files keep loading and keep their meaning: a geometry without `shape` is a rectangle; a force zone without `at_contact_point` keeps its locked point or centroid; files with only rectangles and no contact mode are written with the same keys as before (only `schema_version` changes, to `1.2.0`).
- The magcoupling crate is untouched: no file under `magcoupling-rs/` changes, and gates 4 to 12 stay green.
- No new dependency; both `Cargo.lock` files unchanged.
- Formatting: never run `cargo fmt` in `linkage-sim-rs` (not rustfmt-clean at `f9df219`); new code is written in rustfmt's layout.
- UI text is ASCII where a glyph could be missing from egui's default fonts: no diameter sign (write "diameter").
- Blocks quote the files with LF line endings, as git stores them. "Create `path`:" writes a new file with the block's text and a final newline. "In `path`, replace: ... with: ..." is one exact replacement whose old block occurs exactly once, as whole lines; "replace both occurrences" says so explicitly. Apply a step's blocks in order. If an old block is not found, stop and escalate; never improvise a match.
- Gate runs (Task 0 and every task before its commit): `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/scripts/gate.sh` must end `GATE PASS` with no `SKIP gate` line; then `git -C C:/Users/Cole/source/repos/lsim-shapes checkout -- docs/chebyshev_lambda` (the tests rewrite those PNGs; never commit them).
- Commits: one per task, subjects `feat(linkage): ...` (Task 4: `docs(linkage): ...`). Every commit message ends with exactly two lines: `Co-Authored-By: <the model writing the commit> <noreply@anthropic.com>`, then `Claude-Session: https://claude.ai/code/session_01FSw7N8oV7SUVKTDVrGKLdg`. Nothing is pushed.
- Docs change with code (repo rule): Task 4 updates `docs/reference/FORCE_ELEMENTS.md`, `docs/FEATURES.md`, `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md` and `backlog.yaml`, and commits this plan. Read `docs/ai/*.yaml` first (Task 0).
- Model tiers (CLAUDE.md section 5): Tasks 0 to 4 run on `sonnet` (`model: 'sonnet'`); a `sonnet` attempt that ends blocked, leaves a check red or is rejected in review escalates every retry to the session model. The whole-branch review runs on the session model. Task 2 touches the statics path (the force's application point); its maths is fixed by the decisions and pinned by tests, so it is transcription, but its review must check the residual test and the sweep-equality test ran, not only compiled.

## Review Focus

The inputs and conditions the requests imply but no feature test would naturally hit, most likely first, each pinned by a test in the task that owns the code:

1. **A file written before this change** (a geometry without `shape`, a zone without `at_contact_point`). Expected: it loads as a rectangle and keeps its locked point or centroid; written back, it gains no new keys. Tests: `a_rectangle_is_written_without_a_shape_key_and_old_files_load_as_rectangles` (Task 1), `at_contact_point_is_left_out_when_false_and_defaults_to_false` (Task 2).
2. **A contact point while the shape is outside the zone.** Expected: no force (the overlap gate still decides), but the canvas still marks where the force will act. Test: `the_contact_point_still_needs_the_shape_in_the_zone` (Task 2).
3. **Dragging the F marker of a zone in contact mode.** Expected: the zone switches to Locked at the drop point (today the drag would set a locked point that the contact mode then ignores). Test: `dragging_the_contact_marker_locks_it_where_it_drops` (Task 3).
4. **A level rectangle against a vertical force** (two corners equally low). Expected: the contact point is the bottom edge's midpoint, not one arbitrary corner; tilted, the lowest corner. Test: `a_rectangle_s_extreme_point_is_its_farthest_corner_or_its_edge_midpoint` (Task 1).
5. **A sideways or zero force in contact mode.** Expected: a sideways push meets the shape's side; a zero force falls back to the lowest point (and applies nothing). Tests: `a_sideways_force_meets_the_side_of_the_wheel` (Task 2), `a_zero_direction_gives_the_shape_s_centre` (Task 1, the geometry layer; the zone layer passes a downward direction for a zero force).

## File Structure

Part A (Tasks 1 to 3):

- `linkage-sim-rs/src/core/body.rs`: `GeometryShape`, `CIRCLE_SEGMENTS`, `BodyGeometry.shape`, `BodyGeometry::circle`, `centre_world`, `outline_world`, `extreme_point_world`; `area` covers circles (Task 1).
- `linkage-sim-rs/src/io/schema.rs`: `SCHEMA_VERSION` 1.2.0 (Task 1).
- `linkage-sim-rs/src/forces/elements/element_types.rs`: `ForceZoneElement.at_contact_point`, `ZoneAppMode`, `ForceZoneElement::app_mode`, `with_app_mode`, `is_false` (Task 2).
- `linkage-sim-rs/src/forces/elements/evaluation.rs`: `ZoneApplication`, `force_zone_application`; `evaluate_force_zone` and `force_zone_overlap_ratio` call it (Task 2).
- `linkage-sim-rs/src/solver/reactions.rs`: the independent check's force-zone branch calls it; a contact-mode validation test (Task 2).
- `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs`: `force_zone_app_point_world` and the overlap highlight and F marker call it; "F (contact)" (Task 2).
- `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`: the geometry drawn as a rectangle or a circle; the body tooltip (Task 3).
- `linkage-sim-rs/src/gui/property_panel/mod.rs`, `pending_edits.rs`: the shape switch, Diameter, Add rectangle / Add circle; one `edit_geometry` helper for the blueprint and mechanism copies; `default_geometry` (Task 3).
- `linkage-sim-rs/src/gui/property_panel/force_editor.rs`: the three-way application point (Task 3).
- `linkage-sim-rs/src/gui/canvas/interaction.rs`: dropping the F marker locks it (Task 3); the zone tool's literal gains the new field (Task 2).
- Literal sites of `BodyGeometry` (Task 1) and `ForceZoneElement` (Task 2): `pending_edits.rs`, `dxf_import.rs`, `reactions.rs` tests, `samples/fourbar.rs`, `interaction.rs`, `tests/force_zone_tests.rs`.
- Tests: `tests/geometry_tests.rs` (Task 1), `tests/force_zone_tests.rs` and the new `tests/wheel_contact_sweep.rs` (Task 2), `pending_edits.rs` and `gui/canvas/mod.rs` test modules (Task 3).

Part B: see "Part B: animated HTML export" at the end.

## Tasks

### Task 0: Worktree, baseline and context

**Files:** none changed (a worktree is created).

- [ ] **Step 1: Create the LF worktree**

```bash
git -C C:/Users/Cole/source/repos/linkage_simulation -c core.autocrlf=false worktree add -b linkage/round-geometry C:/Users/Cole/source/repos/lsim-shapes f9df219
git -C C:/Users/Cole/source/repos/lsim-shapes config --worktree core.autocrlf false
git -C C:/Users/Cole/source/repos/lsim-shapes log --oneline -1
```

Expected: `f9df219 merge: BL-042 weights draw over force zones`.

- [ ] **Step 2: Read the project context**

Read `C:/Users/Cole/source/repos/lsim-shapes/docs/ai/01-meta.yaml`, `02-system.yaml` (the force-zone and geometry entries), `03-structure.yaml` (the `geometry` and `forces` modules) and `04-memory.yaml`.

- [ ] **Step 3: Baseline gate**

Run the gate command from Global Constraints. Expected: `GATE PASS`. Record the linkage `cargo test` pass count (gate 1's last `test result` lines). Then `git -C C:/Users/Cole/source/repos/lsim-shapes checkout -- docs/chebyshev_lambda`.

- [ ] **Step 4: Confirm the old blocks exist**

```bash
cd C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs
grep -c 'body_local_app_point: None,' tests/force_zone_tests.rs
grep -n 'pub const SCHEMA_VERSION: &str = "1.1.0";' src/io/schema.rs
```

Expected: `9`, and one line at 18.

- [ ] **Step 5: Report** the baseline counts. No commit.

- [x] **Step 6: Decisions.** Confirmed by the user on 2026-10-06: the recommended option on all of R-1..R-7 and H-1..H-8 (recorded under Decisions to confirm).

---

### Task 1: Circle shapes in the core

**Files:**
- Modify: `linkage-sim-rs/src/core/body.rs:17-42`
- Modify: `linkage-sim-rs/src/io/schema.rs:18`
- Modify (struct literals): `linkage-sim-rs/src/gui/property_panel/pending_edits.rs:134-147`, `linkage-sim-rs/src/gui/dxf_import.rs:1119-1123`, `linkage-sim-rs/src/solver/reactions.rs:1581-1585` and `:1933-1937`
- Test: `linkage-sim-rs/tests/geometry_tests.rs`

**Interfaces:**
- Produces: `pub enum GeometryShape { Rectangle, Circle }` (serde `snake_case`, `Default` = `Rectangle`, `Copy`, `PartialEq`); `pub const CIRCLE_SEGMENTS: usize = 64`; `BodyGeometry { width, height, offset, shape }`; `BodyGeometry::circle(diameter: f64, offset: Vector2<f64>) -> Result<BodyGeometry, BodyError>`; `fn centre_world(&self, body_x: f64, body_y: f64, body_theta: f64) -> Vector2<f64>`; `fn outline_world(&self, body_x: f64, body_y: f64, body_theta: f64) -> Vec<Vector2<f64>>` (CCW; 4 corners or 64 vertices); `fn extreme_point_world(&self, body_x: f64, body_y: f64, body_theta: f64, direction: Vector2<f64>) -> Vector2<f64>`. All in `crate::core::body`.

- [ ] **Step 1: Write the failing tests**

In `linkage-sim-rs/tests/geometry_tests.rs`, replace:

```rust
use linkage_sim_rs::core::body::BodyGeometry;
```

with:

```rust
use linkage_sim_rs::core::body::{BodyGeometry, GeometryShape, CIRCLE_SEGMENTS};
```

Then append to the end of the file:

```rust

// ── Round shapes (schema 1.2.0, decisions R-1, R-3, R-5, R-6) ───────────

#[test]
fn a_circle_keeps_its_diameter_as_width_and_height() {
    let geo = BodyGeometry::circle(0.13208, Vector2::new(0.3, 0.05)).unwrap();
    assert_eq!(geo.shape, GeometryShape::Circle);
    assert_relative_eq!(geo.width, 0.13208);
    assert_relative_eq!(geo.height, 0.13208);
    assert_relative_eq!(
        geo.area(),
        std::f64::consts::PI * 0.13208 * 0.13208 / 4.0,
        epsilon = 1e-15
    );
}

#[test]
fn a_circle_needs_a_positive_finite_diameter() {
    for d in [0.0, -0.01, f64::NAN, f64::INFINITY] {
        assert!(BodyGeometry::circle(d, Vector2::zeros()).is_err(), "diameter {d}");
    }
}

#[test]
fn a_rectangle_outline_is_its_four_corners() {
    let offset = Vector2::new(0.05, 0.0);
    let geo = BodyGeometry::new(0.2, 0.1, offset).unwrap();
    assert_eq!(geo.shape, GeometryShape::Rectangle);
    let outline = geo.outline_world(1.0, 2.0, FRAC_PI_2);
    let corners = body_rect_to_world(1.0, 2.0, FRAC_PI_2, 0.2, 0.1, &offset);
    assert_eq!(outline.len(), 4);
    for (a, b) in outline.iter().zip(corners.iter()) {
        assert_relative_eq!(a.x, b.x, epsilon = 1e-12);
        assert_relative_eq!(a.y, b.y, epsilon = 1e-12);
    }
}

#[test]
fn a_circle_outline_is_a_polygon_on_the_circle_turned_with_the_body() {
    let geo = BodyGeometry::circle(0.1, Vector2::new(0.2, 0.0)).unwrap();
    let outline = geo.outline_world(1.0, 2.0, FRAC_PI_2);
    assert_eq!(outline.len(), CIRCLE_SEGMENTS);
    let centre = geo.centre_world(1.0, 2.0, FRAC_PI_2);
    assert_relative_eq!(centre.x, 1.0, epsilon = 1e-12);
    assert_relative_eq!(centre.y, 2.2, epsilon = 1e-12);
    for p in &outline {
        assert_relative_eq!((p - centre).norm(), 0.05, epsilon = 1e-12);
    }
    // Counter-clockwise, like the rectangle's corners: positive signed area.
    let n = outline.len();
    let signed: f64 = (0..n)
        .map(|i| {
            let (a, b) = (outline[i], outline[(i + 1) % n]);
            a.x * b.y - b.x * a.y
        })
        .sum::<f64>()
        / 2.0;
    assert!(signed > 0.0, "signed area {signed}");
    assert_relative_eq!(polygon_area(&outline), signed, epsilon = 1e-15);
}

#[test]
fn a_circle_s_extreme_point_is_its_centre_plus_the_radius_along_the_direction() {
    let geo = BodyGeometry::circle(0.1, Vector2::new(0.2, 0.0)).unwrap();
    // However the body turns, the point does not turn with it.
    for theta in [0.0, 0.7, -2.0] {
        let centre = geo.centre_world(1.0, 2.0, theta);
        let low = geo.extreme_point_world(1.0, 2.0, theta, Vector2::new(0.0, -3.0));
        assert_relative_eq!(low.x, centre.x, epsilon = 1e-12);
        assert_relative_eq!(low.y, centre.y - 0.05, epsilon = 1e-12);
        let corner = geo.extreme_point_world(1.0, 2.0, theta, Vector2::new(1.0, 1.0));
        let r = 0.05 / 2f64.sqrt();
        assert_relative_eq!(corner.x, centre.x + r, epsilon = 1e-12);
        assert_relative_eq!(corner.y, centre.y + r, epsilon = 1e-12);
    }
}

#[test]
fn a_rectangle_s_extreme_point_is_its_farthest_corner_or_its_edge_midpoint() {
    let geo = BodyGeometry::new(0.2, 0.1, Vector2::zeros()).unwrap();
    // Level: the whole bottom edge meets a surface below; its midpoint.
    let level = geo.extreme_point_world(0.0, 0.0, 0.0, Vector2::new(0.0, -1.0));
    assert_relative_eq!(level.x, 0.0, epsilon = 1e-12);
    assert_relative_eq!(level.y, -0.05, epsilon = 1e-12);
    // Tilted: one corner is lowest.
    let theta = 0.3_f64;
    let tilted = geo.extreme_point_world(0.0, 0.0, theta, Vector2::new(0.0, -1.0));
    let corners = body_rect_to_world(0.0, 0.0, theta, 0.2, 0.1, &Vector2::zeros());
    let lowest = corners.iter().min_by(|a, b| a.y.total_cmp(&b.y)).unwrap();
    assert_relative_eq!(tilted.x, lowest.x, epsilon = 1e-12);
    assert_relative_eq!(tilted.y, lowest.y, epsilon = 1e-12);
}

#[test]
fn a_zero_direction_gives_the_shape_s_centre() {
    let shapes = [
        BodyGeometry::new(0.2, 0.1, Vector2::new(0.05, 0.0)).unwrap(),
        BodyGeometry::circle(0.1, Vector2::new(0.05, 0.0)).unwrap(),
    ];
    for geo in shapes {
        for direction in [Vector2::zeros(), Vector2::new(f64::NAN, 1.0)] {
            let p = geo.extreme_point_world(1.0, 2.0, 0.4, direction);
            let c = geo.centre_world(1.0, 2.0, 0.4);
            assert_relative_eq!(p.x, c.x, epsilon = 1e-15);
            assert_relative_eq!(p.y, c.y, epsilon = 1e-15);
        }
    }
}

#[test]
fn a_rectangle_is_written_without_a_shape_key_and_old_files_load_as_rectangles() {
    let rect = BodyGeometry::new(0.2, 0.1, Vector2::new(0.05, 0.0)).unwrap();
    let json = serde_json::to_value(&rect).unwrap();
    assert!(json.get("shape").is_none(), "{json}");
    let old: BodyGeometry =
        serde_json::from_str(r#"{"width":0.2,"height":0.1,"offset":[0.05,0.0]}"#).unwrap();
    assert_eq!(old.shape, GeometryShape::Rectangle);
}

#[test]
fn a_circle_round_trips_and_keeps_width_and_height_for_older_builds() {
    let circle = BodyGeometry::circle(0.13208, Vector2::new(0.3, 0.05)).unwrap();
    let json = serde_json::to_value(&circle).unwrap();
    assert_eq!(json["shape"], "circle");
    // A build before 1.2.0 ignores "shape" and draws the circle's bounding square.
    assert_eq!(json["width"], json["height"]);
    let back: BodyGeometry = serde_json::from_value(json).unwrap();
    assert_eq!(back.shape, GeometryShape::Circle);
    assert_relative_eq!(back.width, 0.13208);
}

#[test]
fn files_with_shapes_are_schema_1_2_0() {
    assert_eq!(linkage_sim_rs::io::SCHEMA_VERSION, "1.2.0");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --test geometry_tests`
Expected: FAIL to compile ("unresolved imports `GeometryShape`, `CIRCLE_SEGMENTS`").

- [ ] **Step 3: Implement the shape**

In `linkage-sim-rs/src/core/body.rs`, replace:

```rust
/// Visual rectangular geometry attached to a body.
/// Used for force zone overlap computation and canvas rendering.
/// Dimensions are in body-local frame, centered on `offset` from body origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyGeometry {
    pub width: f64,            // meters, extent along body's local x-axis
    pub height: f64,           // meters, extent along body's local y-axis
    pub offset: Vector2<f64>,  // local-frame offset from body origin
}

impl BodyGeometry {
    /// Create a new body geometry. Width and height must be > 0.
    pub fn new(width: f64, height: f64, offset: Vector2<f64>) -> Result<Self, BodyError> {
        if width <= 0.0 || height <= 0.0 {
            return Err(BodyError::InvalidGeometry {
                reason: format!("width ({}) and height ({}) must both be > 0", width, height),
            });
        }
        Ok(Self { width, height, offset })
    }

    /// Total area of the rectangle (m^2).
    pub fn area(&self) -> f64 {
        self.width * self.height
    }
}
```

with:

```rust
/// The shape of a body's visual geometry (decision R-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryShape {
    /// A `width` x `height` rectangle; every file written before schema 1.2.0.
    #[default]
    Rectangle,
    /// A circle of diameter `width`. `height` is kept equal to it, so a build
    /// before schema 1.2.0, which ignores the shape, draws its bounding square.
    Circle,
}

impl GeometryShape {
    /// Rectangles are written without a `shape` key, so files without circles
    /// keep the keys they had before schema 1.2.0.
    fn is_rectangle(&self) -> bool {
        *self == GeometryShape::Rectangle
    }
}

/// Number of sides of the polygon that stands in for a circle in the force-zone
/// overlap test and its centroid; the canvas draws a true circle (decision R-6).
pub const CIRCLE_SEGMENTS: usize = 64;

/// Visual geometry attached to a body: a rectangle or a circle.
/// Used for force zone overlap computation and canvas rendering.
/// Dimensions are in body-local frame, centered on `offset` from body origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyGeometry {
    pub width: f64,            // meters, extent along body's local x-axis (a circle's diameter)
    pub height: f64,           // meters, extent along body's local y-axis (a circle: equal to width)
    pub offset: Vector2<f64>,  // local-frame offset from body origin
    /// Rectangle (the default, left out of files) or circle.
    #[serde(default, skip_serializing_if = "GeometryShape::is_rectangle")]
    pub shape: GeometryShape,
}

impl BodyGeometry {
    /// Create a rectangle. Width and height must be > 0.
    pub fn new(width: f64, height: f64, offset: Vector2<f64>) -> Result<Self, BodyError> {
        if width <= 0.0 || height <= 0.0 {
            return Err(BodyError::InvalidGeometry {
                reason: format!("width ({}) and height ({}) must both be > 0", width, height),
            });
        }
        Ok(Self { width, height, offset, shape: GeometryShape::Rectangle })
    }

    /// Create a circle of `diameter` centred on `offset`. The diameter must be
    /// a finite number > 0.
    pub fn circle(diameter: f64, offset: Vector2<f64>) -> Result<Self, BodyError> {
        if !(diameter > 0.0 && diameter.is_finite()) {
            return Err(BodyError::InvalidGeometry {
                reason: format!("diameter ({}) must be a finite number > 0", diameter),
            });
        }
        Ok(Self { width: diameter, height: diameter, offset, shape: GeometryShape::Circle })
    }

    /// Area (m^2): width x height for a rectangle, pi d^2 / 4 for a circle.
    pub fn area(&self) -> f64 {
        match self.shape {
            GeometryShape::Rectangle => self.width * self.height,
            GeometryShape::Circle => std::f64::consts::PI * self.width * self.width / 4.0,
        }
    }

    /// The shape's centre in the world frame, for a body at (`body_x`, `body_y`)
    /// turned by `body_theta`.
    pub fn centre_world(&self, body_x: f64, body_y: f64, body_theta: f64) -> Vector2<f64> {
        local_to_world(body_x, body_y, body_theta, self.offset)
    }

    /// The outline in the world frame, counter-clockwise: the rectangle's four
    /// corners (bottom-left first), or a `CIRCLE_SEGMENTS`-gon inscribed in the
    /// circle.
    pub fn outline_world(&self, body_x: f64, body_y: f64, body_theta: f64) -> Vec<Vector2<f64>> {
        match self.shape {
            GeometryShape::Rectangle => crate::geometry::body_rect_to_world(
                body_x,
                body_y,
                body_theta,
                self.width,
                self.height,
                &self.offset,
            )
            .to_vec(),
            GeometryShape::Circle => {
                let r = self.width / 2.0;
                (0..CIRCLE_SEGMENTS)
                    .map(|k| {
                        let a = std::f64::consts::TAU * k as f64 / CIRCLE_SEGMENTS as f64;
                        let local = self.offset + Vector2::new(r * a.cos(), r * a.sin());
                        local_to_world(body_x, body_y, body_theta, local)
                    })
                    .collect()
            }
        }
    }

    /// The point of the shape farthest along `direction` (world frame) for a
    /// body at (`body_x`, `body_y`) turned by `body_theta`: where a surface
    /// approaching from that side first touches it (decision R-3). For a
    /// circle, the centre plus the radius along the direction, wherever the
    /// body has turned; for a rectangle, its farthest corner, or the midpoint
    /// of the edge when two corners tie (an edge square to the direction). A
    /// zero or non-finite direction gives the centre.
    pub fn extreme_point_world(
        &self,
        body_x: f64,
        body_y: f64,
        body_theta: f64,
        direction: Vector2<f64>,
    ) -> Vector2<f64> {
        let centre = self.centre_world(body_x, body_y, body_theta);
        let norm = direction.norm();
        if !(norm > 0.0 && norm.is_finite()) {
            return centre;
        }
        let d = direction / norm;
        match self.shape {
            GeometryShape::Circle => centre + d * (self.width / 2.0),
            GeometryShape::Rectangle => {
                let corners = self.outline_world(body_x, body_y, body_theta);
                let best = corners.iter().map(|c| c.dot(&d)).fold(f64::NEG_INFINITY, f64::max);
                let tol = 1e-9 * self.width.max(self.height);
                let tied: Vec<&Vector2<f64>> =
                    corners.iter().filter(|c| best - c.dot(&d) <= tol).collect();
                tied.iter().fold(Vector2::zeros(), |sum, c| sum + **c) / tied.len() as f64
            }
        }
    }
}

/// A body-local point in the world frame, for a body at (`x`, `y`) turned by
/// `theta`.
fn local_to_world(x: f64, y: f64, theta: f64, p: Vector2<f64>) -> Vector2<f64> {
    let (sin_t, cos_t) = theta.sin_cos();
    Vector2::new(x + cos_t * p.x - sin_t * p.y, y + sin_t * p.x + cos_t * p.y)
}
```

- [ ] **Step 4: Bump the schema version (decision R-5)**

In `linkage-sim-rs/src/io/schema.rs`, replace:

```rust
/// Current schema version for the JSON format.
pub const SCHEMA_VERSION: &str = "1.1.0";
```

with:

```rust
/// Current schema version for the JSON format. 1.2.0 added circle geometry
/// (`BodyGeometry.shape`) and a force zone's contact point
/// (`ForceZoneElement.at_contact_point`); both are left out of a file that does
/// not use them, and the loader checks only the major number.
pub const SCHEMA_VERSION: &str = "1.2.0";
```

- [ ] **Step 5: Give the struct literals the shape**

In `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`, replace both occurrences (the blueprint's and the mechanism's) of:

```rust
                        body.geometry = Some(BodyGeometry {
                            width,
                            height,
                            offset: Vector2::zeros(),
                        });
```

with:

```rust
                        body.geometry = Some(BodyGeometry {
                            width,
                            height,
                            offset: Vector2::zeros(),
                            shape: crate::core::body::GeometryShape::Rectangle,
                        });
```

(Task 3 replaces this handler; this step only keeps the crate compiling.)

In `linkage-sim-rs/src/gui/dxf_import.rs`, replace:

```rust
            body.geometry = Some(BodyGeometry {
                width,
                height,
                offset: offset_local,
            });
```

with:

```rust
            body.geometry = Some(BodyGeometry {
                width,
                height,
                offset: offset_local,
                shape: crate::core::body::GeometryShape::Rectangle,
            });
```

In `linkage-sim-rs/src/solver/reactions.rs`, replace both occurrences (the tests `validation_verified_with_force_zone` and `validation_verified_force_zone_centroid_branch`) of:

```rust
        coupler.geometry = Some(BodyGeometry {
            width: 3.0,
            height: 0.4,
            offset: Vector2::new(1.5, 0.0),
        });
```

with:

```rust
        coupler.geometry = Some(BodyGeometry::new(3.0, 0.4, Vector2::new(1.5, 0.0)).unwrap());
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --test geometry_tests`
Expected: PASS (all tests, the ten new ones included).

Then run the gate (Global Constraints). Expected: `GATE PASS`, gate 1's pass count = Task 0's + 10. Restore `docs/chebyshev_lambda`.

- [ ] **Step 7: Commit**

```bash
cd C:/Users/Cole/source/repos/lsim-shapes
git add linkage-sim-rs/src/core/body.rs linkage-sim-rs/src/io/schema.rs linkage-sim-rs/src/gui/property_panel/pending_edits.rs linkage-sim-rs/src/gui/dxf_import.rs linkage-sim-rs/src/solver/reactions.rs linkage-sim-rs/tests/geometry_tests.rs
git commit -m "feat(linkage): circle body geometry (decisions R-1, R-5, R-6)" -m "BodyGeometry gains a shape: rectangle (the default, left out of files) or circle (diameter in width, height equal), with centre_world, outline_world and extreme_point_world. Schema 1.2.0." -m "Co-Authored-By: <model> <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01FSw7N8oV7SUVKTDVrGKLdg"
```

---

### Task 2: A force zone's contact point, one shared helper

**Files:**
- Modify: `linkage-sim-rs/src/forces/elements/element_types.rs:301-328`
- Modify: `linkage-sim-rs/src/forces/elements/evaluation.rs:1-11` (imports), `:445-554`
- Modify: `linkage-sim-rs/src/solver/reactions.rs:284-326`, `:1618-1625`, `:1968-1975`, and a new test after `validation_verified_force_zone_centroid_branch`
- Modify: `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs:77-117`, `:464-550`
- Modify (literals): `linkage-sim-rs/src/gui/canvas/interaction.rs:1113-1120`, `linkage-sim-rs/src/gui/samples/fourbar.rs:350-357`, `linkage-sim-rs/tests/force_zone_tests.rs` (9 literals)
- Test: `linkage-sim-rs/tests/force_zone_tests.rs`, create `linkage-sim-rs/tests/wheel_contact_sweep.rs`

**Interfaces:**
- Consumes (Task 1): `BodyGeometry::outline_world`, `extreme_point_world`, `centre_world`, `BodyGeometry::circle`.
- Produces: `ForceZoneElement.at_contact_point: bool`; `pub enum ZoneAppMode { Centroid, Locked, Contact }`; `ForceZoneElement::app_mode(&self) -> ZoneAppMode`; `ForceZoneElement::with_app_mode(&self, mode: ZoneAppMode, seed: [f64; 2]) -> ForceZoneElement`; `pub struct ZoneApplication { pub overlap: Vec<Vector2<f64>>, pub active: bool, pub point: Option<Vector2<f64>>, pub mode: ZoneAppMode }`; `pub fn force_zone_application(fz: &ForceZoneElement, geo: &BodyGeometry, pose: (f64, f64, f64)) -> ZoneApplication`. All re-exported from `crate::forces::elements`.

- [ ] **Step 1: Write the failing tests**

In `linkage-sim-rs/tests/force_zone_tests.rs`, replace:

```rust
use nalgebra::{DVector, Vector2};

use linkage_sim_rs::core::body::{make_bar, make_ground, BodyGeometry};
use linkage_sim_rs::core::mechanism::Mechanism;
use linkage_sim_rs::forces::elements::{ForceElement, ForceZoneElement};
```

with:

```rust
use approx::assert_relative_eq;
use nalgebra::{DVector, Vector2};

use linkage_sim_rs::core::body::{make_bar, make_ground, BodyGeometry};
use linkage_sim_rs::core::mechanism::Mechanism;
use linkage_sim_rs::forces::elements::{
    force_zone_application, ForceElement, ForceZoneElement, ZoneAppMode,
};
```

Then give every literal the new field: after each of the 9 lines `        body_local_app_point: None,` insert the line `        at_contact_point: false,`:

```bash
cd C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs
sed -i 's/^        body_local_app_point: None,$/        body_local_app_point: None,\n        at_contact_point: false,/' tests/force_zone_tests.rs
grep -c '        at_contact_point: false,' tests/force_zone_tests.rs
```

Expected: `9`.

Then append to the end of the file:

```rust

// ── Contact point (schema 1.2.0, decisions R-3, R-4) ─────────────────────

/// The bar of `build_test_mechanism` carrying a wheel (a 0.05 m circle centred
/// 0.1 m along it), turned to `theta`. Positions are not re-solved: the zone
/// reads the pose straight from q.
fn wheel_bar_at(theta: f64) -> (Mechanism, DVector<f64>) {
    let ground = make_ground(&[("O", 0.0, 0.0)]);
    let mut bar = make_bar("bar", "A", "B", 0.1, 0.0, 0.0);
    bar.geometry = Some(BodyGeometry::circle(0.05, Vector2::new(0.1, 0.0)).unwrap());

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(bar).unwrap();
    mech.add_revolute_joint("J1", "ground", "O", "bar", "A").unwrap();
    mech.add_constant_speed_driver("D1", "ground", "bar", 0.0, 0.0).unwrap();
    mech.build().unwrap();

    let mut q = DVector::zeros(mech.state().n_coords());
    q[mech.state().get_index("bar").unwrap().theta_idx()] = theta;
    (mech, q)
}

/// A zone over the whole wheel, its force at the contact point.
fn contact_zone(force: [f64; 2]) -> ForceZoneElement {
    ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [-1.0, -1.0],
        zone_max: [1.0, 1.0],
        force,
        label: None,
        body_local_app_point: None,
        at_contact_point: true,
    }
}

#[test]
fn the_contact_point_stays_at_the_bottom_of_the_wheel_as_it_turns() {
    for theta in [0.0, 0.6, 1.4, -0.9] {
        let (mech, q) = wheel_bar_at(theta);
        let geo = mech.bodies()["bar"].geometry.clone().unwrap();
        let pose = mech.state().get_pose("bar", &q);
        let app = force_zone_application(&contact_zone([0.0, 500.0]), &geo, pose);
        assert_eq!(app.mode, ZoneAppMode::Contact);
        assert!(app.active, "theta {theta}");
        let centre = geo.centre_world(pose.0, pose.1, pose.2);
        let point = app.point.unwrap();
        assert_relative_eq!(point.x, centre.x, epsilon = 1e-12);
        assert_relative_eq!(point.y, centre.y - 0.025, epsilon = 1e-12);
    }
}

#[test]
fn the_contact_point_loads_the_body_like_the_same_force_at_the_hub() {
    use linkage_sim_rs::forces::helpers::point_force_to_q;
    let (mech, q) = wheel_bar_at(0.6);
    let q_dot = DVector::zeros(q.len());
    let got = ForceElement::ForceZone(contact_zone([0.0, 500.0]))
        .evaluate(mech.state(), mech.bodies(), &q, &q_dot, 0.0);
    // The force at the wheel's bottom, given in the body frame.
    let (bx, by, th) = mech.state().get_pose("bar", &q);
    let geo = mech.bodies()["bar"].geometry.clone().unwrap();
    let bottom = geo.centre_world(bx, by, th) - Vector2::new(0.0, 0.025);
    let (s, c) = th.sin_cos();
    let (dx, dy) = (bottom.x - bx, bottom.y - by);
    let local = Vector2::new(c * dx + s * dy, -s * dx + c * dy);
    let force = Vector2::new(0.0, 500.0);
    let want = point_force_to_q(mech.state(), "bar", &local, &force, &q);
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(*g, *w, epsilon = 1e-12);
    }
    // A vertical force on the hub's vertical line: the same Q as at the hub.
    let at_hub = point_force_to_q(mech.state(), "bar", &Vector2::new(0.1, 0.0), &force, &q);
    for (g, h) in got.iter().zip(at_hub.iter()) {
        assert_relative_eq!(*g, *h, epsilon = 1e-9);
    }
}

#[test]
fn a_sideways_force_meets_the_side_of_the_wheel() {
    let (mech, q) = wheel_bar_at(0.0);
    let geo = mech.bodies()["bar"].geometry.clone().unwrap();
    let pose = mech.state().get_pose("bar", &q);
    let app = force_zone_application(&contact_zone([-500.0, 0.0]), &geo, pose);
    // Pushing towards -x: the surface comes from +x and meets the rightmost point.
    let p = app.point.unwrap();
    assert_relative_eq!(p.x, 0.125, epsilon = 1e-12);
    assert_relative_eq!(p.y, 0.0, epsilon = 1e-12);
    // A zero force falls back to the lowest point.
    let zero = force_zone_application(&contact_zone([0.0, 0.0]), &geo, pose);
    let p = zero.point.unwrap();
    assert_relative_eq!(p.x, 0.1, epsilon = 1e-12);
    assert_relative_eq!(p.y, -0.025, epsilon = 1e-12);
}

#[test]
fn the_contact_point_still_needs_the_shape_in_the_zone() {
    let (mech, q) = wheel_bar_at(0.0);
    let mut fz = contact_zone([0.0, 500.0]);
    fz.zone_min = [10.0, 10.0];
    fz.zone_max = [11.0, 11.0];
    let geo = mech.bodies()["bar"].geometry.clone().unwrap();
    let app = force_zone_application(&fz, &geo, mech.state().get_pose("bar", &q));
    assert!(!app.active);
    assert!(app.point.is_some(), "the canvas still marks where the force will act");
    let q_dot = DVector::zeros(q.len());
    let got = ForceElement::ForceZone(fz).evaluate(mech.state(), mech.bodies(), &q, &q_dot, 0.0);
    assert!(got.iter().all(|v| v.abs() < 1e-12), "{got:?}");
}

#[test]
fn the_contact_point_wins_over_a_locked_point_which_is_kept() {
    let mut fz = contact_zone([0.0, 500.0]);
    fz.body_local_app_point = Some([0.05, 0.0]);
    assert_eq!(fz.app_mode(), ZoneAppMode::Contact);
    fz.at_contact_point = false;
    assert_eq!(fz.app_mode(), ZoneAppMode::Locked);
    fz.body_local_app_point = None;
    assert_eq!(fz.app_mode(), ZoneAppMode::Centroid);
}

#[test]
fn with_app_mode_switches_and_keeps_the_locked_point() {
    let mut fz = contact_zone([0.0, 500.0]);
    fz.at_contact_point = false;
    // Locked with no earlier point: the seed (the panel passes the shape's centre).
    let locked = fz.with_app_mode(ZoneAppMode::Locked, [0.1, 0.0]);
    assert_eq!(locked.app_mode(), ZoneAppMode::Locked);
    assert_eq!(locked.body_local_app_point, Some([0.1, 0.0]));
    // Contact keeps the locked point; Locked again restores it, ignoring the seed.
    let contact = locked.with_app_mode(ZoneAppMode::Contact, [9.0, 9.0]);
    assert_eq!(contact.app_mode(), ZoneAppMode::Contact);
    assert_eq!(contact.body_local_app_point, Some([0.1, 0.0]));
    let back = contact.with_app_mode(ZoneAppMode::Locked, [9.0, 9.0]);
    assert_eq!(back.app_mode(), ZoneAppMode::Locked);
    assert_eq!(back.body_local_app_point, Some([0.1, 0.0]));
    // Overlap centre clears both.
    let centroid = contact.with_app_mode(ZoneAppMode::Centroid, [9.0, 9.0]);
    assert_eq!(centroid.app_mode(), ZoneAppMode::Centroid);
    assert_eq!(centroid.body_local_app_point, None);
    assert!(!centroid.at_contact_point);
}

#[test]
fn at_contact_point_is_left_out_when_false_and_defaults_to_false() {
    let mut fz = contact_zone([0.0, 500.0]);
    fz.at_contact_point = false;
    let json = serde_json::to_value(ForceElement::ForceZone(fz.clone())).unwrap();
    assert!(json.get("at_contact_point").is_none(), "{json}");
    let back: ForceElement = serde_json::from_value(json).unwrap();
    let ForceElement::ForceZone(back) = back else { panic!("a force zone") };
    assert!(!back.at_contact_point);
    fz.at_contact_point = true;
    let json = serde_json::to_value(ForceElement::ForceZone(fz)).unwrap();
    assert_eq!(json["at_contact_point"], true);
}
```

Create `linkage-sim-rs/tests/wheel_contact_sweep.rs`:

```rust
//! A wheel's contact point loads the linkage exactly like the same vertical
//! force locked at the wheel's hub, at every sample of a sweep (decision R-7:
//! a sample linkage stands in for the user's press, which the public repo does
//! not carry).

use linkage_sim_rs::core::body::BodyGeometry;
use linkage_sim_rs::forces::elements::ForceElement;
use linkage_sim_rs::gui::samples::SampleMechanism;
use linkage_sim_rs::gui::AppState;

/// The zone's force at the wheel's contact point (`contact`) or locked at its
/// hub, then rebuild and sweep. The blueprint is the same each time, so the
/// built mechanism orders its bodies the same way and the sweeps compare
/// sample for sample (two separately loaded states need not: each body map
/// iterates in its own order, and the solver may land on another 2 pi turn).
fn sweep_driver_torques(state: &mut AppState, contact: bool) -> Vec<f64> {
    for force in &mut state.blueprint.as_mut().expect("blueprint").forces {
        if let ForceElement::ForceZone(zone) = force {
            zone.at_contact_point = contact;
        }
    }
    state.rebuild();
    state.compute_sweep();
    state.sweep_data.as_ref().expect("a sweep").driver_torques.clone().expect("driver torques")
}

#[test]
fn a_wheel_s_contact_point_loads_the_linkage_like_its_hub() {
    // The Parallelogram Press with its coupler's rectangle swapped for a 40 mm
    // wheel centred where the rectangle was, the zone grown over the whole
    // sweep, and a locked point at the hub (kept while the contact mode is on).
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ParallelogramPress);
    let bp = state.blueprint.as_mut().expect("blueprint");
    let coupler = bp.bodies.get_mut("coupler").expect("coupler");
    let hub = coupler.geometry.as_ref().expect("the sample's geometry").offset;
    coupler.geometry = Some(BodyGeometry::circle(0.04, hub).unwrap());
    for force in &mut bp.forces {
        if let ForceElement::ForceZone(zone) = force {
            zone.zone_min = [-10.0, -10.0];
            zone.zone_max = [10.0, 10.0];
            zone.body_local_app_point = Some([hub.x, hub.y]);
        }
    }
    let hub_torques = sweep_driver_torques(&mut state, false);
    let contact_torques = sweep_driver_torques(&mut state, true);
    assert_eq!(hub_torques.len(), contact_torques.len());
    let mut compared = 0;
    for (h, c) in hub_torques.iter().zip(&contact_torques) {
        assert_eq!(h.is_finite(), c.is_finite(), "hub {h} vs contact {c}");
        if h.is_finite() {
            assert!((h - c).abs() <= 1e-9 * h.abs().max(1.0), "hub {h} vs contact {c}");
            compared += 1;
        }
    }
    assert!(compared > 300, "only {compared} samples solved");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --test force_zone_tests --test wheel_contact_sweep`
Expected: FAIL to compile (`force_zone_application`, `ZoneAppMode`, `at_contact_point` unknown).

- [ ] **Step 3: The zone's mode**

In `linkage-sim-rs/src/forces/elements/element_types.rs`, replace:

```rust
/// A spatial force zone: applies a constant distributed force to a body
/// proportional to the overlap area between the body's geometry and the zone.
///
/// The zone is an axis-aligned rectangle in world space. The body must have
/// `BodyGeometry` set. When `body_local_app_point` is `None`, the force is
/// applied at the centroid of the zone-geometry overlap polygon, projected
/// into the body's local frame each frame. When `Some`, the force is
/// applied at that fixed body-local point, letting the user pin the
/// application location to a specific contact point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceZoneElement {
```

with:

```rust
fn is_false(b: &bool) -> bool {
    !*b
}

/// How a force zone picks its application point (`ForceZoneElement::app_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneAppMode {
    /// The centroid of the geometry's overlap with the zone, found at every pose.
    Centroid,
    /// A fixed body-local point (`body_local_app_point`).
    Locked,
    /// The shape's contact point: its extreme point against the force, found
    /// at every pose (decision R-3).
    Contact,
}

/// A spatial force zone: applies its full constant force to a body whenever
/// the body's geometry overlaps the zone (binary: any overlap, full force).
///
/// The zone is an axis-aligned rectangle in world space. The body must have
/// `BodyGeometry` set. Where the force acts is `app_mode`: the overlap
/// centroid (no locked point), a locked body-local point
/// (`body_local_app_point`), or the shape's contact point
/// (`at_contact_point`), which wins over a locked point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceZoneElement {
```

Then replace:

```rust
    /// Optional override for the application point, in body-local
    /// coordinates (meters). When `Some`, replaces the auto-computed
    /// overlap centroid. See struct-level docs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_local_app_point: Option<[f64; 2]>,
}
```

with:

```rust
    /// Optional override for the application point, in body-local
    /// coordinates (meters). When `Some`, replaces the auto-computed
    /// overlap centroid. See struct-level docs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_local_app_point: Option<[f64; 2]>,
    /// When true, the force acts at the shape's contact point (the bottom of a
    /// wheel for an upward force), found again at every pose, so it slides
    /// round a circle as the body turns. Wins over `body_local_app_point`,
    /// which is kept so switching back restores it. Left out of files when
    /// false; absent from files before schema 1.2.0.
    #[serde(default, skip_serializing_if = "is_false")]
    pub at_contact_point: bool,
}

impl ForceZoneElement {
    /// Where the force acts: the contact point wins, then a locked point, else
    /// the overlap centroid.
    pub fn app_mode(&self) -> ZoneAppMode {
        if self.at_contact_point {
            ZoneAppMode::Contact
        } else if self.body_local_app_point.is_some() {
            ZoneAppMode::Locked
        } else {
            ZoneAppMode::Centroid
        }
    }

    /// This zone switched to `mode` (decision R-4): Centroid clears the locked
    /// point and the contact flag; Locked keeps an earlier locked point, else
    /// locks at `seed` (body-local); Contact sets the flag and keeps the locked
    /// point, so switching back restores it.
    pub fn with_app_mode(&self, mode: ZoneAppMode, seed: [f64; 2]) -> Self {
        let mut zone = self.clone();
        zone.at_contact_point = mode == ZoneAppMode::Contact;
        match mode {
            ZoneAppMode::Centroid => zone.body_local_app_point = None,
            ZoneAppMode::Locked => {
                zone.body_local_app_point = self.body_local_app_point.or(Some(seed));
            }
            ZoneAppMode::Contact => {}
        }
        zone
    }
}
```

- [ ] **Step 4: The shared helper and its callers**

In `linkage-sim-rs/src/forces/elements/evaluation.rs`, replace:

```rust
use crate::core::body::Body;
```

with:

```rust
use crate::core::body::{Body, BodyGeometry};
```

Then replace:

```rust
// ── Force zone evaluation ────────────────────────────────────────────────────

pub fn evaluate_force_zone(
    fz: &ForceZoneElement,
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
) -> DVector<f64> {
    use crate::geometry::{body_rect_to_world, clip_polygon_to_aabb, polygon_area, polygon_centroid};

    let n = state.n_coords();
    let body = match bodies.get(&fz.body_id) {
        Some(b) => b,
        None => return DVector::zeros(n),
    };
    let geo = match &body.geometry {
        Some(g) => g,
        None => return DVector::zeros(n),
    };

    // Get body position from state vector via BodyIndex
    let bi = match state.get_index(&fz.body_id) {
        Ok(idx) => idx,
        Err(_) => return DVector::zeros(n),
    };
    let bx = q[bi.x_idx()];
    let by = q[bi.y_idx()];
    let btheta = q[bi.theta_idx()];

    // Transform body rectangle to world frame
    let corners = body_rect_to_world(bx, by, btheta, geo.width, geo.height, &geo.offset);

    // Clip against zone AABB
    let zone_min = Vector2::new(fz.zone_min[0], fz.zone_min[1]);
    let zone_max = Vector2::new(fz.zone_max[0], fz.zone_max[1]);
    let clipped = clip_polygon_to_aabb(&corners, &zone_min, &zone_max);

    let overlap_area = polygon_area(&clipped);

    if overlap_area < 1e-15 {
        return DVector::zeros(n);
    }

    // Binary overlap: any contact → full force.
    let force_global = Vector2::new(fz.force[0], fz.force[1]);

    // Application point: either the user-pinned body-local override, or
    // the centroid of the overlap polygon projected into body-local
    // coords. The override is useful when the actual contact point isn't
    // the overlap centroid (e.g., a specific contact pad location).
    let local_point = if let Some(lp) = fz.body_local_app_point {
        Vector2::new(lp[0], lp[1])
    } else {
        let centroid_world = polygon_centroid(&clipped);
        let cos_t = btheta.cos();
        let sin_t = btheta.sin();
        let dx = centroid_world.x - bx;
        let dy = centroid_world.y - by;
        Vector2::new(
            cos_t * dx + sin_t * dy,
            -sin_t * dx + cos_t * dy,
        )
    };

    point_force_to_q(state, &fz.body_id, &local_point, &force_global, q)
}

/// Check whether any part of a force zone's target body overlaps the zone.
///
/// Returns 1.0 if any part of the body geometry intersects the zone, 0.0
/// otherwise. Binary semantics: partial overlap applies the full force.
pub fn force_zone_overlap_ratio(
    fz: &ForceZoneElement,
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
) -> f64 {
    use crate::geometry::{body_rect_to_world, clip_polygon_to_aabb, polygon_area};

    let body = match bodies.get(&fz.body_id) {
        Some(b) => b,
        None => return 0.0,
    };
    let geo = match &body.geometry {
        Some(g) => g,
        None => return 0.0,
    };

    let bi = match state.get_index(&fz.body_id) {
        Ok(idx) => idx,
        Err(_) => return 0.0,
    };
    let bx = q[bi.x_idx()];
    let by = q[bi.y_idx()];
    let btheta = q[bi.theta_idx()];

    let corners = body_rect_to_world(bx, by, btheta, geo.width, geo.height, &geo.offset);
    let zone_min = Vector2::new(fz.zone_min[0], fz.zone_min[1]);
    let zone_max = Vector2::new(fz.zone_max[0], fz.zone_max[1]);
    let clipped = clip_polygon_to_aabb(&corners, &zone_min, &zone_max);

    let overlap_area = polygon_area(&clipped);

    if overlap_area < 1e-15 {
        0.0
    } else {
        1.0
    }
}
```

with:

```rust
// ── Force zone evaluation ────────────────────────────────────────────────────

/// Where a force zone's force acts on its body at one pose.
#[derive(Debug, Clone)]
pub struct ZoneApplication {
    /// The body's geometry clipped to the zone (world frame); empty without overlap.
    pub overlap: Vec<Vector2<f64>>,
    /// True when the geometry overlaps the zone: the full force applies.
    pub active: bool,
    /// The application point (world frame) by `mode`: the contact point and a
    /// locked point always exist (the canvas marks them before the body
    /// reaches the zone); the overlap centroid only with overlap.
    pub point: Option<Vector2<f64>>,
    pub mode: ZoneAppMode,
}

/// The overlap gate and the application point of `fz` on a body whose
/// geometry is `geo`, at pose `(x, y, theta)`. The force evaluation, the
/// overlap ratio, the independent equilibrium check and the canvas all call
/// this, so they cannot disagree.
pub fn force_zone_application(
    fz: &ForceZoneElement,
    geo: &BodyGeometry,
    pose: (f64, f64, f64),
) -> ZoneApplication {
    use crate::geometry::{clip_polygon_to_aabb, polygon_area, polygon_centroid};

    let (bx, by, btheta) = pose;
    let outline = geo.outline_world(bx, by, btheta);
    let zone_min = Vector2::new(fz.zone_min[0], fz.zone_min[1]);
    let zone_max = Vector2::new(fz.zone_max[0], fz.zone_max[1]);
    let overlap = clip_polygon_to_aabb(&outline, &zone_min, &zone_max);
    // Binary overlap: any contact -> full force.
    let active = polygon_area(&overlap) >= 1e-15;
    let mode = fz.app_mode();
    let point = match mode {
        ZoneAppMode::Contact => {
            // A surface pushing along the force meets the shape's extreme point
            // against it: the lowest point for an upward push. A zero force
            // falls back to the lowest point.
            let against = Vector2::new(-fz.force[0], -fz.force[1]);
            let direction = if against.norm() > 0.0 { against } else { Vector2::new(0.0, -1.0) };
            Some(geo.extreme_point_world(bx, by, btheta, direction))
        }
        ZoneAppMode::Locked => {
            let lp = fz.body_local_app_point.unwrap_or([0.0, 0.0]);
            let (sin_t, cos_t) = btheta.sin_cos();
            Some(Vector2::new(bx + cos_t * lp[0] - sin_t * lp[1], by + sin_t * lp[0] + cos_t * lp[1]))
        }
        ZoneAppMode::Centroid => active.then(|| polygon_centroid(&overlap)),
    };
    ZoneApplication { overlap, active, point, mode }
}

/// The body's pose in `q` and its geometry, or `None` when the body is
/// unknown or has no geometry.
fn zone_body<'a>(
    fz: &ForceZoneElement,
    state: &State,
    bodies: &'a HashMap<String, Body>,
    q: &DVector<f64>,
) -> Option<(&'a BodyGeometry, (f64, f64, f64))> {
    let geo = bodies.get(&fz.body_id)?.geometry.as_ref()?;
    let bi = state.get_index(&fz.body_id).ok()?;
    Some((geo, (q[bi.x_idx()], q[bi.y_idx()], q[bi.theta_idx()])))
}

pub fn evaluate_force_zone(
    fz: &ForceZoneElement,
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
) -> DVector<f64> {
    let n = state.n_coords();
    let Some((geo, pose)) = zone_body(fz, state, bodies, q) else {
        return DVector::zeros(n);
    };
    let app = force_zone_application(fz, geo, pose);
    let (true, Some(world)) = (app.active, app.point) else {
        return DVector::zeros(n);
    };
    // The application point in the body's local frame, as point_force_to_q takes it.
    let (bx, by, btheta) = pose;
    let (sin_t, cos_t) = btheta.sin_cos();
    let (dx, dy) = (world.x - bx, world.y - by);
    let local_point = Vector2::new(cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy);
    let force_global = Vector2::new(fz.force[0], fz.force[1]);
    point_force_to_q(state, &fz.body_id, &local_point, &force_global, q)
}

/// Check whether any part of a force zone's target body overlaps the zone.
///
/// Returns 1.0 if any part of the body geometry intersects the zone, 0.0
/// otherwise. Binary semantics: partial overlap applies the full force.
pub fn force_zone_overlap_ratio(
    fz: &ForceZoneElement,
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
) -> f64 {
    match zone_body(fz, state, bodies, q) {
        Some((geo, pose)) if force_zone_application(fz, geo, pose).active => 1.0,
        _ => 0.0,
    }
}
```

In `linkage-sim-rs/src/solver/reactions.rs`, replace:

```rust
    // For force zones, two decisions are SHARED with production, not re-derived:
    // the binary overlap gate (polygon_area < 1e-15) and the unpinned-app-point
    // centroid (polygon_centroid). A bug in those shared geometry primitives
    // would corrupt both sides identically and cancel — narrow blast radius
    // (wrong app-POINT or overlap DECISION only, not a moment-arm projection).
    for fe in mech.forces() {
        match fe {
            ForceElement::ForceZone(fz) => {
                use crate::geometry::{
                    body_rect_to_world, clip_polygon_to_aabb, polygon_area, polygon_centroid,
                };
                let Some(body) = mech.bodies().get(&fz.body_id) else { continue };
                let Some(geo) = body.geometry.as_ref() else { continue };
                let (bx, by, bth) = state.get_pose(&fz.body_id, q);
                let corners = body_rect_to_world(bx, by, bth, geo.width, geo.height, &geo.offset);
                let zmin = Vector2::new(fz.zone_min[0], fz.zone_min[1]);
                let zmax = Vector2::new(fz.zone_max[0], fz.zone_max[1]);
                let clipped = clip_polygon_to_aabb(&corners, &zmin, &zmax);
                // Binary overlap: any contact → full force (matches
                // evaluate_force_zone).
                if polygon_area(&clipped) < 1e-15 {
                    continue;
                }
                let force = Vector2::new(fz.force[0], fz.force[1]);
                // World application point: pinned body-local override, else
                // the overlap-polygon centroid.
                let app = if let Some(lp) = fz.body_local_app_point {
                    let (c, s) = (bth.cos(), bth.sin());
                    Vector2::new(
                        bx + c * lp[0] - s * lp[1],
                        by + s * lp[0] + c * lp[1],
                    )
                } else {
                    polygon_centroid(&clipped)
                };
                accum(&mut net_f, &mut net_m, &cg_world, &mut char_len, &fz.body_id, app, force);
                force_scale = force_scale.max(force.norm());
            }
```

with:

```rust
    // For force zones, two decisions are SHARED with production, not re-derived:
    // the binary overlap gate and the world application point (the overlap
    // centroid, the locked point or the shape's contact point), both from
    // force_zone_application. A bug there would corrupt both sides identically
    // and cancel — narrow blast radius (wrong app-POINT or overlap DECISION
    // only, not a moment-arm projection).
    for fe in mech.forces() {
        match fe {
            ForceElement::ForceZone(fz) => {
                use crate::forces::elements::force_zone_application;
                let Some(body) = mech.bodies().get(&fz.body_id) else { continue };
                let Some(geo) = body.geometry.as_ref() else { continue };
                let zone = force_zone_application(fz, geo, state.get_pose(&fz.body_id, q));
                // Binary overlap: any contact → full force (matches
                // evaluate_force_zone).
                let (true, Some(app)) = (zone.active, zone.point) else { continue };
                let force = Vector2::new(fz.force[0], fz.force[1]);
                accum(&mut net_f, &mut net_m, &cg_world, &mut char_len, &fz.body_id, app, force);
                force_scale = force_scale.max(force.norm());
            }
```

In the same file, replace:

```rust
            label: None,
            body_local_app_point: Some([2.5, 0.0]),
        }));
```

with:

```rust
            label: None,
            body_local_app_point: Some([2.5, 0.0]),
            at_contact_point: false,
        }));
```

Then replace:

```rust
            force: [400.0, -1000.0],
            label: None,
            body_local_app_point: None,
        }));
```

with:

```rust
            force: [400.0, -1000.0],
            label: None,
            body_local_app_point: None,
            at_contact_point: false,
        }));
```

Then add the contact-mode check after the centroid test. Replace:

```rust
    /// A mechanism containing an element type the independent check does
    /// not model must report `Unverified` — never a false `Verified`/
    /// `Failed`. Two actuators trips the ">1 actuator" guard.
```

with:

```rust
    /// Pin the force-zone CONTACT branch (schema 1.2.0, decision R-3): a wheel
    /// (circle) on the coupler, the force at its extreme point against a
    /// tilted force, so the check's contact point must agree with production's.
    #[test]
    fn validation_verified_force_zone_contact_branch() {
        use crate::core::body::BodyGeometry;
        use crate::forces::elements::ForceZoneElement;
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let mut coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
        coupler.geometry = Some(BodyGeometry::circle(0.8, Vector2::new(2.0, 0.3)).unwrap());
        let rocker = make_bar("rocker", "D", "C", 2.0, 2.0, 0.02);
        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D").unwrap();
        mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();
        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        mech.add_force(ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "coupler".to_string(),
            point_a: [1.5, 0.0],
            point_a_name: None,
            body_b: "ground".to_string(),
            point_b: [2.0, 0.0],
            point_b_name: None,
            force: 0.0,
            speed_limit: 0.0,
            stroke_min: 0.0,
            stroke_max: 0.0,
            end_stop_stiffness: 0.0,
            end_stop_damping: 0.0,
            end_stop_restitution: 0.0,
        }));
        // A zone over the whole wheel; the force tilted so the contact point
        // sits off the wheel's vertical line and its moment is real.
        mech.add_force(ForceElement::ForceZone(ForceZoneElement {
            body_id: "coupler".to_string(),
            zone_min: [-10.0, -10.0],
            zone_max: [10.0, 10.0],
            force: [300.0, 1000.0],
            label: None,
            body_local_app_point: None,
            at_contact_point: true,
        }));
        mech.build().unwrap();

        let q = seed_pose_at_angle(&mech, PI / 3.0);
        let r = solve_reactions_with_actuator(&mech, &q, PI / 3.0, 1.0).unwrap();
        assert_eq!(
            r.validation(),
            ValidationState::Verified,
            "gravity + actuator + a contact-point zone should independently verify",
        );
        let resid = body_equilibrium_residual(&mech, &q, PI / 3.0, &r.reactions, r.actuator_force)
            .expect("modeled");
        assert!(resid < 1e-9, "contact-point equilibrium residual {resid}");
    }

    /// A mechanism containing an element type the independent check does
    /// not model must report `Unverified` — never a false `Verified`/
    /// `Failed`. Two actuators trips the ">1 actuator" guard.
```

In `linkage-sim-rs/src/gui/canvas/rendering/force_render.rs`, replace:

```rust
/// Compute the current world-space application point of a `ForceZoneElement`.
///
/// - If `body_local_app_point` is `Some`, the override point is transformed
///   by the body's current pose.
/// - Otherwise, the overlap polygon (body geometry ∩ zone AABB) is computed
///   and its centroid is returned.
///
/// Returns `None` when the target body is missing, has no geometry, or
/// there is no overlap and no override.
pub fn force_zone_app_point_world(
    fz: &ForceZoneElement,
    mech: &Mechanism,
    mech_state: &State,
    q: &nalgebra::DVector<f64>,
) -> Option<nalgebra::Vector2<f64>> {
    let body = mech.bodies().get(&fz.body_id)?;
    let geo = body.geometry.as_ref()?;
    let (bx, by, btheta) = mech_state.get_pose(&fz.body_id, q);
    let cos_t = btheta.cos();
    let sin_t = btheta.sin();

    if let Some(lp) = fz.body_local_app_point {
        return Some(nalgebra::Vector2::new(
            bx + cos_t * lp[0] - sin_t * lp[1],
            by + sin_t * lp[0] + cos_t * lp[1],
        ));
    }

    let corners = crate::geometry::body_rect_to_world(
        bx, by, btheta, geo.width, geo.height, &geo.offset,
    );
    let zmin = nalgebra::Vector2::new(fz.zone_min[0], fz.zone_min[1]);
    let zmax = nalgebra::Vector2::new(fz.zone_max[0], fz.zone_max[1]);
    let clipped = crate::geometry::clip_polygon_to_aabb(&corners, &zmin, &zmax);
    if clipped.len() >= 3 {
        Some(crate::geometry::polygon_centroid(&clipped))
    } else {
        None
    }
}
```

with:

```rust
/// Compute the current world-space application point of a `ForceZoneElement`
/// (`force_zone_application`): the contact point or the locked point at the
/// body's current pose, else the overlap centroid.
///
/// Returns `None` when the target body is missing, has no geometry, or the
/// zone uses the centroid and there is no overlap.
pub fn force_zone_app_point_world(
    fz: &ForceZoneElement,
    mech: &Mechanism,
    mech_state: &State,
    q: &nalgebra::DVector<f64>,
) -> Option<nalgebra::Vector2<f64>> {
    let geo = mech.bodies().get(&fz.body_id)?.geometry.as_ref()?;
    force_zone_application(fz, geo, mech_state.get_pose(&fz.body_id, q)).point
}
```

In the same file, replace:

```rust
    // 5. Active overlap highlight + application-point marker.
    if let Some(body) = mech.bodies().get(&fz.body_id) {
        if let Some(ref geo) = body.geometry {
            let (bx, by, btheta) = mech_state.get_pose(&fz.body_id, q);
            let corners = crate::geometry::body_rect_to_world(
                bx, by, btheta, geo.width, geo.height, &geo.offset,
            );
            let zone_min_v = nalgebra::Vector2::new(fz.zone_min[0], fz.zone_min[1]);
            let zone_max_v = nalgebra::Vector2::new(fz.zone_max[0], fz.zone_max[1]);
            let clipped = crate::geometry::clip_polygon_to_aabb(
                &corners, &zone_min_v, &zone_max_v,
            );
            let has_overlap = clipped.len() >= 3;
            if has_overlap {
                let screen_verts: Vec<Pos2> = clipped
                    .iter()
                    .map(|v| {
                        let sp = view.world_to_screen(v.x, v.y);
                        Pos2::new(sp[0], sp[1])
                    })
                    .collect();
                painter.add(egui::epaint::PathShape::convex_polygon(
                    screen_verts,
                    FORCE_ZONE_OVERLAP_FILL,
                    Stroke::new(1.0, FORCE_ZONE_OVERLAP_STROKE),
                ));
            }

            // Application-point marker. Always drawn when the zone has
            // any overlap (or when the user has pinned a body-local
            // override, regardless of overlap, so they can still see
            // where the force would act once the mechanism reaches the
            // zone). Shown as a crosshair + dot. A locked override uses
            // a stronger color; the auto-centroid uses a softer color.
            let cos_t = btheta.cos();
            let sin_t = btheta.sin();

            let (app_world, is_locked) = if let Some(lp) = fz.body_local_app_point {
                // Override: world = body_pose + R(theta) * local
                let wx = bx + cos_t * lp[0] - sin_t * lp[1];
                let wy = by + sin_t * lp[0] + cos_t * lp[1];
                (Some(nalgebra::Vector2::new(wx, wy)), true)
            } else if has_overlap {
                (Some(crate::geometry::polygon_centroid(&clipped)), false)
            } else {
                (None, false)
            };

            if let Some(app_w) = app_world {
                let sp = view.world_to_screen(app_w.x, app_w.y);
                let center = Pos2::new(sp[0], sp[1]);
                let color = if is_locked {
                    Color32::from_rgb(255, 165, 80)
                } else {
                    Color32::from_rgb(255, 215, 120)
                };
```

with:

```rust
    // 5. Active overlap highlight + application-point marker.
    if let Some(body) = mech.bodies().get(&fz.body_id) {
        if let Some(ref geo) = body.geometry {
            let zone = force_zone_application(fz, geo, mech_state.get_pose(&fz.body_id, q));
            if zone.active {
                let screen_verts: Vec<Pos2> = zone
                    .overlap
                    .iter()
                    .map(|v| {
                        let sp = view.world_to_screen(v.x, v.y);
                        Pos2::new(sp[0], sp[1])
                    })
                    .collect();
                painter.add(egui::epaint::PathShape::convex_polygon(
                    screen_verts,
                    FORCE_ZONE_OVERLAP_FILL,
                    Stroke::new(1.0, FORCE_ZONE_OVERLAP_STROKE),
                ));
            }

            // Application-point marker. Drawn whenever the point exists: a
            // locked or contact point even before the body reaches the zone,
            // so the user sees where the force will act; the centroid only
            // with overlap. Shown as a crosshair + dot. A locked or contact
            // point uses a stronger color; the auto-centroid a softer one.
            let (color, label) = match zone.mode {
                ZoneAppMode::Centroid => (Color32::from_rgb(255, 215, 120), "F"),
                ZoneAppMode::Locked => (Color32::from_rgb(255, 165, 80), "F (locked)"),
                ZoneAppMode::Contact => (Color32::from_rgb(255, 165, 80), "F (contact)"),
            };
            if let Some(app_w) = zone.point {
                let sp = view.world_to_screen(app_w.x, app_w.y);
                let center = Pos2::new(sp[0], sp[1]);
```

Then replace:

```rust
                // Label — "F" for auto, "F (locked)" for override.
                let label = if is_locked { "F (locked)" } else { "F" };
                painter.text(
```

with:

```rust
                painter.text(
```

(`force_render.rs` already has `use crate::forces::elements::*;`, which brings `force_zone_application` and `ZoneAppMode`.)

- [ ] **Step 5: The other zone literals**

In `linkage-sim-rs/src/gui/canvas/interaction.rs`, replace:

```rust
                        force: [0.0, -100.0],
                        label: None,
                        body_local_app_point: None,
                    });
```

with:

```rust
                        force: [0.0, -100.0],
                        label: None,
                        body_local_app_point: None,
                        at_contact_point: false,
                    });
```

In `linkage-sim-rs/src/gui/samples/fourbar.rs`, replace:

```rust
        label: Some("Force Zone".to_string()),
        body_local_app_point: None,
    }));
```

with:

```rust
        label: Some("Force Zone".to_string()),
        body_local_app_point: None,
        at_contact_point: false,
    }));
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --test force_zone_tests --test wheel_contact_sweep`
Expected: PASS (every old and new test).

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --lib validation_verified`
Expected: PASS, including `validation_verified_with_force_zone`, `validation_verified_force_zone_centroid_branch` and the new `validation_verified_force_zone_contact_branch`.

Then the gate. Expected: `GATE PASS`; gate 1's count = Task 1's + 9 (7 in `force_zone_tests`, 1 in `wheel_contact_sweep`, 1 in `reactions`). Restore `docs/chebyshev_lambda`.

- [ ] **Step 7: Commit**

```bash
cd C:/Users/Cole/source/repos/lsim-shapes
git add linkage-sim-rs/src/forces/elements/element_types.rs linkage-sim-rs/src/forces/elements/evaluation.rs linkage-sim-rs/src/solver/reactions.rs linkage-sim-rs/src/gui/canvas/rendering/force_render.rs linkage-sim-rs/src/gui/canvas/interaction.rs linkage-sim-rs/src/gui/samples/fourbar.rs linkage-sim-rs/tests/force_zone_tests.rs linkage-sim-rs/tests/wheel_contact_sweep.rs
git commit -m "feat(linkage): a force zone's contact point, one shared helper (decisions R-3, R-4, R-7)" -m "ForceZoneElement.at_contact_point: the force acts at the shape's extreme point against it (a wheel's bottom for an upward force). force_zone_application decides overlap and point for the force evaluation, the overlap ratio, the independent check and the canvas (F (contact))." -m "Co-Authored-By: <model> <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01FSw7N8oV7SUVKTDVrGKLdg"
```

---

### Task 3: Shapes and the application point in the GUI

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas/rendering/mod.rs:176-197`, `:1499-1505`
- Modify: `linkage-sim-rs/src/gui/property_panel/mod.rs:277-362`
- Modify: `linkage-sim-rs/src/gui/property_panel/pending_edits.rs` (variants `:27-32`, handlers `:131-219`, tests)
- Modify: `linkage-sim-rs/src/gui/property_panel/force_editor.rs:903-961`
- Modify: `linkage-sim-rs/src/gui/canvas/interaction.rs:287-290`
- Test: `linkage-sim-rs/src/gui/property_panel/pending_edits.rs` (tests module), `linkage-sim-rs/src/gui/canvas/mod.rs` (tests module)

**Interfaces:**
- Consumes (Tasks 1, 2): `GeometryShape`, `BodyGeometry::{new, circle, centre_world, outline_world}`, `ZoneAppMode`, `ForceZoneElement::{app_mode, with_app_mode}`.
- Produces: `PendingPropertyEdit::AddGeometry { body_id: String, shape: GeometryShape }` (replaces `{ body_id, width, height }`), `SetGeometryShape { body_id: String, shape: GeometryShape }`, `UpdateGeometryDiameter { body_id: String, diameter: f64 }`; `fn default_geometry(body: &crate::io::BodyJson, shape: GeometryShape) -> BodyGeometry` and `fn edit_geometry(state: &mut AppState, body_id: &str, edit: impl Fn(&mut Option<BodyGeometry>))` (private to `pending_edits.rs`).

- [ ] **Step 1: Write the failing tests**

In `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`, replace:

```rust
        assert_eq!(state.find_point_mass(&other, &w).unwrap().mass, 2.0, "the moved weight is untouched");
        assert_eq!(state.undo_history.undo_count(), depth + 1, "only the move recorded an entry");
    }
}
```

with:

```rust
        assert_eq!(state.find_point_mass(&other, &w).unwrap().mass, 2.0, "the moved weight is untouched");
        assert_eq!(state.undo_history.undo_count(), depth + 1, "only the move recorded an entry");
    }

    // ── Geometry shapes (decision R-2) ──────────────────────────────────

    /// `body`'s geometry in the blueprint and in the built mechanism.
    fn geometry_copies(state: &AppState, body: &str) -> [BodyGeometry; 2] {
        let bp = state.blueprint.as_ref().unwrap().bodies[body].geometry.clone();
        let mech = state.mechanism.as_ref().unwrap().bodies()[body].geometry.clone();
        [bp.expect("blueprint geometry"), mech.expect("mechanism geometry")]
    }

    /// The Four-Bar sample's coupler: its length and its midpoint (body-local).
    fn coupler_span(state: &AppState) -> (f64, [f64; 2]) {
        let points: Vec<[f64; 2]> =
            state.blueprint.as_ref().unwrap().bodies["coupler"].attachment_points.values().cloned().collect();
        assert_eq!(points.len(), 2, "the sample coupler is a two-point bar");
        let (a, b) = (points[0], points[1]);
        let span = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        (span, [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0])
    }

    fn add_geometry(state: &mut AppState, shape: GeometryShape) {
        apply_pending(state, Some(PendingPropertyEdit::AddGeometry { body_id: "coupler".into(), shape }));
    }

    #[test]
    fn add_circle_centres_a_wheel_half_the_link_long_in_both_copies() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Circle);
        let (span, mid) = coupler_span(&state);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Circle);
            assert!((geo.width - span / 2.0).abs() < 1e-12 && (geo.height - span / 2.0).abs() < 1e-12);
            assert!((geo.offset.x - mid[0]).abs() < 1e-12 && (geo.offset.y - mid[1]).abs() < 1e-12);
        }
    }

    #[test]
    fn add_rectangle_spans_the_link_a_quarter_as_deep() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Rectangle);
        let (span, mid) = coupler_span(&state);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Rectangle);
            assert!((geo.width - span).abs() < 1e-12 && (geo.height - span / 4.0).abs() < 1e-12);
            assert!((geo.offset.x - mid[0]).abs() < 1e-12 && (geo.offset.y - mid[1]).abs() < 1e-12);
        }
    }

    #[test]
    fn switching_shape_keeps_the_width_and_squares_the_height() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Rectangle);
        let (span, _) = coupler_span(&state);
        let set = |state: &mut AppState, shape| {
            apply_pending(state, Some(PendingPropertyEdit::SetGeometryShape { body_id: "coupler".into(), shape }));
        };
        set(&mut state, GeometryShape::Circle);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Circle);
            assert!((geo.width - span).abs() < 1e-12 && (geo.height - span).abs() < 1e-12);
        }
        set(&mut state, GeometryShape::Rectangle);
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!(geo.shape, GeometryShape::Rectangle);
            assert!((geo.width - span).abs() < 1e-12 && (geo.height - span).abs() < 1e-12);
        }
    }

    #[test]
    fn a_diameter_edit_sets_width_and_height_in_both_copies() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        add_geometry(&mut state, GeometryShape::Circle);
        apply_pending(
            &mut state,
            Some(PendingPropertyEdit::UpdateGeometryDiameter { body_id: "coupler".into(), diameter: 0.05 }),
        );
        for geo in geometry_copies(&state, "coupler") {
            assert_eq!((geo.width, geo.height), (0.05, 0.05));
        }
        apply_pending(&mut state, Some(PendingPropertyEdit::RemoveGeometry { body_id: "coupler".into() }));
        assert!(state.blueprint.as_ref().unwrap().bodies["coupler"].geometry.is_none());
        assert!(state.mechanism.as_ref().unwrap().bodies()["coupler"].geometry.is_none());
    }
}
```

In `linkage-sim-rs/src/gui/canvas/mod.rs`, replace (the end of the `weight_readout` module and of the tests module, as Task BL-042 left them):

```rust
            let highlight = highlight.expect("the coupler's overlap highlight is drawn");
            let marker = marker.expect("the weight marker is drawn");
            assert!(marker > highlight, "the weight (shape {marker}) is painted under the highlight (shape {highlight})");
        }
    }
}
```

with:

```rust
            let highlight = highlight.expect("the coupler's overlap highlight is drawn");
            let marker = marker.expect("the weight marker is drawn");
            assert!(marker > highlight, "the weight (shape {marker}) is painted under the highlight (shape {highlight})");
        }
    }

    /// Headless canvas frames checking how a circle geometry and a force
    /// zone's contact point are drawn and dragged (decisions R-2, R-3, R-4).
    mod geometry_drawing {
        use eframe::egui::{self, Pos2};

        use crate::core::body::{BodyGeometry, GeometryShape};
        use crate::forces::elements::{ForceElement, ZoneAppMode};
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::AppState;
        use crate::gui::test_support::{drawn_texts, primary_button, visit_shapes};
        use super::weight_clicks::frame;

        /// The Parallelogram Press with a 20 mm wheel on its coupler where its
        /// rectangle was, the zone grown over the canvas, the zone's force at
        /// the wheel's contact point when `contact`; after two idle frames.
        fn press_with_wheel(contact: bool) -> (egui::Context, AppState) {
            let mut state = AppState::default();
            state.load_sample(SampleMechanism::ParallelogramPress);
            let bp = state.blueprint.as_mut().expect("blueprint");
            let coupler = bp.bodies.get_mut("coupler").expect("coupler");
            let hub = coupler.geometry.as_ref().expect("the sample's geometry").offset;
            coupler.geometry = Some(BodyGeometry::circle(0.02, hub).unwrap());
            for force in &mut bp.forces {
                if let ForceElement::ForceZone(zone) = force {
                    zone.zone_min = [-10.0, -10.0];
                    zone.zone_max = [10.0, 10.0];
                    zone.at_contact_point = contact;
                }
            }
            state.rebuild();
            let ctx = egui::Context::default();
            frame(&ctx, &mut state, Vec::new());
            frame(&ctx, &mut state, Vec::new());
            (ctx, state)
        }

        fn zone(state: &AppState) -> crate::forces::elements::ForceZoneElement {
            state
                .mechanism
                .as_ref()
                .unwrap()
                .forces()
                .iter()
                .find_map(|f| if let ForceElement::ForceZone(z) = f { Some(z.clone()) } else { None })
                .expect("the press's zone")
        }

        #[test]
        fn a_circle_geometry_is_drawn_as_a_circle_of_its_radius() {
            let (ctx, mut state) = press_with_wheel(false);
            let output = frame(&ctx, &mut state, Vec::new());
            let geo = state.mechanism.as_ref().unwrap().bodies()["coupler"].geometry.clone().unwrap();
            assert_eq!(geo.shape, GeometryShape::Circle);
            let a = state.view.world_to_screen(0.0, 0.0);
            let b = state.view.world_to_screen(0.01, 0.0);
            let radius_px = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            let mut found = false;
            visit_shapes(&output, |shape| {
                if let egui::Shape::Circle(c) = shape {
                    if c.stroke.color == egui::Color32::from_rgb(255, 165, 0) && (c.radius - radius_px).abs() < 0.5 {
                        found = true;
                    }
                }
            });
            assert!(found, "a circle of radius {radius_px} px in the geometry colour");
        }

        #[test]
        fn the_contact_point_marker_reads_f_contact() {
            let (ctx, mut state) = press_with_wheel(true);
            let texts = drawn_texts(&frame(&ctx, &mut state, Vec::new()));
            assert!(texts.iter().any(|t| t == "F (contact)"), "{texts:?}");
            assert!(!texts.iter().any(|t| t == "F (locked)"), "{texts:?}");
        }

        #[test]
        fn dragging_the_contact_marker_locks_it_where_it_drops() {
            let (ctx, mut state) = press_with_wheel(true);
            let fz = zone(&state);
            let mech = state.mechanism.as_ref().unwrap();
            let world = crate::gui::canvas::rendering::force_zone_app_point_world(&fz, mech, mech.state(), &state.q)
                .expect("the contact point");
            let s = state.view.world_to_screen(world.x, world.y);
            let from = Pos2::new(s[0], s[1]);
            let to = from + egui::vec2(40.0, 30.0);
            // egui starts the drag once the pointer passes its click distance
            // (6 px) and reports the pointer's position then, which must still
            // be within the marker's hit radius (12 px): a first step of 8 px.
            frame(&ctx, &mut state, vec![egui::Event::PointerMoved(from)]);
            frame(&ctx, &mut state, vec![primary_button(from, true)]);
            frame(&ctx, &mut state, vec![egui::Event::PointerMoved(from + egui::vec2(8.0, 0.0))]);
            frame(&ctx, &mut state, vec![egui::Event::PointerMoved(to)]);
            frame(&ctx, &mut state, vec![primary_button(to, false)]);
            let fz = zone(&state);
            assert_eq!(fz.app_mode(), ZoneAppMode::Locked, "the drop locks the point");
            let [wx, wy] = state.view.screen_to_world(to.x, to.y);
            let mech = state.mechanism.as_ref().unwrap();
            let dropped = crate::gui::canvas::rendering::force_zone_app_point_world(&fz, mech, mech.state(), &state.q)
                .expect("the locked point");
            assert!((dropped.x - wx).abs() < 1e-9 && (dropped.y - wy).abs() < 1e-9, "{dropped:?} vs ({wx}, {wy})");
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --lib geometry`
Expected: FAIL to compile (`AddGeometry { shape }`, `SetGeometryShape`, `UpdateGeometryDiameter` unknown).

- [ ] **Step 3: The edits**

In `linkage-sim-rs/src/gui/property_panel/pending_edits.rs`, replace:

```rust
use nalgebra::Vector2;
use crate::core::body::BodyGeometry;
```

with:

```rust
use nalgebra::Vector2;
use crate::core::body::{BodyGeometry, GeometryShape};
```

Then replace:

```rust
    AddGeometry { body_id: String, width: f64, height: f64 },
    UpdateGeometryWidth { body_id: String, width: f64 },
```

with:

```rust
    /// Give a link without geometry a `shape` sized from the link (decision R-2).
    AddGeometry { body_id: String, shape: GeometryShape },
    /// Switch a link's geometry to `shape`, keeping its width.
    SetGeometryShape { body_id: String, shape: GeometryShape },
    /// Set a circle's diameter (width and height).
    UpdateGeometryDiameter { body_id: String, diameter: f64 },
    UpdateGeometryWidth { body_id: String, width: f64 },
```

Then replace the six geometry handlers (as Task 1 left the first one):

```rust
            PendingPropertyEdit::AddGeometry { body_id, width, height } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        body.geometry = Some(BodyGeometry {
                            width,
                            height,
                            offset: Vector2::zeros(),
                            shape: crate::core::body::GeometryShape::Rectangle,
                        });
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        body.geometry = Some(BodyGeometry {
                            width,
                            height,
                            offset: Vector2::zeros(),
                            shape: crate::core::body::GeometryShape::Rectangle,
                        });
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGeometryWidth { body_id, width } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.width = width;
                        }
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.width = width;
                        }
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGeometryHeight { body_id, height } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.height = height;
                        }
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.height = height;
                        }
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGeometryOffsetX { body_id, offset_x } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.offset.x = offset_x;
                        }
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.offset.x = offset_x;
                        }
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::UpdateGeometryOffsetY { body_id, offset_y } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.offset.y = offset_y;
                        }
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        if let Some(ref mut geo) = body.geometry {
                            geo.offset.y = offset_y;
                        }
                    }
                }
                state.mark_sweep_dirty();
            }
            PendingPropertyEdit::RemoveGeometry { body_id } => {
                if let Some(bp) = &mut state.blueprint {
                    if let Some(body) = bp.bodies.get_mut(&body_id) {
                        body.geometry = None;
                    }
                }
                if let Some(mech) = &mut state.mechanism {
                    if let Some(body) = mech.body_mut(&body_id) {
                        body.geometry = None;
                    }
                }
                state.mark_sweep_dirty();
            }
```

with:

```rust
            PendingPropertyEdit::AddGeometry { body_id, shape } => {
                let new = state
                    .blueprint
                    .as_ref()
                    .and_then(|bp| bp.bodies.get(&body_id))
                    .map(|body| default_geometry(body, shape));
                if let Some(new) = new {
                    edit_geometry(state, &body_id, |geo| *geo = Some(new.clone()));
                }
            }
            PendingPropertyEdit::SetGeometryShape { body_id, shape } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.shape = shape;
                        g.height = g.width;
                    }
                });
            }
            PendingPropertyEdit::UpdateGeometryDiameter { body_id, diameter } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.width = diameter;
                        g.height = diameter;
                    }
                });
            }
            PendingPropertyEdit::UpdateGeometryWidth { body_id, width } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.width = width;
                    }
                });
            }
            PendingPropertyEdit::UpdateGeometryHeight { body_id, height } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.height = height;
                    }
                });
            }
            PendingPropertyEdit::UpdateGeometryOffsetX { body_id, offset_x } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.offset.x = offset_x;
                    }
                });
            }
            PendingPropertyEdit::UpdateGeometryOffsetY { body_id, offset_y } => {
                edit_geometry(state, &body_id, |geo| {
                    if let Some(g) = geo {
                        g.offset.y = offset_y;
                    }
                });
            }
            PendingPropertyEdit::RemoveGeometry { body_id } => {
                edit_geometry(state, &body_id, |geo| *geo = None);
            }
```

Then add the two helpers before `apply_pending`. Replace:

```rust
/// Apply a pending property edit.
pub(super) fn apply_pending(state: &mut AppState, pending: Option<PendingPropertyEdit>) {
```

with:

```rust
/// Apply `edit` to `body_id`'s geometry in the blueprint and in the built
/// mechanism (the two copies the GUI keeps), then mark the sweep dirty.
fn edit_geometry(state: &mut AppState, body_id: &str, edit: impl Fn(&mut Option<BodyGeometry>)) {
    if let Some(body) = state.blueprint.as_mut().and_then(|bp| bp.bodies.get_mut(body_id)) {
        edit(&mut body.geometry);
    }
    if let Some(body) = state.mechanism.as_mut().and_then(|mech| mech.body_mut(body_id)) {
        edit(&mut body.geometry);
    }
    state.mark_sweep_dirty();
}

/// A new `shape` for `body` (decision R-2): centred on its attachment points'
/// centroid (body-local) and sized from their span L, the largest distance
/// between two of them (0.1 m with fewer than two): a rectangle L x L/4, a
/// circle of diameter L/2.
fn default_geometry(body: &crate::io::BodyJson, shape: GeometryShape) -> BodyGeometry {
    let points: Vec<Vector2<f64>> =
        body.attachment_points.values().map(|p| Vector2::new(p[0], p[1])).collect();
    let centre = if points.is_empty() {
        Vector2::zeros()
    } else {
        points.iter().sum::<Vector2<f64>>() / points.len() as f64
    };
    let span = points
        .iter()
        .flat_map(|a| points.iter().map(move |b| (a - b).norm()))
        .fold(0.0, f64::max);
    let span = if span > 0.0 { span } else { 0.1 };
    match shape {
        GeometryShape::Rectangle => BodyGeometry::new(span, span / 4.0, centre),
        GeometryShape::Circle => BodyGeometry::circle(span / 2.0, centre),
    }
    .expect("a positive span gives a valid shape")
}

/// Apply a pending property edit.
pub(super) fn apply_pending(state: &mut AppState, pending: Option<PendingPropertyEdit>) {
```

- [ ] **Step 4: The property panel**

In `linkage-sim-rs/src/gui/property_panel/mod.rs`, replace:

```rust
                                    .show(ui, |ui| {
                                        if let Some(ref geo) = bp_body.geometry {
                                            // Width slider (mm display, m internal)
                                            let mut width_mm = geo.width * 1e3;
```

with:

```rust
                                    .show(ui, |ui| {
                                        if let Some(ref geo) = bp_body.geometry {
                                            // Shape switch (decision R-2).
                                            let mut shape = geo.shape;
                                            ui.horizontal(|ui| {
                                                ui.label("Shape:");
                                                ui.selectable_value(&mut shape, GeometryShape::Rectangle, "Rectangle");
                                                ui.selectable_value(&mut shape, GeometryShape::Circle, "Circle");
                                            });
                                            if shape != geo.shape {
                                                pending = Some(PendingPropertyEdit::SetGeometryShape {
                                                    body_id: body_id.clone(),
                                                    shape,
                                                });
                                            }

                                            if geo.shape == GeometryShape::Circle {
                                            // Diameter slider (mm display, m internal)
                                            let mut diameter_mm = geo.width * 1e3;
                                            let dr = ui.add(
                                                egui::Slider::new(&mut diameter_mm, 1.0..=500.0)
                                                    .text("Diameter (mm)")
                                                    .clamping(egui::SliderClamping::Never)
                                                    .logarithmic(true),
                                            ).on_hover_text("Circle diameter in mm (visual geometry for force zones)");
                                            if dr.drag_stopped() || (dr.changed() && !dr.dragged()) {
                                                pending = Some(PendingPropertyEdit::UpdateGeometryDiameter {
                                                    body_id: body_id.clone(),
                                                    diameter: diameter_mm * 1e-3,
                                                });
                                            }
                                            } else {
                                            // Width slider (mm display, m internal)
                                            let mut width_mm = geo.width * 1e3;
```

Then replace:

```rust
                                            if hr.drag_stopped() || (hr.changed() && !hr.dragged()) {
                                                pending = Some(PendingPropertyEdit::UpdateGeometryHeight {
                                                    body_id: body_id.clone(),
                                                    height: height_mm * 1e-3,
                                                });
                                            }
```

with:

```rust
                                            if hr.drag_stopped() || (hr.changed() && !hr.dragged()) {
                                                pending = Some(PendingPropertyEdit::UpdateGeometryHeight {
                                                    body_id: body_id.clone(),
                                                    height: height_mm * 1e-3,
                                                });
                                            }
                                            }
```

Then replace:

```rust
                                            ui.horizontal(|ui| {
                                                if ui.button("Redraw").on_hover_text("Drag on canvas to redefine this body's geometry rectangle").clicked() {
                                                    pending = Some(PendingPropertyEdit::EnterDrawGeometryMode {
                                                        body_id: body_id.clone(),
                                                    });
                                                }
                                                if ui.button("Remove").on_hover_text("Remove the visual geometry rectangle from this body").clicked() {
                                                    pending = Some(PendingPropertyEdit::RemoveGeometry {
                                                        body_id: body_id.clone(),
                                                    });
                                                }
                                            });
                                        } else {
                                            if ui.button("Draw Geometry").on_hover_text("Drag on canvas to define a geometry rectangle for this body (required for force zones)").clicked() {
                                                pending = Some(PendingPropertyEdit::EnterDrawGeometryMode {
                                                    body_id: body_id.clone(),
                                                });
                                            }
                                        }
```

with:

```rust
                                            if ui.button("Remove").on_hover_text("Remove the visual geometry from this body").clicked() {
                                                pending = Some(PendingPropertyEdit::RemoveGeometry {
                                                    body_id: body_id.clone(),
                                                });
                                            }
                                        } else {
                                            // Add a shape sized from the link (decision R-2).
                                            ui.horizontal(|ui| {
                                                if ui.button("Add rectangle").on_hover_text("Give this body a rectangle as long as the link and a quarter as deep, centred on it (required for force zones)").clicked() {
                                                    pending = Some(PendingPropertyEdit::AddGeometry {
                                                        body_id: body_id.clone(),
                                                        shape: GeometryShape::Rectangle,
                                                    });
                                                }
                                                if ui.button("Add circle").on_hover_text("Give this body a circle half the link's length across, centred on it: a wheel (required for force zones)").clicked() {
                                                    pending = Some(PendingPropertyEdit::AddGeometry {
                                                        body_id: body_id.clone(),
                                                        shape: GeometryShape::Circle,
                                                    });
                                                }
                                            });
                                        }
```

Then add the import. In the same file, find the line `use crate::core::state::GROUND_ID;` with `grep -n "^use crate::core" linkage-sim-rs/src/gui/property_panel/mod.rs` and add directly after the first `use crate::core...` line:

```rust
use crate::core::body::GeometryShape;
```

(If the file already imports from `crate::core::body`, add `GeometryShape` to that import instead; `cargo check` must report no unused or duplicate import.)

- [ ] **Step 5: The canvas**

In `linkage-sim-rs/src/gui/canvas/rendering/mod.rs`, replace:

```rust
        // Draw body geometry rectangle (behind the link bar).
        if let Some(ref geo) = body.geometry {
            let pos = mech_state.get_position(body_id, q);
            let theta = mech_state.get_angle(body_id, q);
            let corners = crate::geometry::body_rect_to_world(
                pos.x, pos.y, theta, geo.width, geo.height, &geo.offset,
            );
            let screen_corners: Vec<Pos2> = corners
                .iter()
                .map(|c| {
                    let sp = view.world_to_screen(c.x, c.y);
                    Pos2::new(sp[0], sp[1])
                })
                .collect();
            let geo_fill = Color32::from_rgba_premultiplied(64, 42, 0, 64);
            let geo_stroke = Stroke::new(2.0, Color32::from_rgb(255, 165, 0));
            painter.add(egui::epaint::PathShape::convex_polygon(
                screen_corners,
                geo_fill,
                geo_stroke,
            ));
        }
```

with:

```rust
        // Draw body geometry (behind the link bar): a rectangle, or a circle.
        if let Some(ref geo) = body.geometry {
            let pos = mech_state.get_position(body_id, q);
            let theta = mech_state.get_angle(body_id, q);
            let geo_fill = Color32::from_rgba_premultiplied(64, 42, 0, 64);
            let geo_stroke = Stroke::new(2.0, Color32::from_rgb(255, 165, 0));
            if geo.shape == crate::core::body::GeometryShape::Circle {
                // A true circle: its radius on screen from a world-frame offset,
                // which the view's rotation (mounting angle) does not change.
                let c = geo.centre_world(pos.x, pos.y, theta);
                let sc = view.world_to_screen(c.x, c.y);
                let se = view.world_to_screen(c.x + geo.width / 2.0, c.y);
                let radius = ((se[0] - sc[0]).powi(2) + (se[1] - sc[1]).powi(2)).sqrt();
                painter.circle(Pos2::new(sc[0], sc[1]), radius, geo_fill, geo_stroke);
            } else {
                let screen_corners: Vec<Pos2> = geo
                    .outline_world(pos.x, pos.y, theta)
                    .iter()
                    .map(|c| {
                        let sp = view.world_to_screen(c.x, c.y);
                        Pos2::new(sp[0], sp[1])
                    })
                    .collect();
                painter.add(egui::epaint::PathShape::convex_polygon(
                    screen_corners,
                    geo_fill,
                    geo_stroke,
                ));
            }
        }
```

Then replace:

```rust
        if let Some(ref geo) = body.geometry {
            ui.label(format!(
                "Geometry: {:.1} \u{00d7} {:.1} mm",
                geo.width * 1e3,
                geo.height * 1e3
            ));
        }
```

with:

```rust
        if let Some(ref geo) = body.geometry {
            ui.label(match geo.shape {
                crate::core::body::GeometryShape::Rectangle => format!(
                    "Geometry: {:.1} \u{00d7} {:.1} mm",
                    geo.width * 1e3,
                    geo.height * 1e3
                ),
                crate::core::body::GeometryShape::Circle => {
                    format!("Geometry: circle, diameter {:.1} mm", geo.width * 1e3)
                }
            });
        }
```

- [ ] **Step 6: The force editor and the marker drop (decision R-4)**

In `linkage-sim-rs/src/gui/property_panel/force_editor.rs`, replace:

```rust
            // Application-point override: auto-centroid vs user-pinned.
            ui.separator();
            ui.label("Application point:");
            let mut locked = fz.body_local_app_point.is_some();
            if ui
                .checkbox(&mut locked, "Lock to body-local point")
                .on_hover_text(
                    "Unchecked: force applies at the overlap centroid (auto). Checked: force applies at a fixed point on the body — pin it to the real contact location.",
                )
                .changed()
            {
                let mut updated = fz.clone();
                updated.body_local_app_point = if locked {
                    // Seed the override with the current centroid so the
                    // force doesn't jump when locking.
                    fz.body_local_app_point.or(Some([0.0, 0.0]))
                } else {
                    None
                };
                *pending = Some(PendingPropertyEdit::UpdateForce {
                    index,
                    force: ForceElement::ForceZone(updated),
                });
            }

            if let Some(lp) = fz.body_local_app_point {
```

with:

```rust
            // Application point: overlap centre, a locked body point, or the
            // shape's contact point (decision R-4).
            ui.separator();
            ui.label("Application point:");
            let mode = fz.app_mode();
            let mut picked = mode;
            ui.horizontal(|ui| {
                ui.radio_value(&mut picked, ZoneAppMode::Centroid, "Overlap centre")
                    .on_hover_text("The force acts at the centroid of the shape's overlap with the zone, found again at every pose.");
                ui.radio_value(&mut picked, ZoneAppMode::Locked, "Locked point")
                    .on_hover_text("The force acts at a fixed point on the body: set it below, or drag the F marker on the canvas.");
                ui.radio_value(&mut picked, ZoneAppMode::Contact, "Contact point")
                    .on_hover_text("The force acts where a surface pushing in the force's direction meets the shape: its lowest point for an upward force. On a wheel it stays straight below the hub as the wheel turns.");
            });
            if picked != mode {
                // A first lock seeds the shape's centre, not the body origin.
                let seed = blueprint
                    .bodies
                    .get(&fz.body_id)
                    .and_then(|b| b.geometry.as_ref())
                    .map_or([0.0, 0.0], |g| [g.offset.x, g.offset.y]);
                *pending = Some(PendingPropertyEdit::UpdateForce {
                    index,
                    force: ForceElement::ForceZone(fz.with_app_mode(picked, seed)),
                });
            }

            if let (ZoneAppMode::Locked, Some(lp)) = (mode, fz.body_local_app_point) {
```

Then replace:

```rust
                if ui.button("Reset to auto (overlap centroid)")
                    .on_hover_text("Clear the locked application point and revert to the auto-computed overlap centroid each frame.")
                    .clicked()
                {
                    let mut updated = fz.clone();
                    updated.body_local_app_point = None;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::ForceZone(updated),
                    });
                }
                if ap_changed {
```

with:

```rust
                if ap_changed {
```

Then add the import: run `grep -n "^use crate::forces" linkage-sim-rs/src/gui/property_panel/force_editor.rs`; in that import add `ZoneAppMode` (or, if the file uses a glob `crate::forces::elements::*`, nothing to add). `cargo check` must report no unused or missing import.

In `linkage-sim-rs/src/gui/canvas/interaction.rs`, replace:

```rust
                    let mut updated = fz.clone();
                    updated.body_local_app_point = Some(local);
                    state.update_force_element(
```

with:

```rust
                    // Dropping the marker locks the point there, whatever the
                    // zone's mode was (decision R-4).
                    let mut updated = fz.with_app_mode(
                        crate::forces::elements::ZoneAppMode::Locked,
                        local,
                    );
                    updated.body_local_app_point = Some(local);
                    state.update_force_element(
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-shapes/linkage-sim-rs/Cargo.toml --lib geometry`
Expected: PASS (the four `pending_edits` tests and the three `geometry_drawing` tests, plus older tests matching "geometry").

Then the gate. Expected: `GATE PASS`; gate 1's count = Task 2's + 7. Restore `docs/chebyshev_lambda`.

- [ ] **Step 8: Commit**

```bash
cd C:/Users/Cole/source/repos/lsim-shapes
git add linkage-sim-rs/src/gui/canvas/rendering/mod.rs linkage-sim-rs/src/gui/property_panel/mod.rs linkage-sim-rs/src/gui/property_panel/pending_edits.rs linkage-sim-rs/src/gui/property_panel/force_editor.rs linkage-sim-rs/src/gui/canvas/interaction.rs linkage-sim-rs/src/gui/canvas/mod.rs
git commit -m "feat(linkage): circles and the contact point in the editor and on the canvas (decisions R-2, R-4)" -m "Add rectangle / Add circle, a shape switch and Diameter in the property panel; the force editor's Overlap centre / Locked point / Contact point; the canvas draws circles; dropping the F marker locks it." -m "Co-Authored-By: <model> <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01FSw7N8oV7SUVKTDVrGKLdg"
```

---

### Task 4: Docs, backlog, the press check and the browser check

**Files:**
- Modify: `docs/reference/FORCE_ELEMENTS.md:373-399`, `docs/FEATURES.md` (before `## Planned / Future`), `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`
- Create: `docs/superpowers/plans/2026-10-06-linkage-html-export-and-round-geometry.md` (this file, copied in)

- [ ] **Step 1: The force-element reference**

In `docs/reference/FORCE_ELEMENTS.md`, replace from the line `Spatial force field defined as an axis-aligned rectangular zone. Applies a distributed force proportional to the overlap area between the zone and the body's geometry.` through the line `- Returns zero if overlap area < \`1e-15\`.` with:

````markdown
Spatial force field defined as an axis-aligned rectangular zone. Applies its full force whenever the body's geometry overlaps the zone (binary: any overlap, full force).

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_id` | String | -- | Target body (must have `BodyGeometry` set) |
| `zone_min` | [f64; 2] | m | World-space bottom-left corner of zone |
| `zone_max` | [f64; 2] | m | World-space top-right corner of zone |
| `force` | [f64; 2] | N | Force vector while the geometry overlaps the zone |
| `label` | Option | -- | Optional display label |
| `body_local_app_point` | Option<[f64; 2]> | m | Locked application point, body-local |
| `at_contact_point` | bool | -- | Apply at the shape's contact point (schema 1.2.0; default false, left out of files when false) |

**Equation:**

```
overlap = polygon_clip(geometry_outline_world, zone_AABB)   (a circle is a 64-gon here)
F = force            if area(overlap) >= 1e-15, else 0
```

**Application point** (`force_zone_application`, shared by the force evaluation, the overlap ratio, the independent equilibrium check and the canvas):
- **Contact point** (`at_contact_point`): the shape's extreme point against the force, found at every pose: for an upward force the lowest point (a wheel's bottom, straight below its hub however the wheel turns); a level rectangle edge gives the edge's midpoint. Wins over a locked point.
- **Locked point** (`body_local_app_point`): a fixed point on the body.
- **Overlap centre** (neither): the centroid of the clipped overlap polygon.

The point is converted to body-local coordinates for the generalized force mapping.

**Special behavior:**
- Body must have `BodyGeometry`: a rectangle (`width`, `height`, `offset`) or a circle (`shape: "circle"`, diameter in `width`). Returns zero if geometry is missing.
- The outline is transformed to world space, then clipped against the axis-aligned zone using Sutherland-Hodgman polygon clipping.
- Returns zero if overlap area < `1e-15`, in every application mode.
````

- [ ] **Step 2: Features**

In `docs/FEATURES.md`, replace:

```markdown
## Planned / Future
```

with:

```markdown
### Schema v1.2.0: round shapes and the contact point

- Body geometry can be a circle (a wheel): "Add circle" / "Add rectangle" in the link's Geometry section, a Rectangle / Circle switch and a Diameter slider; circles draw as circles
- Force zones apply at the overlap centre, a locked point, or the shape's contact point (a wheel's bottom for an upward force, found at every pose); the F marker reads "F (contact)"
- Older builds still open schema 1.2.0 files (circles show as their bounding squares)

## Planned / Future
```

- [ ] **Step 3: docs/ai**

Run `grep -n "Linkage canvas: FORCE_ZONE_OVERLAP_FILL" docs/ai/02-system.yaml` and add, directly above that line (same indentation, two spaces then `- "`), the entry:

```yaml
  - "Linkage force zones: force_zone_application (forces::elements) is the one place that decides a zone's overlap (binary) and its application point (overlap centroid, locked point, or the shape's contact point: BodyGeometry::extreme_point_world against the force). The force evaluation, the overlap ratio, the independent equilibrium check (reactions.rs) and the canvas marker all call it; never rebuild a body's outline for a zone elsewhere. A circle is a CIRCLE_SEGMENTS (64)-gon in the overlap test, a true circle on the canvas, and exact (centre plus radius) for the contact point."
```

In `docs/ai/03-structure.yaml`, replace (part of one line):

```yaml
    responsibility: low-level 2D geometry helpers (polygon clipping, body_rect_to_world, centroid)
```

with:

```yaml
    responsibility: low-level 2D geometry helpers (polygon clipping, body_rect_to_world, centroid); a body's shape (rectangle or circle, core::body::GeometryShape) and its world outline and contact point are BodyGeometry methods (outline_world, centre_world, extreme_point_world)
```

In `docs/ai/04-memory.yaml`, append to the end of the file (same indentation as the file's other open items; run `tail -5 docs/ai/04-memory.yaml` first and match it):

```yaml
  - "DONE 2026-10-06 (Part A of docs/superpowers/plans/2026-10-06-linkage-html-export-and-round-geometry.md, decisions R-1 to R-7): circle body geometry and a force zone's contact point (schema 1.2.0). NEXT: Part B, the animated HTML export (decisions H-1 to H-8), its code written in that plan against main after Part A merges."
```

In `docs/ai/05-update-tracker.md`, replace:

```markdown
---

## 2026-10-05 — Weights draw over force zones (BL-042, branch fix/weights-over-force-zones)
```

with:

```markdown
---

## 2026-10-06 — Round shapes and the contact point (Part A, branch linkage/round-geometry)
- `BodyGeometry.shape`: rectangle (default, left out of files) or circle (diameter in width); schema 1.2.0.
- Force zones gain a third application mode, the shape's contact point (`at_contact_point`): its extreme point against the force, found at every pose, so a wheel's load acts at its bottom. One helper, `force_zone_application`, now decides overlap and point for the evaluation, the overlap ratio, the independent check and the canvas.
- Property panel: Add rectangle / Add circle, a shape switch, Diameter; the stub Draw Geometry / Redraw buttons removed (BL-044). Force editor: Overlap centre / Locked point / Contact point; dropping the F marker locks it.

## 2026-10-05 — Weights draw over force zones (BL-042, branch fix/weights-over-force-zones)
```

In `docs/ai/backlog.yaml`, append to the end of the file:

```yaml

- id: BL-043
  title: "Scale mechanism scales a force zone's box but not its locked application point"
  dimension: physics
  risk: mechanical
  evidence: "blueprint_ops.rs scale_mechanism (2026-10-06 map of the force-zone code): it multiplies zone_min/zone_max and the body geometry by the factor but leaves ForceZoneElement.body_local_app_point unscaled, so after a scale the locked force acts at the old distance from the body origin."
  acceptance: "scaling a mechanism by k scales body_local_app_point by k; a test scales a zone with a locked point and checks the point and the statics (driver torque scales by k for a force-only load)"
  priority: 3
  status: open
  notes: "Found while planning the round-geometry work (plan 2026-10-06, Part A); not changed there."

- id: BL-044
  title: "The canvas Draw Geometry tool is a stub"
  dimension: gui
  risk: mechanical
  evidence: "interaction.rs handle_draw_body_geometry is '#[allow(dead_code)] // Stubbed: full implementation pending'; EditorTool::DrawBodyGeometry does nothing. Until 2026-10-06 the property panel's Draw Geometry and Redraw buttons entered this mode, so they did nothing; Part A of the round-geometry plan replaced them with Add rectangle / Add circle (decision R-2). The stub, DrawBodyGeometryState and PendingPropertyEdit::EnterDrawGeometryMode remain."
  acceptance: "either the tool draws a rectangle or a circle by dragging (with a headless test), or the stub, its state and the edit variant are removed"
  priority: 4
  status: open
  notes: "Decision R-2 option (b) if the user wants dragging."
```

- [ ] **Step 4: Commit this plan**

```bash
cp C:/Users/Cole/source/repos/linkage_simulation/docs/superpowers/plans/2026-10-06-linkage-html-export-and-round-geometry.md C:/Users/Cole/source/repos/lsim-shapes/docs/superpowers/plans/
```

(The controller first records the user's decision answers in this file, Task 0 Step 6.)

- [ ] **Step 5: The press check (local, not committed; decision R-7)**

The controller builds the user's press with the wheel as a circle at the hub and the contact mode (from `press_w5.json` in the session scratchpad: set `bodies.link_2.geometry.shape` to `"circle"` and the force zone's `at_contact_point` to `true`), loads it through `AppState::load_from_json_str` in a throwaway test in a scratch copy of the worktree (never committed), and checks the actuator force: 21,506 N at crank 28 deg and 5,776 N at 65 deg (within 1 N), equal to `press_w5` at every sample. The scratch copy is deleted afterwards.

- [ ] **Step 6: Gate and the browser check**

Run the gate. Expected: `GATE PASS` with Task 3's count. Restore `docs/chebyshev_lambda`. Then build and serve the web bundle (`bash linkage-sim-rs/scripts/build_web.sh`, `bash linkage-sim-rs/scripts/serve_web.sh 8765`) and run the gui-smoke workflow by its script path (`.claude/workflows/gui-smoke.js`): zero console errors. Then open the press link from Step 5 on the local server and screenshot the canvas: the wheel is a circle and "F (contact)" sits at its bottom. Kill the server by PID.

- [ ] **Step 7: Commit**

```bash
cd C:/Users/Cole/source/repos/lsim-shapes
git add docs/reference/FORCE_ELEMENTS.md docs/FEATURES.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/ai/backlog.yaml docs/superpowers/plans/2026-10-06-linkage-html-export-and-round-geometry.md
git commit -m "docs(linkage): round shapes and the contact point; BL-043, BL-044; the plan" -m "Co-Authored-By: <model> <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01FSw7N8oV7SUVKTDVrGKLdg"
```

After Task 4: the whole-branch review (session model), one fix wave if needed, merge into local `main`, and ask the user before pushing. After the push, give the user the press link with the circle and the contact mode.

---

## Part B: animated HTML export (design; code written after Part A merges, decision H-8)

**What it makes:** File -> "Export animation (HTML)..." saves `mechanism_animation.html`: one self-contained page (decision H-6) that plays the current mechanism through its sweep, like the user's press animation, for any mechanism.

**Data, precomputed in Rust (no solver in JS):** a new module `linkage-sim-rs/src/gui/export/animation.rs` builds an `AnimationData` (serde `Serialize`) from the `AppState`:
- the sweep's samples (decision H-3): the same driver values `compute_sweep` uses (display angle via `driver_display_offset`, BL-041; stroke in mm for a linear drive), positions re-solved with the sweep's own `solve_sweep_positions` (made `pub(crate)`) so the branch matches the plots; unsolved samples left out (decision H-5);
- per sample: every moving body's attachment and mount points in the world frame, its geometry outline (rectangle corners or a circle's centre and radius), the ground pivots; force visuals (actuator endpoints and its force, each zone's application point from `force_zone_application` and its force, each weight's position and gravity force, joint reaction vectors from `solve_reactions_with_actuator`); readouts from `SweepData` at the same index (actuator force statics and inverse dynamics, or the driver torque without an actuator (decision H-2), actuator length, mechanical advantage, the weight breakdown's shares, reaction magnitudes);
- units (decision H-4): the app's units (N, N m, mm or m) with lbf beside every force.

**Page:** `linkage-sim-rs/src/gui/export/animation_template.html` (included with `include_str!`), its data placeholder replaced by the JSON; a small JS player adapted from the press animation: SVG view fitted to every sample, links, joints, ground markers, geometry (circles as circles), force arrows, weights, the driver-angle arc; the readout panel; a mini chart of the actuator force (or driver torque) with a moving marker; play/pause, bounce, speed, a scrub slider over the solved samples; checkboxes for forces, weights and the arc. No `http` reference anywhere in the page.

**Tasks (code to be written here against main after Part A merges):**
- B1: `animation.rs` data builder, test-first: sample count = solved samples; positions equal `coupler_traces`; actuator force equals `SweepData::actuator_forces` at each sample; the press-like wheel's contact point at its bottom; NaN-free JSON.
- B2: the template and `animation_html(&AppState) -> Result<String, String>`, test-first: the placeholder is replaced, the embedded JSON parses back, no `http`, the page names every body.
- B3: the File-menu item (native save dialog and web download through `export::download::download_text`, decision H-7), docs (README feature list, FEATURES.md, docs/ai), and a Playwright check of an exported file: loads with no console error, scrubbing changes the readout.

## Self-review record

- Spec coverage: circle shape (Task 1), contact point (Task 2), editor and canvas (Task 3), docs and checks (Task 4); export (Part B, staged by decision H-8).
- Placeholders: Part B carries no code by design (H-8); Part A's steps give exact code and commands.
- Type consistency: `GeometryShape`, `CIRCLE_SEGMENTS`, `BodyGeometry::{circle, centre_world, outline_world, extreme_point_world}`, `ZoneAppMode`, `ForceZoneElement::{app_mode, with_app_mode}`, `force_zone_application`, `ZoneApplication { overlap, active, point, mode }`, `PendingPropertyEdit::{AddGeometry { body_id, shape }, SetGeometryShape, UpdateGeometryDiameter}` are named the same in every task.
- Review Focus: each of the five lines names its test and owning task.
- Dry run (2026-10-06, by the plan's writer): every Part A replacement block (37) applied in order to `f9df219` copies; with the scripted steps (the 9 test literals, the property panel's import) and the created test file, `cargo test --all` passed 1,058 tests, 0 failed. The two "append to the end of the file" steps (Task 1's 10 geometry tests, Task 2's 7 force-zone tests) were not part of that replay, so those tests were first compiled and run by the Task 1 and Task 2 implementers. Two of the plan's tests were corrected on the way: the marker drag takes an 8 px first step (egui reports the pointer's position when the drag starts, which must still be inside the 12 px hit radius), and the sweep comparison switches one state's zone between the two modes instead of comparing two separately loaded states (each body map iterates in its own order, so their solutions can differ by whole turns and in the last digits). The user's press with the wheel as a circle and the contact mode gives 21,506.0 N at crank 28 deg and 5,776.0 N at 65 deg, equal to the hub-locked model (worst relative difference 3.9e-14); not committed (decision R-7).
